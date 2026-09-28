"""
Receive a continuous stream of 1-second PCM audio chunks from the STM32 over
UART and play them back as they arrive.

Protocol (little-endian), matching Core/Src/audio.c:

  Stream header (sent once at startup):
      uint32  magic            0x4B575354  ('KWST')
      uint32  sample_rate_hz
      uint16  bits_per_sample
      uint16  num_channels
      uint32  chunk_size_bytes   (one 1-second chunk)

  Per-chunk header (sent before every 1-second chunk):
      uint32  magic            0x4B575344  ('KWSD')
      uint32  seq              capture count; a gap means the firmware missed chunks
      uint32  len
  followed by `len` bytes of raw PCM payload.

printf text from the firmware shares the same UART; it is skipped by
resyncing on the magic numbers.

Playback runs from a sounddevice callback fed by a queue, so the main thread
can keep reading the serial port without gaps. PREBUFFER_CHUNKS chunks are
queued before playback starts, to absorb UART burst timing and clock drift
between the board and the host sound card.

Usage:
    python receive_and_play.py /dev/tty.usbmodemXXXX

Requirements:
    pip install pyserial sounddevice numpy
"""

import argparse
import queue
import struct
import sys

import numpy as np
import serial
import sounddevice as sd

BAUD_RATE = 921600

PREBUFFER_CHUNKS = 2    # chunks to queue before starting playback

STREAM_HEADER_MAGIC = 0x4B575354
CHUNK_HEADER_MAGIC = 0x4B575344

STREAM_HEADER_FMT = "<IIHHI"   # magic, sample_rate, bits_per_sample, channels, chunk_size
STREAM_HEADER_LEN = struct.calcsize(STREAM_HEADER_FMT)

CHUNK_HEADER_FMT = "<III"      # magic, seq, len
CHUNK_HEADER_LEN = struct.calcsize(CHUNK_HEADER_FMT)


def read_exact(ser, n):
    """Read exactly n bytes, raising if the port stalls out."""
    data = bytearray()
    while len(data) < n:
        chunk = ser.read(n - len(data))
        if not chunk:
            raise TimeoutError(f"Serial read timed out, wanted {n} bytes, got {len(data)}")
        data += chunk
    return bytes(data)


def resync_to_magic(ser, magic_bytes):
    """Byte-by-byte scan until we see the given 4-byte magic, in case framing
    was ever lost (dropped/garbled byte on the wire, or printf text)."""
    window = bytearray(4)
    while True:
        b = ser.read(1)
        if not b:
            raise TimeoutError("Serial read timed out while resyncing")
        window = window[1:] + bytearray(b)
        if bytes(window) == magic_bytes:
            return


def read_stream_header(ser):
    magic_bytes = struct.pack("<I", STREAM_HEADER_MAGIC)
    resync_to_magic(ser, magic_bytes)
    rest = read_exact(ser, STREAM_HEADER_LEN - 4)
    magic, sample_rate, bits_per_sample, channels, chunk_size = struct.unpack(
        STREAM_HEADER_FMT, magic_bytes + rest
    )
    print(
        f"Stream header: {sample_rate} Hz, {bits_per_sample}-bit, "
        f"{channels} ch, {chunk_size}-byte chunks "
        f"({chunk_size / (sample_rate * (bits_per_sample // 8) * channels):.1f}s each)"
    )
    return sample_rate, bits_per_sample, channels, chunk_size


def read_next_chunk(ser, expected_len):
    magic_bytes = struct.pack("<I", CHUNK_HEADER_MAGIC)
    resync_to_magic(ser, magic_bytes)

    # print("Received chunk header magic...")

    rest = read_exact(ser, CHUNK_HEADER_LEN - 4)
    # print("Received chunk header body...")

    magic, seq, length = struct.unpack(CHUNK_HEADER_FMT, magic_bytes + rest)

    if length != expected_len:
        # Most likely a corrupted header; don't trust `length` to read the payload.
        print(f"[warn] chunk {seq}: unexpected length {length} (expected {expected_len}), skipping")
        return seq, None

    payload = read_exact(ser, length)
    # print(f"Received chunk data (len = {len(payload)})...")

    return seq, payload


class ChunkPlayer:
    """Feeds queued sample blocks to a sounddevice output callback,
    outputting silence on underrun."""

    def __init__(self, channels):
        self.channels = channels
        self.queue = queue.Queue()
        self.current = np.zeros((0, channels), dtype=np.int16)
        self.underruns = 0

    def callback(self, outdata, frames, time_info, status):
        filled = 0
        while filled < frames:
            if len(self.current) == 0:
                try:
                    self.current = self.queue.get_nowait()
                except queue.Empty:
                    outdata[filled:] = 0
                    self.underruns += 1
                    return
            n = min(frames - filled, len(self.current))
            outdata[filled:filled + n] = self.current[:n]
            self.current = self.current[n:]
            filled += n


def parse_args():
    parser = argparse.ArgumentParser(
        description="Receive 1-second PCM chunks from the STM32 over UART and play them."
    )
    parser.add_argument(
        "port",
        help="serial port of the board's ST-Link VCP (e.g. /dev/tty.usbmodem2121303 or /dev/ttyACM0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Opening {args.port} @ {BAUD_RATE} baud...")
    ser = serial.Serial(args.port, BAUD_RATE, timeout=None)

    sample_rate, bits_per_sample, channels, chunk_size = read_stream_header(ser)
    if bits_per_sample != 16:
        print(f"Only 16-bit PCM is supported by this script, got {bits_per_sample}-bit.")
        sys.exit(1)

    player = ChunkPlayer(channels)
    stream = sd.OutputStream(
        samplerate=sample_rate, channels=channels, dtype="int16", callback=player.callback
    )

    print("Waiting for chunks... (Ctrl-C to stop)")
    last_seq = None
    last_underruns = 0
    try:
        while True:
            seq, payload = read_next_chunk(ser, chunk_size)
            if payload is None:
                continue

            if last_seq is not None and seq != last_seq + 1:
                print(f"[warn] gap in sequence: {last_seq} -> {seq}")
            last_seq = seq

            samples = np.frombuffer(payload, dtype="<i2").reshape(-1, channels)
            player.queue.put(samples)

            if not stream.active and player.queue.qsize() >= PREBUFFER_CHUNKS:
                stream.start()
                print("Playback started.")

            if player.underruns != last_underruns:
                print(f"[warn] playback underrun (total {player.underruns})")
                last_underruns = player.underruns

            print(f"Chunk {seq}: {len(payload)} bytes, queued {player.queue.qsize()}")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        stream.stop()
        stream.close()
        ser.close()


if __name__ == "__main__":
    main()
