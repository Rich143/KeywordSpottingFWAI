"""
Receive a 1-second-chunked PCM audio stream from the STM32 over UART and
play it on the local speakers as it arrives.

Protocol (little-endian), matching audio_record.c:

  Stream header (sent once at startup):
      uint32  magic            0x4B575354  ('KWST')
      uint32  sample_rate_hz
      uint16  bits_per_sample
      uint16  num_channels
      uint32  chunk_size_bytes

  Per-chunk header (sent before every chunk):
      uint32  magic            0x4B575344  ('KWSD')
      uint32  seq
      uint32  len
  followed by `len` bytes of raw PCM payload.

Requirements:
    pip install pyserial sounddevice numpy
"""

import struct
import sys

import numpy as np
import serial
import sounddevice as sd

SERIAL_PORT = "/dev/tty.usbmodem2121303"       # CHANGE to your board's serial port (e.g. "/dev/ttyACM0")
BAUD_RATE = 921600

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
    was ever lost (dropped/garbled byte on the wire)."""
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
        f"{channels} ch, {chunk_size}-byte chunks"
    )
    return sample_rate, bits_per_sample, channels, chunk_size


def read_next_chunk(ser, expected_len):
    magic_bytes = struct.pack("<I", CHUNK_HEADER_MAGIC)
    resync_to_magic(ser, magic_bytes)
    rest = read_exact(ser, CHUNK_HEADER_LEN - 4)
    magic, seq, length = struct.unpack(CHUNK_HEADER_FMT, magic_bytes + rest)

    if length != expected_len:
        print(f"[warn] chunk {seq}: unexpected length {length} (expected {expected_len})")

    payload = read_exact(ser, length)
    return seq, payload


def main():
    print(f"Opening {SERIAL_PORT} @ {BAUD_RATE} baud...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=None)

    sample_rate, bits_per_sample, channels, chunk_size = read_stream_header(ser)
    if bits_per_sample != 16:
        print(f"Only 16-bit PCM is supported by this script, got {bits_per_sample}-bit.")
        sys.exit(1)

    stream = sd.OutputStream(samplerate=sample_rate, channels=channels, dtype="int16")
    stream.start()

    print("Playing... (Ctrl-C to stop)")
    last_seq = None
    try:
        while True:
            seq, payload = read_next_chunk(ser, chunk_size)
            print("Read chunk:", seq)

            if last_seq is not None and seq != last_seq + 1:
                print(f"[warn] gap in sequence: {last_seq} -> {seq}")
            last_seq = seq

            samples = np.frombuffer(payload, dtype="<i2").reshape(-1, channels)
            print("Writing audio to portaudio stream...")
            stream.write(samples)  # blocks until portaudio has room - keeps pace with playback
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        stream.stop()
        stream.close()
        ser.close()


if __name__ == "__main__":
    main()
