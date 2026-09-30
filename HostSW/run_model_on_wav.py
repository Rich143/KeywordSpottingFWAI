"""
Run the KWS tflite model over a WAV recording with a sliding 1-second window,
and save the per-window results to CSV.

Intended for recordings made with receive_and_play.py -o, to check how the
trained model performs on real board-mic audio before running it on target.

Pipeline per window (matches TrainModel.ipynb):
  int16 PCM -> float32 in [-1, 1)  (same scaling as tf.audio.decode_wav)
  -> log-mel spectrogram (30 x 45) from the C preprocessing library via ctypes
  -> quantise with the model's input scale / zero point (int8 models only)
  -> tflite model -> dequantised softmax -> argmax label

CSV columns:
  window, start_s, end_s, label_idx, label, p_<label>..., clipped
start_s / end_s are the window's position in seconds from the start of the
WAV. `clipped` is the number of spectrogram cells that saturated when
quantised to int8 (a sign of a level mismatch with the training data).

Usage:
    python run_model_on_wav.py -m tflite_models/kws_v7_int8_softmax.tflite -w capture.wav
    python run_model_on_wav.py -m tflite_models/kws_v7_int8_softmax.tflite -w capture.wav -s 0.25 -o results.csv

Requirements:
    pip install numpy ai-edge-litert
    and the host preprocessing library built from HostSW/CMakeLists.txt
"""

import argparse
import csv
import ctypes
import pathlib
import sys
import wave
from collections import Counter

import numpy as np
from ai_edge_litert.interpreter import Interpreter

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_LIB_PATH = SCRIPT_DIR / "build" / "audioproc-build" / "libAudioPreprocessing.dylib"

# Model output order, from TrainModel.ipynb (DESIRED_KEYWORDS + ['other'])
LABELS = ["forward", "backward", "stop", "_background_noise_", "other"]

# --- Constants matching the C header (see TrainModel.ipynb) ---
AUDIO_SPECTROGRAM_FRAME_LEN = 512
AUDIO_SAMPLE_RATE_HZ = 16000
AUDIO_SPECTROGRAM_NMELS = 30
AUDIO_SPECTROGRAM_ROWS = AUDIO_SPECTROGRAM_NMELS
AUDIO_SPECTROGRAM_STRIDE_LEN = 352  # 22 ms at 16000 Hz
AUDIO_SPECTROGRAM_COLS = 45
WINDOW_LEN = AUDIO_SAMPLE_RATE_HZ   # 1 second

STATUS_OK = 0


class AudioPreprocessingError(RuntimeError):
    pass


class AudioPreprocessor:
    """ctypes wrapper around the C preprocessing library.

    Copied from TrainModel.ipynb, which is the source of truth: keep the two
    in sync so features here match the ones the model was trained on.
    NOT thread-safe (the C library uses static state)."""

    def __init__(self, lib_path):
        self.lib = ctypes.CDLL(str(lib_path))
        self._declare_signatures()
        status = self.lib.audio_preprocessing_init()
        if status != STATUS_OK:
            raise AudioPreprocessingError(f"init failed with status {status}")

    def _declare_signatures(self):
        self.lib.audio_preprocessing_init.argtypes = []
        self.lib.audio_preprocessing_init.restype = ctypes.c_int

        self.lib.audio_preprocessing_process_frame.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self.lib.audio_preprocessing_process_frame.restype = ctypes.c_int

        self.lib.audio_preprocessing_get_spectrogram.argtypes = []
        self.lib.audio_preprocessing_get_spectrogram.restype = ctypes.POINTER(ctypes.c_float)

        self.lib.audio_preprocessing_get_spectrogram_len.argtypes = []
        self.lib.audio_preprocessing_get_spectrogram_len.restype = ctypes.c_uint32

        self.lib.audio_preprocessing_get_spectrogram_filled_cols.argtypes = []
        self.lib.audio_preprocessing_get_spectrogram_filled_cols.restype = ctypes.c_uint32

        self.lib.audio_preprocessing_clear_spectrogram.argtypes = []
        self.lib.audio_preprocessing_clear_spectrogram.restype = ctypes.c_int

    def compute_spectrogram(self, signal):
        """signal: 1 second of float32 audio (16000 samples).
        Returns an np.ndarray of shape (AUDIO_SPECTROGRAM_ROWS, AUDIO_SPECTROGRAM_COLS)."""
        signal = np.array(signal, dtype=np.float32, copy=True)

        n_frames = (len(signal) - AUDIO_SPECTROGRAM_FRAME_LEN) // AUDIO_SPECTROGRAM_STRIDE_LEN + 1
        if n_frames != AUDIO_SPECTROGRAM_COLS:
            raise ValueError(
                f"signal length {len(signal)} produces {n_frames} frames, "
                f"expected exactly {AUDIO_SPECTROGRAM_COLS}"
            )

        self.lib.audio_preprocessing_clear_spectrogram()

        for i in range(n_frames):
            start = i * AUDIO_SPECTROGRAM_STRIDE_LEN
            # process_frame modifies its input buffer, so pass a copy rather
            # than a view into `signal` (frames overlap).
            frame = np.array(signal[start : start + AUDIO_SPECTROGRAM_FRAME_LEN], copy=True)
            frame_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            status = self.lib.audio_preprocessing_process_frame(frame_ptr)
            if status != STATUS_OK:
                raise AudioPreprocessingError(f"process_frame failed at frame {i} with status {status}")

        filled = self.lib.audio_preprocessing_get_spectrogram_filled_cols()
        if filled != AUDIO_SPECTROGRAM_COLS:
            raise AudioPreprocessingError(
                f"expected {AUDIO_SPECTROGRAM_COLS} filled columns, got {filled}"
            )

        spec_ptr = self.lib.audio_preprocessing_get_spectrogram()
        spec_len = self.lib.audio_preprocessing_get_spectrogram_len()
        flat = np.ctypeslib.as_array(spec_ptr, shape=(spec_len,)).copy()

        return flat.reshape(AUDIO_SPECTROGRAM_COLS, AUDIO_SPECTROGRAM_ROWS).T


class KwsModel:
    """tflite interpreter plus the input quantisation / output dequantisation.
    Works with both the int8 and float32 exports."""

    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        in_shape = tuple(self.input_details["shape"])
        expected_shape = (1, AUDIO_SPECTROGRAM_ROWS, AUDIO_SPECTROGRAM_COLS)
        if in_shape != expected_shape:
            raise ValueError(f"model input shape {in_shape}, expected {expected_shape}")
        n_out = int(np.prod(self.output_details["shape"]))
        if n_out != len(LABELS):
            raise ValueError(f"model has {n_out} outputs, expected {len(LABELS)} ({LABELS})")

        self.in_dtype = self.input_details["dtype"]
        self.in_scale, self.in_zero_point = self.input_details["quantization"]
        self.out_scale, self.out_zero_point = self.output_details["quantization"]
        self.is_quantised = np.issubdtype(self.in_dtype, np.integer)

    def describe(self):
        if not self.is_quantised:
            return f"float model, input {self.in_dtype.__name__}"
        return (
            f"input {self.in_dtype.__name__} scale {self.in_scale:.6g} zp {self.in_zero_point}, "
            f"output scale {self.out_scale:.6g} zp {self.out_zero_point}"
        )

    def quantise(self, spec):
        """Returns (model input tensor, number of clipped cells)."""
        if not self.is_quantised:
            return spec.astype(self.in_dtype)[np.newaxis, ...], 0

        info = np.iinfo(self.in_dtype)
        q = np.round(spec / self.in_scale + self.in_zero_point)
        clipped = int(np.count_nonzero((q < info.min) | (q > info.max)))
        q = np.clip(q, info.min, info.max).astype(self.in_dtype)
        return q[np.newaxis, ...], clipped

    def predict(self, x):
        """Runs the model on a quantised input, returns float softmax probabilities."""
        self.interpreter.set_tensor(self.input_details["index"], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details["index"]).reshape(-1)
        if self.out_scale != 0:
            return (out.astype(np.float32) - self.out_zero_point) * self.out_scale
        return out.astype(np.float32)


def load_wav(path):
    """Loads a 16 kHz 16-bit WAV as float32 in [-1, 1), matching tf.audio.decode_wav.
    Multi-channel files use channel 0."""
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        channels = wav.getnchannels()
        raw = wav.readframes(wav.getnframes())

    if sample_width != 2:
        sys.exit(f"{path}: only 16-bit PCM is supported, got {sample_width * 8}-bit")
    if sample_rate != AUDIO_SAMPLE_RATE_HZ:
        sys.exit(f"{path}: sample rate must be {AUDIO_SAMPLE_RATE_HZ} Hz, got {sample_rate} Hz")

    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)[:, 0]
    if channels > 1:
        print(f"[info] {channels}-channel WAV, using channel 0")
    return samples.astype(np.float32) / 32768.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the KWS tflite model over a WAV file with a sliding 1-second window."
    )
    parser.add_argument("-m", "--model", required=True, help="path to the .tflite model")
    parser.add_argument("-w", "--wav", required=True, help="path to the 16 kHz 16-bit .wav recording")
    parser.add_argument(
        "-s", "--stride", type=float, default=0.1,
        help="window stride in seconds (default: 0.1)",
    )
    parser.add_argument(
        "-o", "--output", metavar="CSV_PATH",
        help="output CSV path (default: the WAV path with a .csv suffix)",
    )
    parser.add_argument(
        "--lib", default=str(DEFAULT_LIB_PATH),
        help=f"path to the preprocessing dynamic library (default: {DEFAULT_LIB_PATH})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    stride_samples = round(args.stride * AUDIO_SAMPLE_RATE_HZ)
    if stride_samples < 1:
        sys.exit(f"stride {args.stride}s is less than one sample")

    output_path = pathlib.Path(args.output) if args.output else pathlib.Path(args.wav).with_suffix(".csv")

    model = KwsModel(args.model)
    print(f"Model {args.model}: {model.describe()}")

    preproc = AudioPreprocessor(args.lib)

    signal = load_wav(args.wav)
    duration_s = len(signal) / AUDIO_SAMPLE_RATE_HZ
    if len(signal) < WINDOW_LEN:
        sys.exit(f"{args.wav}: {duration_s:.3f}s long, need at least 1s")

    starts = range(0, len(signal) - WINDOW_LEN + 1, stride_samples)
    print(
        f"WAV {args.wav}: {duration_s:.2f}s, {len(starts)} windows "
        f"(stride {stride_samples / AUDIO_SAMPLE_RATE_HZ:.4f}s)"
    )

    label_counts = Counter()
    total_clipped = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["window", "start_s", "end_s", "label_idx", "label"]
            + [f"p_{label}" for label in LABELS]
            + ["clipped"]
        )

        for window_idx, start in enumerate(starts):
            spec = preproc.compute_spectrogram(signal[start : start + WINDOW_LEN])
            x, clipped = model.quantise(spec)
            probs = model.predict(x)
            label_idx = int(np.argmax(probs))

            label_counts[LABELS[label_idx]] += 1
            total_clipped += clipped

            writer.writerow(
                [
                    window_idx,
                    f"{start / AUDIO_SAMPLE_RATE_HZ:.4f}",
                    f"{(start + WINDOW_LEN) / AUDIO_SAMPLE_RATE_HZ:.4f}",
                    label_idx,
                    LABELS[label_idx],
                ]
                + [f"{p:.4f}" for p in probs]
                + [clipped]
            )

    print(f"Saved results to {output_path}")
    print("Predicted label counts:")
    for label in LABELS:
        print(f"  {label:<20} {label_counts[label]}")

    if total_clipped:
        total_cells = len(starts) * AUDIO_SPECTROGRAM_ROWS * AUDIO_SPECTROGRAM_COLS
        print(
            f"[warn] {total_clipped} of {total_cells} spectrogram cells "
            f"({100 * total_clipped / total_cells:.2f}%) clipped when quantising the model input"
        )


if __name__ == "__main__":
    main()
