"""
Build a long test WAV from random Speech Commands clips, with a ground-truth
CSV, for checking the KWS model with run_model_on_wav.py and view_results.py.

Clips are drawn uniformly across the model's 5 classes and separated by random
silence gaps:
  forward / backward / stop   from dataset/<keyword>/
  _background_noise_          from dataset/_background_noise_split_/ (1 s clips)
  other                       from a random non-keyword word dir

Note: TrainModel.ipynb makes its own train/val/test split, so clips picked here
may have been in the training set. Treat results as an optimistic sanity check,
not an accuracy measurement.

Outputs:
  <output>.wav          16 kHz, 16-bit, mono
  <output>_truth.csv    start_s, end_s, label, file   (one row per clip)

Usage:
    python make_test_wav.py -o sc_test.wav
    python make_test_wav.py -o sc_test.wav -n 20 --gap 0.3 1.0 --seed 3
"""

import argparse
import csv
import pathlib
import sys
import wave

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset"

SAMPLE_RATE_HZ = 16000

KEYWORDS = ["forward", "backward", "stop"]
BACKGROUND_NOISE_LABEL = "_background_noise_"
BACKGROUND_NOISE_DIR = "_background_noise_split_"
OTHER_LABEL = "other"
CLASSES = KEYWORDS + [BACKGROUND_NOISE_LABEL, OTHER_LABEL]

# Dataset dirs that are not 'other' words
NON_OTHER_DIRS = set(KEYWORDS) | {BACKGROUND_NOISE_LABEL, BACKGROUND_NOISE_DIR}


def read_clip(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getsampwidth(), wav.getnchannels()) != (SAMPLE_RATE_HZ, 2, 1):
            raise ValueError(f"{path}: expected 16 kHz 16-bit mono")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")


def list_class_files(dataset_dir):
    """Returns {label: [wav paths]} for the keyword and noise classes, plus
    {word: [wav paths]} for the 'other' word dirs."""
    class_files = {kw: sorted((dataset_dir / kw).glob("*.wav")) for kw in KEYWORDS}
    class_files[BACKGROUND_NOISE_LABEL] = sorted((dataset_dir / BACKGROUND_NOISE_DIR).glob("*.wav"))

    other_words = {}
    for d in sorted(dataset_dir.iterdir()):
        if d.is_dir() and d.name not in NON_OTHER_DIRS:
            files = sorted(d.glob("*.wav"))
            if files:
                other_words[d.name] = files

    for label, files in class_files.items():
        if not files:
            sys.exit(f"no .wav files found for class '{label}' in {dataset_dir}")
    if not other_words:
        sys.exit(f"no 'other' word dirs found in {dataset_dir}")
    return class_files, other_words


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stitch random Speech Commands clips into a test WAV with a ground-truth CSV."
    )
    parser.add_argument("-o", "--output", required=True, help="output .wav path")
    parser.add_argument("-n", "--num-clips", type=int, default=12, help="number of clips (default: 12)")
    parser.add_argument(
        "--gap", type=float, nargs=2, metavar=("MIN", "MAX"), default=[0.5, 1.5],
        help="random silence between clips, in seconds (default: 0.5 1.5)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--dataset", default=str(DEFAULT_DATASET_DIR),
        help=f"Speech Commands dataset dir (default: {DEFAULT_DATASET_DIR})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gap_min, gap_max = args.gap
    if not 0 <= gap_min <= gap_max:
        sys.exit("--gap needs 0 <= MIN <= MAX")

    dataset_dir = pathlib.Path(args.dataset)
    class_files, other_words = list_class_files(dataset_dir)
    rng = np.random.default_rng(args.seed)

    def random_gap():
        return np.zeros(round(rng.uniform(gap_min, gap_max) * SAMPLE_RATE_HZ), dtype=np.int16)

    pieces = [random_gap()]
    pos = len(pieces[0])
    truth = []

    for _ in range(args.num_clips):
        label = CLASSES[rng.integers(len(CLASSES))]
        if label == OTHER_LABEL:
            words = sorted(other_words)
            files = other_words[words[rng.integers(len(words))]]
        else:
            files = class_files[label]
        path = files[rng.integers(len(files))]

        clip = read_clip(path)
        truth.append((pos / SAMPLE_RATE_HZ, (pos + len(clip)) / SAMPLE_RATE_HZ, label,
                      str(path.relative_to(dataset_dir))))
        gap = random_gap()
        pieces += [clip, gap]
        pos += len(clip) + len(gap)

    signal = np.concatenate(pieces)

    wav_path = pathlib.Path(args.output)
    truth_path = wav_path.with_name(wav_path.stem + "_truth.csv")

    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE_HZ)
        wav.writeframes(signal.astype("<i2").tobytes())

    with open(truth_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_s", "end_s", "label", "file"])
        for start_s, end_s, label, file in truth:
            writer.writerow([f"{start_s:.4f}", f"{end_s:.4f}", label, file])

    print(f"Wrote {wav_path} ({len(signal) / SAMPLE_RATE_HZ:.2f}s, {args.num_clips} clips)")
    print(f"Wrote {truth_path}")
    for start_s, end_s, label, file in truth:
        print(f"  {start_s:7.2f} - {end_s:7.2f}s  {label:<20} {file}")

    print("\nNext:")
    print(f"  python {SCRIPT_DIR / 'run_model_on_wav.py'} -m <model.tflite> -w {wav_path}")
    print(f"  python {SCRIPT_DIR / 'view_results.py'} -w {wav_path}")


if __name__ == "__main__":
    main()
