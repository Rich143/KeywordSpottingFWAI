"""
Convert an Audacity label export to the ground-truth CSV used by
view_results.py and score_results.py (and back again with --reverse).

Audacity labels (File > Export > Export Labels) are tab-separated:
    start_s <TAB> end_s <TAB> label
Lines starting with '\\' hold the spectral selection of the label above and
are skipped.

Truth CSV (same format as make_test_wav.py writes):
    start_s, end_s, label, file        ('file' left empty here)

Allowed labels: forward, backward, stop (keywords), ignore (audio that
counts neither as a hit nor as a false accept), plus the model's non-keyword
classes other and _background_noise_ (treated as negative audio by the scorer;
accepted so make_test_wav.py truth files round-trip).

Usage:
    python EvalScripts/audacity_labels_to_truth.py -i eval_01_speech_labels.txt          # -> eval_01_speech_truth.csv
    python EvalScripts/audacity_labels_to_truth.py -i eval_01_speech_truth.csv --reverse  # -> eval_01_speech_labels.txt
"""

import argparse
import csv
import pathlib
import sys

ALLOWED_LABELS = ["forward", "backward", "stop", "ignore", "other", "_background_noise_"]

LABELS_SUFFIX = "_labels"
TRUTH_SUFFIX = "_truth"


def default_output(input_path, reverse):
    """eval_01_labels.txt <-> eval_01_truth.csv"""
    stem = input_path.stem
    if reverse:
        base = stem[: -len(TRUTH_SUFFIX)] if stem.endswith(TRUTH_SUFFIX) else stem
        return input_path.with_name(base + LABELS_SUFFIX + ".txt")
    base = stem[: -len(LABELS_SUFFIX)] if stem.endswith(LABELS_SUFFIX) else stem
    return input_path.with_name(base + TRUTH_SUFFIX + ".csv")


def read_audacity_labels(path):
    """Returns a sorted list of (start_s, end_s, label). Exits on bad lines or unknown labels."""
    spans = []
    errors = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\r\n")
            if not line.strip() or line.startswith("\\"):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                errors.append(f"line {line_no}: expected 'start<TAB>end<TAB>label', got {line!r}")
                continue
            try:
                start_s, end_s = float(fields[0]), float(fields[1])
            except ValueError:
                errors.append(f"line {line_no}: bad time in {line!r}")
                continue
            label = fields[2].strip().lower()
            if label not in ALLOWED_LABELS:
                errors.append(f"line {line_no}: unknown label {fields[2]!r} (allowed: {', '.join(ALLOWED_LABELS)})")
                continue
            spans.append((start_s, end_s, label))

    if errors:
        sys.exit(f"{path}:\n  " + "\n  ".join(errors))

    spans.sort()
    for i, (start_s, end_s, label) in enumerate(spans):
        if end_s <= start_s:
            print(f"[warn] zero-length span at {start_s:.3f}s ({label}): select the word, not a point")
        if i > 0 and start_s < spans[i - 1][1]:
            print(f"[warn] span at {start_s:.3f}s ({label}) overlaps the previous one ({spans[i - 1][2]})")
    return spans


def read_truth(path):
    with open(path, newline="") as f:
        return [(float(r["start_s"]), float(r["end_s"]), r["label"]) for r in csv.DictReader(f)]


def write_truth(path, spans):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_s", "end_s", "label", "file"])
        for start_s, end_s, label in spans:
            writer.writerow([f"{start_s:.4f}", f"{end_s:.4f}", label, ""])


def write_audacity_labels(path, spans):
    with open(path, "w") as f:
        for start_s, end_s, label in spans:
            f.write(f"{start_s:.6f}\t{end_s:.6f}\t{label}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Audacity labels to a truth CSV (or a truth CSV to Audacity labels with --reverse)."
    )
    parser.add_argument("-i", "--input", required=True, help="Audacity label .txt (or truth .csv with --reverse)")
    parser.add_argument(
        "-o", "--output",
        help="output path (default: <name>_labels.txt -> <name>_truth.csv, or the reverse)",
    )
    parser.add_argument("--reverse", action="store_true", help="convert a truth CSV to Audacity labels")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output) if args.output else default_output(input_path, args.reverse)

    if args.reverse:
        spans = read_truth(input_path)
        write_audacity_labels(output_path, spans)
    else:
        spans = read_audacity_labels(input_path)
        write_truth(output_path, spans)

    counts = {}
    for _, _, label in spans:
        counts[label] = counts.get(label, 0) + 1
    print(f"Wrote {output_path}: {len(spans)} spans " + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
