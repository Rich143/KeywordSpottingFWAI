"""
Score one model on every labelled board recording in eval_recordings/ and
append the results to a history CSV, so each new model version can be
compared with the last.

For each eval_recordings/<stem>.wav that has labels:
  1. if <stem>_labels.txt is newer than <stem>_truth.csv (or there is no truth
     yet), convert it (same as audacity_labels_to_truth.py)
  2. run_model_on_wav.py with the given model and stride  -> <stem>.csv
  3. score with score_results.py's detect() / score(), using the given settings

Recordings listed in eval_recordings/eval_ignore.txt are skipped: one WAV
filename (or glob, e.g. eval_0[12]_*.wav) per line, '#' starts a comment.

History CSV (default eval_recordings/eval_history.csv, committed): one row per
recording plus one ALL row per snapshot. The ALL row pools the recordings:
total hits / total keyword spans, and total false accepts / total minutes of
negative audio. Re-running the same model version with the same settings is
refused unless --replace is given, which drops the old rows first.

Usage:
    python snapshot_eval.py -v v7 -m tflite_models/kws_v7_int8_softmax.tflite
    python snapshot_eval.py -v v8 -m tflite_models/kws_v8_int8_softmax.tflite --notes "noise aug + LibriSpeech negatives"
    python snapshot_eval.py -v v7 -m tflite_models/kws_v7_int8_softmax.tflite --threshold 0.6 --avg 1.0
"""

import argparse
import csv
import datetime
import fnmatch
import hashlib
import pathlib
import subprocess
import sys

import audacity_labels_to_truth as labels_conv
import score_results as sr

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_EVAL_DIR = SCRIPT_DIR / "eval_recordings"
IGNORE_FILE_NAME = "eval_ignore.txt"
HISTORY_FILE_NAME = "eval_history.csv"
DEFAULT_STRIDE_S = 0.1
ALL_ROW = "ALL"

# Columns that identify a snapshot: same values = same snapshot
SNAPSHOT_KEY = ["model_version", "stride_s", "avg_s", "threshold", "suppress_s", "tolerance_s"]

HISTORY_FIELDS = [
    "timestamp", "model_version", "model_path", "model_sha256",
    "stride_s", "avg_s", "threshold", "suppress_s", "tolerance_s",
    "recording", "duration_min", "negative_min",
    "keyword_spans", "hits", "recall",
] + [f"recall_{kw}" for kw in sr.KEYWORDS] + [
    "false_accepts", "substitutions", "duplicates", "ignored",
    "fa_per_min", "fa_per_hour", "notes",
]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()[:12]


def display_path(path):
    """Path relative to HostSW when possible, so the history CSV doesn't depend on where the repo is checked out."""
    try:
        return str(path.resolve().relative_to(SCRIPT_DIR))
    except ValueError:
        return str(path)


def read_ignore_patterns(eval_dir):
    path = eval_dir / IGNORE_FILE_NAME
    if not path.exists():
        return []
    patterns = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            patterns.append(line)
    return patterns


def find_recordings(eval_dir, ignore_patterns):
    """Returns (labelled WAVs to score, skipped WAVs with reasons)."""
    recordings, skipped = [], []
    for wav in sorted(eval_dir.glob("*.wav")):
        labels = wav.with_name(wav.stem + labels_conv.LABELS_SUFFIX + ".txt")
        truth = wav.with_name(wav.stem + labels_conv.TRUTH_SUFFIX + ".csv")
        if any(fnmatch.fnmatch(wav.name, p) for p in ignore_patterns):
            skipped.append((wav, f"listed in {IGNORE_FILE_NAME}"))
        elif not labels.exists() and not truth.exists():
            skipped.append((wav, "not labelled"))
        else:
            recordings.append(wav)
    return recordings, skipped


def update_truth(wav):
    """Converts <stem>_labels.txt to <stem>_truth.csv if the labels are newer.
    Returns the truth path."""
    labels = wav.with_name(wav.stem + labels_conv.LABELS_SUFFIX + ".txt")
    truth = wav.with_name(wav.stem + labels_conv.TRUTH_SUFFIX + ".csv")
    if labels.exists() and (not truth.exists() or labels.stat().st_mtime > truth.stat().st_mtime):
        labels_conv.write_truth(truth, labels_conv.read_audacity_labels(labels))
        print(f"  converted {labels.name} -> {truth.name}")
    return truth


def run_model(model_path, wav, stride_s):
    results = wav.with_suffix(".csv")
    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_model_on_wav.py"),
        "-m", str(model_path), "-w", str(wav), "-o", str(results), "-s", str(stride_s),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"run_model_on_wav.py failed on {wav.name}:\n{proc.stdout}{proc.stderr}")
    return results


def score_recording(results_path, truth_path, args, tolerance_s):
    """Returns a dict of the per-recording metric columns."""
    labels, _, end_s, probs = sr.load_results(results_path)
    keyword_spans, ignore_spans = sr.load_truth(truth_path)
    duration_s = float(end_s.max())

    detections = sr.detect(labels, end_s, probs, args.avg, args.threshold, args.suppress)
    hit, _, n_false, _, neg_min = sr.score(detections, keyword_spans, ignore_spans, tolerance_s, duration_s)

    row = {
        "duration_min": duration_s / 60,
        "negative_min": neg_min,
        "keyword_spans": len(keyword_spans),
        "hits": sum(hit),
        "false_accepts": n_false,
    }
    for kw in sr.KEYWORDS:
        idx = [j for j, s in enumerate(keyword_spans) if s[2] == kw]
        row[f"hits_{kw}"] = sum(hit[j] for j in idx)
        row[f"spans_{kw}"] = len(idx)
    for outcome, col in (("substitution", "substitutions"), ("duplicate", "duplicates"), ("ignored", "ignored")):
        row[col] = sum(d.outcome == outcome for d in detections)
    return row


def finish_row(counts):
    """Adds the derived rates to a dict of summed counts."""
    def ratio(a, b):
        return a / b if b else float("nan")

    row = dict(counts)
    row["recall"] = ratio(counts["hits"], counts["keyword_spans"])
    for kw in sr.KEYWORDS:
        row[f"recall_{kw}"] = ratio(counts[f"hits_{kw}"], counts[f"spans_{kw}"])
    row["fa_per_min"] = ratio(counts["false_accepts"], counts["negative_min"])
    row["fa_per_hour"] = 60 * row["fa_per_min"]
    return row


def format_row(row):
    out = {}
    for k in HISTORY_FIELDS:
        v = row.get(k, "")
        if isinstance(v, float):
            v = f"{v:.4f}" if k.startswith(("recall", "fa_", "duration", "negative")) else f"{v:g}"
        out[k] = v
    return out


def read_history(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != HISTORY_FIELDS:
            sys.exit(f"{path}: columns don't match this script's (was it edited, or written by an older version?)")
        return list(reader)


def write_history(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def same_snapshot(row, key):
    return all(row[k] == key[k] for k in SNAPSHOT_KEY)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score a model on all labelled eval recordings and append to the history CSV."
    )
    parser.add_argument("-v", "--version", required=True, help="model version label, e.g. v7")
    parser.add_argument("-m", "--model", required=True, help="tflite model path")
    parser.add_argument("-s", "--stride", type=float, default=DEFAULT_STRIDE_S,
                        help=f"window stride in seconds (default: {DEFAULT_STRIDE_S})")
    parser.add_argument("--avg", type=float, default=0.5, help="smoothing window, s (default: 0.5)")
    parser.add_argument("--threshold", type=float, default=0.7, help="detection threshold (default: 0.7)")
    parser.add_argument("--suppress", type=float, default=1.5, help="suppression time, s (default: 1.5)")
    parser.add_argument("--tolerance", type=float, help="match tolerance after a span, s (default: 1.0 + avg)")
    parser.add_argument("--notes", default="", help="free text stored with the snapshot")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR), help=f"recordings dir (default: {DEFAULT_EVAL_DIR})")
    parser.add_argument("--history", help=f"history CSV (default: <eval dir>/{HISTORY_FILE_NAME})")
    parser.add_argument("--replace", action="store_true",
                        help="replace an existing snapshot with the same version and settings")
    return parser.parse_args()


def main():
    args = parse_args()
    eval_dir = pathlib.Path(args.eval_dir)
    history_path = pathlib.Path(args.history) if args.history else eval_dir / HISTORY_FILE_NAME
    model_path = pathlib.Path(args.model)
    tolerance_s = args.tolerance if args.tolerance is not None else sr.WINDOW_LEN_S + args.avg

    if not model_path.exists():
        sys.exit(f"model {model_path} not found")

    snapshot = format_row({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_version": args.version,
        "model_path": display_path(model_path),
        "model_sha256": file_sha256(model_path),
        "stride_s": args.stride,
        "avg_s": args.avg,
        "threshold": args.threshold,
        "suppress_s": args.suppress,
        "tolerance_s": tolerance_s,
        "notes": args.notes,
    })

    history = read_history(history_path)
    if any(same_snapshot(r, snapshot) for r in history):
        if not args.replace:
            sys.exit(f"{history_path} already has a snapshot for {args.version} with these settings "
                     f"(use --replace to overwrite it)")
        history = [r for r in history if not same_snapshot(r, snapshot)]

    recordings, skipped = find_recordings(eval_dir, read_ignore_patterns(eval_dir))
    for wav, reason in skipped:
        print(f"Skipping {wav.name}: {reason}")
    if not recordings:
        sys.exit(f"no labelled recordings in {eval_dir}")

    count_keys = None
    totals = {}
    rows = []
    for wav in recordings:
        print(f"{wav.name}:")
        truth = update_truth(wav)
        results = run_model(model_path, wav, args.stride)
        counts = score_recording(results, truth, args, tolerance_s)
        count_keys = count_keys or list(counts)
        for k in count_keys:
            totals[k] = totals.get(k, 0) + counts[k]
        row = finish_row(counts)
        rows.append(format_row({**snapshot, **row, "recording": wav.name}))
        print(f"  recall {row['hits']}/{row['keyword_spans']} ({100 * row['recall']:.0f}%), "
              f"false accepts {row['false_accepts']} ({row['fa_per_min']:.2f}/min)")

    all_row = finish_row(totals)
    rows.append(format_row({**snapshot, **all_row, "recording": ALL_ROW}))

    previous = [r for r in history if r["recording"] == ALL_ROW]
    write_history(history_path, history + rows)

    print(f"\n{args.version} ({len(recordings)} recording{'s' if len(recordings) != 1 else ''}, avg {args.avg}s, threshold {args.threshold}, "
          f"suppress {args.suppress}s):")
    print(f"  recall {all_row['hits']}/{all_row['keyword_spans']} ({100 * all_row['recall']:.0f}%)  "
          + "  ".join(f"{kw} {100 * all_row[f'recall_{kw}']:.0f}%" for kw in sr.KEYWORDS))
    print(f"  false accepts {all_row['false_accepts']} in {all_row['negative_min']:.1f} min "
          f"= {all_row['fa_per_min']:.2f}/min ({all_row['fa_per_hour']:.0f}/hour)")
    if previous:
        print("\nPrevious snapshots (ALL rows; recordings may differ between snapshots):")
        for r in previous[-5:]:
            print(f"  {r['timestamp']}  {r['model_version']:<8} avg {r['avg_s']} thr {r['threshold']} "
                  f"sup {r['suppress_s']}  recall {100 * float(r['recall']):.0f}%  "
                  f"FA {float(r['fa_per_min']):.2f}/min  {r['notes']}")
    print(f"\nAppended {len(rows)} rows to {history_path}")


if __name__ == "__main__":
    main()
