"""
Score run_model_on_wav.py results against a ground-truth CSV the way a
streaming keyword spotter would be used: smooth the per-window outputs, fire
a detection when a keyword is confident enough, then suppress repeats.

Reports keyword recall and false accepts per minute of non-keyword audio,
the two numbers model / data changes are judged by.

Detection (detect(), deliberately simple, to be improved later):
  - causal moving average of the softmax vectors over the last --avg seconds
    of windows (at least min(3, N) windows, as in TF Micro's RecognizeCommands)
  - fire if the top averaged class is a keyword with p >= --threshold and at
    least --suppress seconds have passed since the last detection
  - the detection time is the end of the newest window: when the device would
    actually have heard that audio

Matching a detection at time t, label L, against truth spans (start, end):
  hit           first L detection with start <= t <= end + tolerance
                (the window must reach the word, and the word must still be in
                the window, plus the averaging lag; default tolerance 1 s + avg)
  duplicate     a further L detection matching an already-hit span
  ignored       matches an 'ignore' span (cut-off keyword, cough, TV...)
  substitution  matches a span of a different keyword
  false_accept  matches nothing
Duplicates, substitutions and false accepts all count as false accepts.
Negative time = audio duration minus the keyword and 'ignore' spans. Truth
labels 'other' / '_background_noise_' (make_test_wav.py) are negative audio.

Usage:
    python score_results.py -r eval_01_speech.csv                 # truth: eval_01_speech_truth.csv
    python score_results.py -r eval_01_speech.csv --sweep
    python score_results.py -r eval_01_speech.csv --avg 1.0 --threshold 0.6 --suppress 1.5
"""

import argparse
import csv
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

KEYWORDS = ["forward", "backward", "stop"]
IGNORE_LABEL = "ignore"
NEGATIVE_LABELS = ["other", "_background_noise_"]

WINDOW_LEN_S = 1.0
MAX_MIN_COUNT = 3   # RecognizeCommands' minimum_count
SWEEP_THRESHOLDS = np.arange(0.50, 0.951, 0.05)


@dataclass
class Detection:
    time_s: float
    label: str
    p: float
    outcome: str = ""


def load_results(path):
    """Returns (labels, start_s, end_s, probs)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        prob_cols = [c for c in reader.fieldnames if c.startswith("p_")]
        rows = list(reader)
    if not rows:
        sys.exit(f"{path}: no result rows")
    labels = [c[len("p_"):] for c in prob_cols]
    missing = [k for k in KEYWORDS if k not in labels]
    if missing:
        sys.exit(f"{path}: no probability columns for keywords {missing}")
    start_s = np.array([float(r["start_s"]) for r in rows])
    end_s = np.array([float(r["end_s"]) for r in rows])
    probs = np.array([[float(r[c]) for c in prob_cols] for r in rows])
    return labels, start_s, end_s, probs


def load_truth(path):
    """Returns (keyword spans, ignore spans), each a list of (start_s, end_s, label)."""
    keyword_spans, ignore_spans = [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            span = (float(r["start_s"]), float(r["end_s"]), r["label"])
            if span[2] in KEYWORDS:
                keyword_spans.append(span)
            elif span[2] == IGNORE_LABEL:
                ignore_spans.append(span)
            elif span[2] not in NEGATIVE_LABELS:
                print(f"[warn] {path}: unknown truth label '{span[2]}' at {span[0]:.2f}s, treated as negative audio")
    return sorted(keyword_spans), sorted(ignore_spans)


def detect(labels, end_s, probs, avg_s, threshold, suppress_s):
    """Smoothing + threshold + suppression over per-window softmax outputs.
    Returns a list of Detection in time order."""
    stride_s = float(np.median(np.diff(end_s))) if len(end_s) > 1 else WINDOW_LEN_S
    n_avg = max(1, round(avg_s / stride_s))
    min_count = min(MAX_MIN_COUNT, n_avg)
    keyword_idx = {labels.index(k) for k in KEYWORDS}

    detections = []
    last_time = -np.inf
    for i in range(len(probs)):
        lo = max(0, i - n_avg + 1)
        if i - lo + 1 < min_count:
            continue
        avg = probs[lo : i + 1].mean(axis=0)
        k = int(np.argmax(avg))
        if k in keyword_idx and avg[k] >= threshold and end_s[i] - last_time >= suppress_s:
            detections.append(Detection(float(end_s[i]), labels[k], float(avg[k])))
            last_time = end_s[i]
    return detections


def match(detections, keyword_spans, ignore_spans, tolerance_s):
    """Sets each detection's outcome. Returns the list of hit flags, one per keyword span."""
    def matches(t, span):
        return span[0] <= t <= span[1] + tolerance_s

    hit = [False] * len(keyword_spans)
    for d in detections:
        same = [j for j, s in enumerate(keyword_spans) if s[2] == d.label and matches(d.time_s, s)]
        unhit = [j for j in same if not hit[j]]
        if unhit:
            hit[unhit[0]] = True
            d.outcome = "hit"
        elif same:
            d.outcome = "duplicate"
        elif any(matches(d.time_s, s) for s in ignore_spans):
            d.outcome = "ignored"
        elif any(matches(d.time_s, s) for s in keyword_spans):
            d.outcome = "substitution"
        else:
            d.outcome = "false_accept"
    return hit


def negative_duration_s(duration_s, spans):
    """Audio duration minus the union of the given spans."""
    covered = 0.0
    cur_start = cur_end = None
    for start_s, end_s, _ in sorted(spans):
        start_s, end_s = max(0.0, start_s), min(duration_s, end_s)
        if end_s <= start_s:
            continue
        if cur_end is None or start_s > cur_end:
            if cur_end is not None:
                covered += cur_end - cur_start
            cur_start, cur_end = start_s, end_s
        else:
            cur_end = max(cur_end, end_s)
    if cur_end is not None:
        covered += cur_end - cur_start
    return duration_s - covered


def score(detections, keyword_spans, ignore_spans, tolerance_s, duration_s):
    hit = match(detections, keyword_spans, ignore_spans, tolerance_s)
    n_false = sum(d.outcome in ("false_accept", "substitution", "duplicate") for d in detections)
    neg_min = negative_duration_s(duration_s, keyword_spans + ignore_spans) / 60
    recall = sum(hit) / len(keyword_spans) if keyword_spans else float("nan")
    fa_per_min = n_false / neg_min if neg_min > 0 else float("nan")
    return hit, recall, n_false, fa_per_min, neg_min


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score run_model_on_wav.py results: keyword recall and false accepts per minute."
    )
    parser.add_argument("-r", "--results", required=True, help="results CSV from run_model_on_wav.py")
    parser.add_argument("-t", "--truth", help="truth CSV (default: <results stem>_truth.csv)")
    parser.add_argument("--avg", type=float, default=0.5, help="smoothing window in seconds (default: 0.5)")
    parser.add_argument("--threshold", type=float, default=0.7, help="averaged probability to fire (default: 0.7)")
    parser.add_argument("--suppress", type=float, default=1.5, help="seconds between detections (default: 1.5)")
    parser.add_argument(
        "--tolerance", type=float,
        help="seconds after a span's end a detection may still match it (default: 1.0 + avg)",
    )
    parser.add_argument("--sweep", action="store_true", help="also print recall / false accepts vs threshold")
    parser.add_argument("-o", "--output", help="detections CSV (default: <results stem>_detections.csv)")
    return parser.parse_args()


def main():
    args = parse_args()
    results_path = pathlib.Path(args.results)
    truth_path = pathlib.Path(args.truth) if args.truth else results_path.with_name(results_path.stem + "_truth.csv")
    output_path = pathlib.Path(args.output) if args.output else results_path.with_name(results_path.stem + "_detections.csv")
    tolerance_s = args.tolerance if args.tolerance is not None else WINDOW_LEN_S + args.avg

    if not truth_path.exists():
        sys.exit(f"truth file {truth_path} not found (label the WAV and convert it, or pass -t)")

    labels, start_s, end_s, probs = load_results(results_path)
    keyword_spans, ignore_spans = load_truth(truth_path)
    duration_s = float(end_s.max())

    detections = detect(labels, end_s, probs, args.avg, args.threshold, args.suppress)
    hit, recall, n_false, fa_per_min, neg_min = score(
        detections, keyword_spans, ignore_spans, tolerance_s, duration_s
    )

    print(f"Results {results_path} ({len(probs)} windows, {duration_s:.1f}s), truth {truth_path}")
    print(f"Settings: avg {args.avg}s, threshold {args.threshold}, suppress {args.suppress}s, tolerance {tolerance_s}s")
    print(f"Keyword spans {len(keyword_spans)}, ignore spans {len(ignore_spans)}, negative audio {neg_min:.2f} min\n")

    print("Recall:")
    for kw in KEYWORDS:
        idx = [j for j, s in enumerate(keyword_spans) if s[2] == kw]
        n_hit = sum(hit[j] for j in idx)
        print(f"  {kw:<10} {n_hit:3d} / {len(idx):<3d}" + (f" ({100 * n_hit / len(idx):.0f}%)" if idx else ""))
    print(f"  {'overall':<10} {sum(hit):3d} / {len(keyword_spans):<3d} ({100 * recall:.0f}%)")

    outcomes = {o: sum(d.outcome == o for d in detections) for o in ("false_accept", "substitution", "duplicate", "ignored")}
    print(f"\nFalse accepts: {n_false} ({outcomes['false_accept']} on negative audio, "
          f"{outcomes['substitution']} substitutions, {outcomes['duplicate']} duplicates; "
          f"{outcomes['ignored']} in ignore spans not counted)")
    print(f"  {fa_per_min:.2f} per minute, {60 * fa_per_min:.0f} per hour")

    misses = [s for s, h in zip(keyword_spans, hit) if not h]
    if misses:
        print("\nMissed keywords:")
        for start, end, label in misses:
            print(f"  {start:7.2f} - {end:7.2f}s  {label}")
    false = [d for d in detections if d.outcome in ("false_accept", "substitution", "duplicate")]
    if false:
        print("\nFalse accepts:")
        for d in false:
            print(f"  {d.time_s:7.2f}s  {d.label:<10} p={d.p:.2f}  {d.outcome}")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "label", "p", "outcome"])
        for d in detections:
            writer.writerow([f"{d.time_s:.4f}", d.label, f"{d.p:.4f}", d.outcome])
    print(f"\nSaved detections to {output_path}")

    if args.sweep:
        print(f"\nThreshold sweep (avg {args.avg}s, suppress {args.suppress}s):")
        print("  threshold  recall  false accepts  per min")
        for thr in SWEEP_THRESHOLDS:
            dets = detect(labels, end_s, probs, args.avg, thr, args.suppress)
            _, r, nf, fpm, _ = score(dets, keyword_spans, ignore_spans, tolerance_s, duration_s)
            print(f"  {thr:9.2f}  {100 * r:5.0f}%  {nf:13d}  {fpm:7.2f}")


if __name__ == "__main__":
    main()
