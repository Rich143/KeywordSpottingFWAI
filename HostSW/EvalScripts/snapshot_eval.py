"""
Score one model on a split of an eval set, write one summary row to that split's history
CSV, and log the result to MLflow.

Eval sets and splits (--dataset, --split):
  board   eval_recordings/<split>/         board-mic recordings, labelled in Audacity
  public  eval_recordings_public/<split>/  LibriSpeech / AMI / SWC speech, and speech-free background
                                           noise from MUSAN / DEMAND / Sounds of Home (make_public_eval.py)
Split 'val' is for comparing models and making decisions. Split 'test' is held back: it's only
scored when asked for by name (--split test), so look at it rarely, e.g. once per finished
model, and never tune on it.

For each <stem>.wav in the split dir that has labels:
  1. if <stem>_labels.txt is newer than <stem>_truth.csv (or there is no truth
     yet), convert it (same as audacity_labels_to_truth.py)
  2. run_model_on_wav.py with the given model and stride  -> <stem>.csv
     (--jobs files at a time)
  3. score with score_results.py's detect() / score(), using the given settings
Recordings listed in <split dir>/eval_ignore.txt are skipped: one WAV filename (or glob,
e.g. eval_0[12]_*.wav) per line, '#' starts a comment.

Recall at a false accept budget, which compares models fairly: two models at the same threshold
can be at very different points of their trade-off, so instead each is held to the same false
accept rate. FA_BUDGETS are false accepts per hour for the whole device, all keywords together,
with one threshold shared by all keywords, as the device runs. A threshold sweep
(score_results.keyword_curve()) gives each keyword's recall and false accepts at each threshold;
a budget's threshold is the one with the highest mean recall whose false accepts, all keywords
together, fit in the budget. The row reports
  recall_at_<b>fa           mean of the keywords' recalls, so 'forward' being half the keywords
                            doesn't dominate it. Empty if no threshold reaches the budget
  fa_per_hour_at_<b>fa      the false accepts per hour at that threshold, all keywords
  threshold_at_<b>fa        the threshold
  fa_events_at_<b>fa        how many false accepts the rate rests on
  recall_at_<b>fa_<kw>, fa_events_at_<b>fa_<kw>
                            per keyword, at the same threshold: which keywords miss, and which
                            ones the false accepts come from
  fa_per_hour_at_<b>fa_speech / _noise, fa_events_at_<b>fa_speech / _noise
                            public: the same threshold's false accepts on speech and on background
                            noise only, each over its own hours. The budget itself is on all of it
                            together, as the device hears it
The threshold is chosen on val only. A test split is scored at the threshold its dataset's val row
chose for the same model, so score val first: nothing is tuned on test, and its false accept rate
can come out above the budget. Only val has a sweep, so only val logs the curve.

History CSV (<split dir>/eval_history.csv, committed): one row per snapshot, pooled over
all recordings (total hits / total keyword spans, total false accepts / total minutes of
negative audio), with recording = ALL. Public rows also have
  fa_per_hour_speech, false_accepts_speech, negative_min_speech   speech files only (ls, ami, swc)
  fa_per_hour_noise, false_accepts_noise, negative_min_noise      noise files only (musan, demand, soh)
  recall_ls, fa_per_hour_ami, ...                                 per source (noise: fa_per_hour and
                                                                  negative_min only)
  board   one row per run. Re-running the same version with the same settings is
          refused unless --replace, which drops the old rows first.
  public  one row per model version: re-scoring a version replaces its row.
Board histories from before this format also have per-recording rows; they're kept. Columns
added since a row was written are left empty.

MLflow (eval_tracking.py), recorded in the row's mlflow_run_id and run_type columns:
  run_type training    from TrainModel.ipynb: mlflow_run_id = the training run. The model
                       must be the one that run logged (int8_tflite_sha256 tag).
  run_type standalone  run as a script: a new eval run, nested under the version's
                       training run.
Metrics are named <dataset>_<split>_<metric> (board_val_recall, public_val_fa_per_hour,
public_val_fa_per_hour_noise, public_val_recall_ls, public_val_recall_at_100fa, ...) in both.
Artifacts in reports/: the per-recording results, for val the detection curve
(<prefix>_curve.csv, with speech / noise false accepts per threshold for public; <prefix>_curve.png), and
the split's whole history as written, as MLflow tables (sortable and filterable in the run's
Artifacts tab) in <prefix>_history/: one for the default threshold, one per source (public), one per
false accept budget and one of run settings (history_tables()).

Usage:
    python EvalScripts/snapshot_eval.py -v v10 -m tflite_models/kws_v10_int8_softmax.tflite
    python EvalScripts/snapshot_eval.py -v v10 -m tflite_models/kws_v10_int8_softmax.tflite --dataset public --jobs 6
    python EvalScripts/snapshot_eval.py -v v10 -m tflite_models/kws_v10_int8_softmax.tflite --threshold 0.6 --avg 1.0
    python EvalScripts/snapshot_eval.py -v v10 -m tflite_models/kws_v10_int8_softmax.tflite --dataset public --split test

From Python (TrainModel.ipynb does this at the end of a training run):
    import snapshot_eval
    snapshot_eval.snapshot_eval("v11", "tflite_models/kws_v11_int8_softmax.tflite", dataset="public",
                                mlflow_run_id=run.info.run_id, jobs=6)
"""

import argparse
import concurrent.futures
import csv
import datetime
import fnmatch
import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile

import audacity_labels_to_truth as labels_conv
import eval_tracking
import score_results as sr

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
HOSTSW_DIR = SCRIPT_DIR.parent
DATASET_DIRS = {
    "board": HOSTSW_DIR / "eval_recordings",
    "public": HOSTSW_DIR / "eval_recordings_public",
}
SPLITS = ["val", "test"]
TEST_SPLIT = "test"
# File name prefixes in eval_recordings_public/<split>: speech, and background noise (no keywords)
SPEECH_SOURCES = ["ls", "ami", "swc"]
NOISE_SOURCES = ["musan", "demand", "soh"]
PUBLIC_SOURCES = SPEECH_SOURCES + NOISE_SOURCES
KINDS = {"speech": SPEECH_SOURCES, "noise": NOISE_SOURCES}
IGNORE_FILE_NAME = "eval_ignore.txt"
HISTORY_FILE_NAME = "eval_history.csv"
DEFAULT_STRIDE_S = 0.1
DEFAULT_AVG_S = 0.5
DEFAULT_THRESHOLD = 0.7
DEFAULT_SUPPRESS_S = 1.5
DEFAULT_JOBS = 4
MAX_PRINTED_RECORDINGS = 10   # more than this (the public sets): per-recording results only in MLflow
# False accepts per hour for the whole device, all keywords together, at one shared threshold. Roughly where
# v10 is, 3x and 10x better. As models improve, add lower budgets rather than replacing these:
# each budget has history columns, and a history with columns not listed here is refused
FA_BUDGETS = [300, 100, 30]
FEW_EVENTS = 10   # fewer false accepts than this behind a rate: flagged as too few to rely on
ALL_ROW = "ALL"
RUN_TYPE_TRAINING = "training"     # history column values
RUN_TYPE_STANDALONE = "standalone"

# Columns that identify a snapshot: same values = same snapshot
SNAPSHOT_KEY = ["model_version", "stride_s", "avg_s", "threshold", "suppress_s", "tolerance_s"]
SETTINGS = ["stride_s", "avg_s", "threshold", "suppress_s", "tolerance_s"]

# The board history's columns before mlflow_run_id / run_type were added
BASE_FIELDS = [
    "timestamp", "model_version", "model_path", "model_sha256",
    "stride_s", "avg_s", "threshold", "suppress_s", "tolerance_s",
    "recording", "duration_min", "negative_min",
    "keyword_spans", "hits", "recall",
] + [f"recall_{kw}" for kw in sr.KEYWORDS] + [
    "false_accepts", "substitutions", "duplicates", "ignored",
    "fa_per_min", "fa_per_hour", "notes",
]
RUN_FIELDS = ["mlflow_run_id", "run_type"]
SOURCE_FIELDS = ["recall", "fa_per_hour", "keyword_spans", "negative_min"]
NOISE_SOURCE_FIELDS = ["fa_per_hour", "negative_min"]   # no keywords, so no recall
KIND_FIELDS = ["fa_per_hour", "false_accepts", "negative_min"]


def source_fields(src):
    return SOURCE_FIELDS if src in SPEECH_SOURCES else NOISE_SOURCE_FIELDS


def budget_fields(kinds=False):
    """Recall-at-budget columns; kinds: also each budget's speech / noise false accepts (public)."""
    fields = []
    for budget in FA_BUDGETS:
        fields += [f"recall_at_{budget}fa", f"fa_per_hour_at_{budget}fa", f"threshold_at_{budget}fa",
                   f"fa_events_at_{budget}fa"]
        fields += [f"recall_at_{budget}fa_{kw}" for kw in sr.KEYWORDS]
        fields += [f"fa_events_at_{budget}fa_{kw}" for kw in sr.KEYWORDS]
        if kinds:
            for kind in KINDS:
                fields += [f"fa_per_hour_at_{budget}fa_{kind}", f"fa_events_at_{budget}fa_{kind}"]
    return fields


# History columns that are text even when they look like numbers (a hex SHA can be all digits)
HISTORY_TEXT_FIELDS = {"timestamp", "model_version", "model_path", "model_sha256", "recording", "notes",
                       "mlflow_run_id", "run_type"}


def history_fields(per_source):
    fields = list(BASE_FIELDS)
    if per_source:
        fields += [f"{f}_{kind}" for kind in KINDS for f in KIND_FIELDS]
        fields += [f"{f}_{src}" for src in PUBLIC_SOURCES for f in source_fields(src)]
    return fields + budget_fields(per_source) + RUN_FIELDS


def kind_indices(recordings):
    """{kind: [index into recordings]} of public recordings, by file name prefix."""
    out = {kind: [] for kind in KINDS}
    for i, wav in enumerate(recordings):
        for kind, sources in KINDS.items():
            if wav.name.split("_")[0] in sources:
                out[kind].append(i)
    return {kind: idx for kind, idx in out.items() if idx}


def sum_counts(counts):
    """keyword_counts() results added up (they're per-recording sums, so this pools them)."""
    return {kw: {k: sum(c[kw][k] for c in counts) for k in ("hits", "false_accepts")} for kw in sr.KEYWORDS}


class SnapshotError(RuntimeError):
    """The snapshot can't be taken (missing model/recordings, or it already exists).
    The script turns it into an exit message; a notebook gets a normal exception."""


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def display_path(path):
    """Path relative to HostSW when possible, so the history CSV doesn't depend on where the repo is checked out."""
    try:
        return str(path.resolve().relative_to(HOSTSW_DIR))
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
        raise SnapshotError(f"run_model_on_wav.py failed on {wav.name}:\n{proc.stdout}{proc.stderr}")
    return results


def score_recording(recording, threshold, suppress_s, tolerance_s):
    """Returns a dict of the per-recording metric columns."""
    keyword_spans = recording.keyword_spans
    duration_s = recording.duration_s
    detections = sr.fire(recording.labels, recording.smoothed, threshold, suppress_s)
    hit, _, n_false, _, neg_min = sr.score(detections, keyword_spans, recording.ignore_spans, tolerance_s, duration_s)

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


def pool(rows):
    """Sums the count columns of per-recording rows and adds the rates."""
    totals = {}
    for row in rows:
        for k, v in row.items():
            if k != "recording":
                totals[k] = totals.get(k, 0) + v
    return finish_row(totals)


def format_row(row, fields):
    out = {}
    for k in fields:
        v = row.get(k, "")
        if isinstance(v, float):
            v = f"{v:.4f}" if k.startswith(("recall", "fa_", "duration", "negative")) else f"{v:g}"
        out[k] = v
    return out


def read_history(path, fields):
    """Rows of a history CSV. One from before columns were added is upgraded in memory (the new
    columns empty); it's rewritten with the new header."""
    if not path.exists():
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        unknown = set(reader.fieldnames) - set(fields)
        if unknown:
            raise SnapshotError(f"{path}: columns this script doesn't know: {sorted(unknown)} "
                                f"(was it edited, or written by another version?)")
        return [{**{k: "" for k in fields}, **r} for r in reader]


def write_history(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def same_snapshot(row, key):
    return all(row[k] == key[k] for k in SNAPSHOT_KEY)


def mean_recall(counts, all_row):
    """Mean of the keywords' recalls from keyword_counts(), or None if a keyword has no spans."""
    spans = [all_row[f"spans_{kw}"] for kw in sr.KEYWORDS]
    if not all(spans):
        return None
    return sum(counts[kw]["hits"] / n for kw, n in zip(sr.KEYWORDS, spans)) / len(spans)


def total_false_accepts(counts):
    return sum(counts[kw]["false_accepts"] for kw in sr.KEYWORDS)


def choose_thresholds(curve, all_row):
    """{budget: threshold or None}: the curve threshold with the highest mean recall among those
    whose false accepts, all keywords together, fit in the budget (the highest such threshold if
    several tie). None if none does."""
    hours = all_row["negative_min"] / 60
    chosen = {}
    for budget in FA_BUDGETS:
        allowed = [(mean_recall(counts, all_row), threshold) for threshold, counts in curve
                   if total_false_accepts(counts) <= budget * hours and mean_recall(counts, all_row) is not None]
        chosen[budget] = max(allowed)[1] if allowed else None
    return chosen


def val_thresholds(dataset, snapshot, fields):
    """The thresholds the val split chose for this model, as choose_thresholds() returns them."""
    path = DATASET_DIRS[dataset] / "val" / HISTORY_FILE_NAME
    key = ["model_version", "model_sha256", "stride_s", "avg_s", "suppress_s", "tolerance_s"]
    rows = [r for r in read_history(path, fields) if r["recording"] == ALL_ROW and all(r[k] == snapshot[k] for k in key)
            and any(r[f"threshold_at_{b}fa"] for b in FA_BUDGETS)]
    if not rows:
        raise SnapshotError(f"{path} has no budget thresholds for this model and these settings: test uses the "
                            f"thresholds chosen on val, so score {dataset} val first")
    row = rows[-1]
    return {b: float(row[f"threshold_at_{b}fa"]) if row[f"threshold_at_{b}fa"] else None for b in FA_BUDGETS}


def budget_columns(recordings, chosen, all_row, suppress_s, tolerance_s, kinds=None):
    """The history columns for the chosen thresholds, scored at exactly those thresholds.
    kinds ({kind: [recording index]}, public): also each kind's false accepts over its own hours."""
    hours = all_row["negative_min"] / 60
    columns = {f: "" for f in budget_fields(kinds is not None)}
    for budget in FA_BUDGETS:
        threshold = chosen[budget]
        if threshold is None or not hours:
            continue
        for kind, idx in (kinds or {}).items():
            kind_counts = sr.keyword_counts([recordings[i] for i in idx], threshold, suppress_s, tolerance_s)
            kind_hours = all_row[f"negative_min_{kind}"] / 60
            events = total_false_accepts(kind_counts)
            columns[f"fa_per_hour_at_{budget}fa_{kind}"] = events / kind_hours if kind_hours else ""
            columns[f"fa_events_at_{budget}fa_{kind}"] = events
        counts = sr.keyword_counts(recordings, threshold, suppress_s, tolerance_s)
        recall, events = mean_recall(counts, all_row), total_false_accepts(counts)
        columns[f"recall_at_{budget}fa"] = recall if recall is not None else ""
        columns[f"fa_per_hour_at_{budget}fa"] = events / hours
        columns[f"threshold_at_{budget}fa"] = threshold
        columns[f"fa_events_at_{budget}fa"] = events
        for kw in sr.KEYWORDS:
            spans = all_row[f"spans_{kw}"]
            columns[f"recall_at_{budget}fa_{kw}"] = counts[kw]["hits"] / spans if spans else ""
            columns[f"fa_events_at_{budget}fa_{kw}"] = counts[kw]["false_accepts"]
    return columns


def curve_rows(curve, all_row, kind_curves=None):
    """The detection curve as CSV rows, one per threshold and keyword, with the threshold's
    false accepts for all keywords together (what the budgets are measured against), and with
    kind_curves ({kind: curve}, public) those on each kind's audio over its own hours."""
    hours = all_row["negative_min"] / 60
    rows = []
    for n, (threshold, counts) in enumerate(curve):
        total = total_false_accepts(counts)
        by_kind = {}
        for kind, kind_curve in (kind_curves or {}).items():
            kind_hours = all_row[f"negative_min_{kind}"] / 60
            events = total_false_accepts(kind_curve[n][1])
            by_kind[f"total_fa_per_hour_{kind}"] = events / kind_hours if kind_hours else ""
        for kw in sr.KEYWORDS:
            spans = all_row[f"spans_{kw}"]
            if spans and hours:
                rows.append({"threshold": threshold, "keyword": kw, "hits": counts[kw]["hits"], "spans": spans,
                             "recall": counts[kw]["hits"] / spans, "false_accepts": counts[kw]["false_accepts"],
                             "fa_per_hour": counts[kw]["false_accepts"] / hours,
                             "total_false_accepts": total, "total_fa_per_hour": total / hours, **by_kind})
    return rows


def save_curve_plot(path, rows, title):
    """DET plot of curve_rows(): each keyword's miss rate, and their mean, against the false accepts
    per hour of all keywords together at the same shared threshold, with the budgets marked.
    Without pyplot, so a notebook's backend is left alone."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colours = dict(zip(sr.KEYWORDS, ["#2a78d6", "#eb6834", "#1baf7a"]))   # validated for colour blindness
    fig = Figure(figsize=(7.5, 4.8), dpi=150, facecolor=surface)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(facecolor=surface)
    by_threshold = {}
    for r in rows:
        by_threshold.setdefault(r["threshold"], []).append(r)
    curves = {kw: [] for kw in sr.KEYWORDS}
    mean = []
    for threshold, group in by_threshold.items():
        x = group[0]["total_fa_per_hour"]
        if group[0]["total_false_accepts"] == 0:
            continue   # no place on a log axis
        for r in group:
            curves[r["keyword"]].append((x, 1 - r["recall"]))
        if len(group) == len(sr.KEYWORDS):
            mean.append((x, 1 - sum(r["recall"] for r in group) / len(group)))
    lines = [(kw, colours[kw], "-", sorted(points)) for kw, points in curves.items() if points]
    if mean:
        lines.append(("mean", muted, "--", sorted(mean)))
    for label, colour, style, points in lines:
        ax.plot(*zip(*points), color=colour, linewidth=1.5, linestyle=style, label=label, solid_capstyle="round")
    # Direct labels at the line ends, nudged apart where the ends are close
    ends = sorted((points[-1][1], points[-1][0], label) for label, _, _, points in lines)
    placed = []
    for y, x, label in ends:
        if placed and y - placed[-1] < 0.045:
            y = placed[-1] + 0.045
        placed.append(y)
        ax.text(x * 1.12, y, label, va="center", fontsize=8, color=ink)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    for budget in FA_BUDGETS:
        ax.axvline(budget, color=muted, linewidth=0.8, linestyle=":")
        ax.annotate(f"{budget}/h", (budget, 1), xytext=(3, -3), textcoords="offset points", va="top",
                    fontsize=7, color=muted, rotation=90)
    ax.set_xlabel("False accepts per hour, all keywords (one shared threshold)", color=ink, fontsize=9)
    ax.set_ylabel("Miss rate (1 \u2212 recall)", color=ink, fontsize=9)
    ax.set_title(f"{title}: lower left is better", color=ink, fontsize=10, loc="left")
    ax.grid(color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=muted, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)


def log_to_mlflow(run_id, prefix, all_row, per_source, per_recording, settings, curve=None, title=""):
    """Metrics <prefix>_<name>, detection settings as <prefix>_<setting> tags (tags, not
    params: a later snapshot with other settings should replace them), the per-recording
    rows as reports/<prefix>_per_recording.csv, and the detection curve (curve_rows()) as
    reports/<prefix>_curve.csv and .png."""
    import mlflow
    client = mlflow.MlflowClient()
    metrics = {f"{prefix}_{k}": all_row[k] for k in
               ["recall", "fa_per_hour", "false_accepts", "substitutions", "negative_min", "keyword_spans"]}
    for kw in sr.KEYWORDS:
        metrics[f"{prefix}_recall_{kw}"] = all_row[f"recall_{kw}"]
    for src, row in per_source.items():
        metrics[f"{prefix}_recall_{src}"] = row["recall"]
        metrics[f"{prefix}_fa_per_hour_{src}"] = row["fa_per_hour"]
    for kind in KINDS:
        if f"fa_per_hour_{kind}" in all_row:
            metrics[f"{prefix}_fa_per_hour_{kind}"] = all_row[f"fa_per_hour_{kind}"]
    for name in budget_fields(kinds=True):
        if not name.startswith("fa_events") and all_row.get(name, "") != "":
            metrics[f"{prefix}_{name}"] = all_row[name]
    for name, value in metrics.items():
        if value == value:   # skip NaN (e.g. no keywords in a source)
            client.log_metric(run_id, name, float(value))
    for key, value in settings.items():
        client.set_tag(run_id, f"{prefix}_{key}", str(value))
    client.set_tag(run_id, f"{prefix}_eval_timestamp", all_row["timestamp"])
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, f"{prefix}_per_recording.csv")
        fields = ["recording"] + [k for k in per_recording[0] if k != "recording"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(per_recording)
        client.log_artifact(run_id, path, "reports")
        if curve:
            path = os.path.join(tmp_dir, f"{prefix}_curve.csv")
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(curve[0]))
                writer.writeheader()
                writer.writerows(curve)
            client.log_artifact(run_id, path, "reports")
            path = os.path.join(tmp_dir, f"{prefix}_curve.png")
            save_curve_plot(path, curve, title)
            client.log_artifact(run_id, path, "reports")
    return metrics


def history_tables(rows, fields):
    """The history split into narrower tables, {name: (rows, {history column: table column})}.
    Names are numbered so the MLflow Artifacts tab lists them in this order. Every history column
    is in one of them, apart from fa_per_min (fa_per_hour / 60)."""
    first = {"model_version": "model_version", "timestamp": "timestamp"}

    def same(*cols):
        return {c: c for c in cols if c in fields}

    tables = {}
    tables["default_threshold"] = (rows, {**first, **same(
        "recording", "recall", *[f"recall_{kw}" for kw in sr.KEYWORDS], "fa_per_hour",
        *[f"fa_per_hour_{kind}" for kind in KINDS], "false_accepts", *[f"false_accepts_{kind}" for kind in KINDS],
        "substitutions", "duplicates", "ignored", "hits", "keyword_spans", "negative_min",
        *[f"negative_min_{kind}" for kind in KINDS], "duration_min")})
    if f"recall_{PUBLIC_SOURCES[0]}" in fields:
        tables["default_threshold_by_source"] = (rows, {**first, **same(
            *[f"{f}_{src}" for src in PUBLIC_SOURCES for f in source_fields(src)])})
    # Only rows scored since the budgets existed: older ones would be all empty
    scored = [r for r in rows if any(r.get(f"threshold_at_{b}fa") for b in FA_BUDGETS)]
    for budget in FA_BUDGETS:
        at = f"_at_{budget}fa"
        columns = {**first, f"recall{at}": "recall", f"fa_per_hour{at}": "fa_per_hour"}
        columns.update({f"fa_per_hour{at}_{kind}": f"fa_per_hour_{kind}" for kind in KINDS if f"fa_per_hour{at}_{kind}" in fields})
        columns.update({f"threshold{at}": "threshold", f"fa_events{at}": "fa_events"})
        columns.update({f"fa_events{at}_{kind}": f"fa_events_{kind}" for kind in KINDS if f"fa_events{at}_{kind}" in fields})
        columns.update({f"recall{at}_{kw}": f"recall_{kw}" for kw in sr.KEYWORDS})
        columns.update({f"fa_events{at}_{kw}": f"fa_events_{kw}" for kw in sr.KEYWORDS})
        tables[f"budget_{budget}fa"] = (scored, columns)
    tables["run_settings"] = (rows, {**first, **same(
        "recording", "run_type", "threshold", "avg_s", "suppress_s", "stride_s", "tolerance_s",
        "model_path", "model_sha256", "notes", "mlflow_run_id")})
    return {f"{n}_{name}": table for n, (name, table) in enumerate(tables.items(), 1)}


def log_history_table(run_id, prefix, rows, fields):
    """The split's whole history, rows as written to the CSV, as MLflow tables in
    reports/<prefix>_history/ (history_tables()): sortable and filterable in the run's Artifacts
    tab. Numbers are numbers, so they sort as numbers; empty cells are empty."""
    import mlflow
    import pandas as pd
    for name, (table_rows, columns) in history_tables(rows, fields).items():
        table = pd.DataFrame([{new: r.get(old, "") for old, new in columns.items()} for r in table_rows],
                             columns=list(columns.values()))
        for old, new in columns.items():
            if old not in HISTORY_TEXT_FIELDS:
                table[new] = pd.to_numeric(table[new].replace("", None), errors="coerce")
        table = table.astype(object).where(table.notna(), None)
        mlflow.log_table(table, artifact_file=f"reports/{prefix}_history/{name}.json", run_id=run_id)


def print_budgets(row, split):
    """The recall-at-budget block of the summary, with the events behind each false accept rate."""
    source = "val's" if split == TEST_SPLIT else "chosen per budget"
    print(f"  Recall at a false accept budget (FA/hour for the device, one shared threshold, {source}):")
    few = False
    for budget in FA_BUDGETS:
        threshold, events = row[f"threshold_at_{budget}fa"], row[f"fa_events_at_{budget}fa"]
        if events == "":
            print(f"    {budget:4d}/hour  not reached")
            continue
        mark = "*" if events < FEW_EVENTS else ""
        few = few or events < FEW_EVENTS
        recall = row[f"recall_at_{budget}fa"]
        cells = []
        for kw in sr.KEYWORDS:
            spans = row[f"spans_{kw}"]
            if spans:
                hits = round(row[f"recall_at_{budget}fa_{kw}"] * spans)
                cells.append(f"{kw} {100 * hits / spans:.0f}% ({hits}/{spans}, {row[f'fa_events_at_{budget}fa_{kw}']} FA)")
        total = f"recall {100 * recall:3.0f}%" if recall != "" else "recall n/a"
        kinds = "".join(f", {kind} {row[f'fa_per_hour_at_{budget}fa_{kind}']:.1f}"
                        for kind in KINDS if row.get(f"fa_per_hour_at_{budget}fa_{kind}", "") != "")
        print(f"    {budget:4d}/hour  {total} at {threshold:g}, {row[f'fa_per_hour_at_{budget}fa']:5.1f} FA/hour "
              f"({events}{mark} FA{kinds})  |  " + "  ".join(cells))
    if few:
        print(f"    * fewer than {FEW_EVENTS} false accepts behind this rate: too few to rely on")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score a model on one split of an eval set, append a summary row to its history CSV and log it to MLflow."
    )
    parser.add_argument("-v", "--version", required=True, help="model version label, e.g. v10")
    parser.add_argument("-m", "--model", required=True, help="tflite model path")
    parser.add_argument("--dataset", choices=list(DATASET_DIRS), default="board",
                        help="eval set: board (eval_recordings/) or public (eval_recordings_public/) (default: board)")
    parser.add_argument("--split", choices=SPLITS, default="val",
                        help="split to score (default: val). test is held back: score it rarely")
    parser.add_argument("-s", "--stride", type=float, default=DEFAULT_STRIDE_S,
                        help=f"window stride in seconds (default: {DEFAULT_STRIDE_S})")
    parser.add_argument("--avg", type=float, default=DEFAULT_AVG_S,
                        help=f"smoothing window, s (default: {DEFAULT_AVG_S})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"detection threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--suppress", type=float, default=DEFAULT_SUPPRESS_S,
                        help=f"suppression time, s (default: {DEFAULT_SUPPRESS_S})")
    parser.add_argument("--tolerance", type=float, help="match tolerance after a span, s (default: 1.0 + avg)")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                        help=f"recordings run through the model at once (default: {DEFAULT_JOBS})")
    parser.add_argument("--notes", default="", help="free text stored with the snapshot")
    parser.add_argument("--eval-dir", help="recordings dir (default: from --dataset and --split)")
    parser.add_argument("--history", help=f"history CSV (default: <eval dir>/{HISTORY_FILE_NAME})")
    parser.add_argument("--replace", action="store_true",
                        help="board: replace an existing snapshot with the same version and settings")
    return parser.parse_args()


def snapshot_eval(version, model_path, dataset="board", split="val", stride_s=DEFAULT_STRIDE_S,
                  avg_s=DEFAULT_AVG_S, threshold=DEFAULT_THRESHOLD, suppress_s=DEFAULT_SUPPRESS_S,
                  tolerance_s=None, notes="", eval_dir=None, history_path=None, replace=False,
                  jobs=DEFAULT_JOBS, mlflow_run_id=None):
    """Scores model_path on every labelled recording of one split, writes the summary row to
    the split's history CSV and logs it to MLflow (into mlflow_run_id if given, else a new
    standalone eval run).

    Same options as the command line; tolerance_s defaults to the window length + avg_s.
    Returns the summary row as written to the CSV, or None if the split has no recordings
    (nothing is written). Raises SnapshotError if the snapshot already exists (board,
    unless replace), the model doesn't match mlflow_run_id, or a model run fails.
    """
    if dataset not in DATASET_DIRS:
        raise SnapshotError(f"unknown dataset {dataset!r} (one of {', '.join(DATASET_DIRS)})")
    if split not in SPLITS:
        raise SnapshotError(f"unknown split {split!r} (one of {', '.join(SPLITS)})")
    eval_dir = pathlib.Path(eval_dir) if eval_dir else DATASET_DIRS[dataset] / split
    if eval_dir.name == TEST_SPLIT and split != TEST_SPLIT:
        raise SnapshotError(f"{eval_dir} is a test split: score it only with split='{TEST_SPLIT}'")
    history_path = pathlib.Path(history_path) if history_path else eval_dir / HISTORY_FILE_NAME
    model_path = pathlib.Path(model_path)
    per_source = dataset == "public"
    fields = history_fields(per_source)
    prefix = f"{dataset}_{split}"
    if tolerance_s is None:
        tolerance_s = sr.WINDOW_LEN_S + avg_s
    if not model_path.exists():
        raise SnapshotError(f"model {model_path} not found")

    recordings, skipped = find_recordings(eval_dir, read_ignore_patterns(eval_dir)) if eval_dir.is_dir() else ([], [])
    for wav, reason in skipped:
        print(f"Skipping {wav.name}: {reason}")
    if not recordings:
        print(f"{prefix}: no labelled recordings in {eval_dir}, nothing scored")
        return None

    model_sha = file_sha256(model_path)
    eval_tracking.init_mlflow()
    if mlflow_run_id:
        import mlflow
        run_sha = mlflow.MlflowClient().get_run(mlflow_run_id).data.tags.get("int8_tflite_sha256", "")
        if run_sha != model_sha:
            raise SnapshotError(f"MLflow run {mlflow_run_id} logged an int8 model with sha256 "
                                f"{run_sha[:12] or '(none)'}, but {model_path} has {model_sha[:12]}")

    snapshot = format_row({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_version": version,
        "model_path": display_path(model_path),
        "model_sha256": model_sha[:12],
        "stride_s": stride_s,
        "avg_s": avg_s,
        "threshold": threshold,
        "suppress_s": suppress_s,
        "tolerance_s": tolerance_s,
        "notes": notes,
    }, fields)

    history = read_history(history_path, fields)
    if dataset == "board" and any(same_snapshot(r, snapshot) for r in history):
        if not replace:
            raise SnapshotError(f"{history_path} already has a snapshot for {version} with these settings "
                                f"(use --replace to overwrite it)")
        history = [r for r in history if not same_snapshot(r, snapshot)]
    chosen = val_thresholds(dataset, snapshot, fields) if split == TEST_SPLIT else None

    # Model runs in parallel (each is a separate process); scoring in order
    print(f"{prefix}: running {model_path.name} on {len(recordings)} recording(s), {jobs} at a time...")
    truths = {wav: update_truth(wav) for wav in recordings}
    with concurrent.futures.ThreadPoolExecutor(max(1, jobs)) as pool_ex:
        results = dict(zip(recordings, pool_ex.map(lambda w: run_model(model_path, w, stride_s), recordings)))
    per_recording, loaded = [], []
    for wav in recordings:
        loaded.append(sr.load_recording(results[wav], truths[wav], avg_s))
        counts = score_recording(loaded[-1], threshold, suppress_s, tolerance_s)
        per_recording.append({"recording": wav.name, **counts})
        row = finish_row(counts)
        if len(recordings) <= MAX_PRINTED_RECORDINGS:
            print(f"  {wav.name}: recall {row['hits']}/{row['keyword_spans']} ({100 * row['recall']:.0f}%), "
                  f"false accepts {row['false_accepts']} ({row['fa_per_min']:.2f}/min)")
    if len(recordings) > MAX_PRINTED_RECORDINGS:
        print(f"  (per-recording results: reports/{prefix}_per_recording.csv in the MLflow run)")

    all_row = pool(per_recording)
    sources = {}
    kinds = kind_indices(recordings) if per_source else None
    if per_source:
        for src in PUBLIC_SOURCES:
            rows = [r for r in per_recording if r["recording"].startswith(f"{src}_")]
            if rows:
                sources[src] = pool(rows)
                for f in source_fields(src):
                    all_row[f"{f}_{src}"] = sources[src][f]
        for kind, idx in kinds.items():
            pooled = pool([per_recording[i] for i in idx])
            for f in KIND_FIELDS:
                all_row[f"{f}_{kind}"] = pooled[f]

    # Shared thresholds for the false accept budgets: chosen on val, reused on test. Public: a
    # sweep per kind, added up for the whole set's
    curve = None
    if chosen is None:
        if kinds:
            kind_sweeps = {kind: sr.keyword_curve([loaded[i] for i in idx], suppress_s, tolerance_s)
                           for kind, idx in kinds.items()}
            rest = [rec for i, rec in enumerate(loaded) if not any(i in idx for idx in kinds.values())]
            parts = list(kind_sweeps.values()) + ([sr.keyword_curve(rest, suppress_s, tolerance_s)] if rest else [])
            sweep = [(points[0][0], sum_counts([p[1] for p in points])) for points in zip(*parts)]
        else:
            kind_sweeps, sweep = None, sr.keyword_curve(loaded, suppress_s, tolerance_s)
        chosen = choose_thresholds(sweep, all_row)
        curve = curve_rows(sweep, all_row, kind_sweeps)
    all_row.update(budget_columns(loaded, chosen, all_row, suppress_s, tolerance_s, kinds))

    # MLflow: into the training run (notebook) or a new standalone eval run
    settings = {k: snapshot[k] for k in SETTINGS}
    if mlflow_run_id:
        run_id, run_type = mlflow_run_id, RUN_TYPE_TRAINING
    else:
        run_id = eval_tracking.start_eval_run(version, dataset, split, {
            **settings, "model_path": snapshot["model_path"], "model_sha256": model_sha, "eval_dir": display_path(eval_dir)})
        run_type = RUN_TYPE_STANDALONE
    row = format_row({**snapshot, **all_row, "recording": ALL_ROW, "mlflow_run_id": run_id, "run_type": run_type}, fields)
    previous = [r for r in history if r["recording"] == ALL_ROW]
    if dataset == "public":
        replaced = [r for r in history if r["model_version"] == version]
        history = [r for r in history if r["model_version"] != version]
        if replaced:
            print(f"Replacing the previous {version} row in {history_path} (from {replaced[-1]['timestamp']})")
    history.append(row)

    log_to_mlflow(run_id, prefix, {**all_row, "timestamp": snapshot["timestamp"]}, sources, per_recording, settings,
                  curve, f"{version} on {prefix}")
    log_history_table(run_id, prefix, history, fields)
    if not mlflow_run_id:
        import mlflow
        mlflow.MlflowClient().set_terminated(run_id)
    write_history(history_path, history, fields)

    print(f"\n{version} on {prefix} ({len(recordings)} recording{'s' if len(recordings) != 1 else ''}, avg {avg_s}s, "
          f"threshold {threshold}, suppress {suppress_s}s):")
    print(f"  recall {all_row['hits']}/{all_row['keyword_spans']} ({100 * all_row['recall']:.0f}%)  "
          + "  ".join(f"{kw} {100 * all_row[f'recall_{kw}']:.0f}%" for kw in sr.KEYWORDS))
    print(f"  false accepts {all_row['false_accepts']} in {all_row['negative_min']:.1f} min "
          f"= {all_row['fa_per_min']:.2f}/min ({all_row['fa_per_hour']:.0f}/hour)")
    for kind in KINDS:
        if f"fa_per_hour_{kind}" in all_row:
            print(f"    {kind:6s} {all_row[f'fa_per_hour_{kind}']:6.1f} FA/hour ({all_row[f'false_accepts_{kind}']} "
                  f"in {all_row[f'negative_min_{kind}']:.0f} min)")
    for src, r in sources.items():
        recall = f"recall {100 * r['recall']:3.0f}% of {r['keyword_spans']:4d}, " if r["keyword_spans"] else " " * 22
        print(f"    {src:6s} {recall}{r['fa_per_hour']:6.1f} FA/hour over {r['negative_min']:.0f} min")
    print_budgets(all_row, split)
    print(f"  MLflow: {run_type} run {run_id}")
    if previous:
        print("\nPrevious snapshots (recordings may differ between snapshots):")
        for r in previous[-5:]:
            print(f"  {r['timestamp']}  {r['model_version']:<8} avg {r['avg_s']} thr {r['threshold']} "
                  f"sup {r['suppress_s']}  recall {100 * float(r['recall']):.0f}%  "
                  f"FA {float(r['fa_per_hour']):.1f}/hour  {r['notes']}")
    print(f"\nWrote the summary row to {history_path}")
    return row


def main():
    args = parse_args()
    try:
        snapshot_eval(args.version, args.model, dataset=args.dataset, split=args.split, stride_s=args.stride,
                      avg_s=args.avg, threshold=args.threshold, suppress_s=args.suppress,
                      tolerance_s=args.tolerance, notes=args.notes, eval_dir=args.eval_dir,
                      history_path=args.history, replace=args.replace, jobs=args.jobs)
    except SnapshotError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
