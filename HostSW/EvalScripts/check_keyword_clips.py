"""
Check Speech Commands' keyword clips (dataset/{forward,backward,stop}/) for partial or wrong
words, so they can be left out of training: a clip cut from the recording too early or too late
holds only part of the word ("for", "ward", "back"), which teaches the model that a fragment
is the keyword.

Two independent checks per clip:
  Whisper  large-v3-turbo (mlx-whisper, on the Apple GPU) transcribes it, with no prompt so it
           isn't nudged towards the keyword. The transcript is classed (whisper_class()):
             ok       the keyword (or "forwards" / "backwards"), possibly repeated
             partial  a fragment of it: a prefix or suffix ("for", "ward", "back", "top"), or a
                      word a fragment is usually heard as (PARTIAL_WORDS: "four", "bag")
             near     about as long, and starting or ending like it: probably all of it, heard
                      differently ("for what", "backboard", "awkward", "stopped", "start")
             extra    the keyword plus other words
             other    something unrelated ("thank you", "okay": Whisper's usual output for
                      noise or unclear audio)
             empty    no words
  Energy   10 ms frames; the word is the run of loud frames (above the clip's floor + ACTIVE_FLOOR_DB
           and within ACTIVE_PEAK_DB of its peak, gaps up to MERGE_GAP_S merged) around the peak:
             cut_start / cut_end  the first / last EDGE_S of the audio is within EDGE_PEAK_DB of the
                                  peak: the word runs into the clip edge (some files are shorter
                                  than 1 s; their real end is used)
             short                the word is under SHORT_FRACTION of the keyword's median length
             quiet                the peak is less than QUIET_SNR_DB above the floor
Each check flags things the other can't: Whisper a clean but incomplete word, energy a word
that's cut but still recognisable.

Output (dataset/ is gitignored, like the clips):
  dataset/keyword_check.csv           one row per clip: transcript, class, energy measures.
                                      Re-running only checks clips that aren't in it yet
  dataset/keyword_check_spotcheck.*   a listening file (.wav, Audacity _labels.txt, .csv): up to
                                      --per-flag clips of each keyword and flag, plus --controls
                                      unflagged clips per keyword, at their original level

The Whisper model (~1.6 GB) is downloaded once from Hugging Face into ~/.cache/huggingface.
Runs at ~0.5 s per clip on an M2 Pro (the encoder always sees a 30 s window), ~1 h for all.

Usage (from HostSW/):
    python EvalScripts/check_keyword_clips.py
    python EvalScripts/check_keyword_clips.py --report-only
"""

import argparse
import csv
import pathlib
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
HOSTSW_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = HOSTSW_DIR / "dataset"
RESULTS_NAME = "keyword_check.csv"
SPOTCHECK_NAME = "keyword_check_spotcheck"

KEYWORDS = ["forward", "backward", "stop"]
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
BATCH_SIZE = 8                # clips per decode call (the GPU is busy either way)
SAMPLE_RATE_HZ = 16000

# Transcript classes
VARIANTS = {"forward": {"forward", "forwards"}, "backward": {"backward", "backwards"}, "stop": {"stop"}}
# Words a fragment of the keyword is usually transcribed as (besides its own prefixes / suffixes)
PARTIAL_WORDS = {
    "forward": {"for", "four", "fore", "ford", "forth", "ward", "wards", "word", "words", "war"},
    "backward": {"back", "backs", "bag", "bach", "pack", "ward", "wards", "word", "words", "war"},
    "stop": {"top", "tops", "sto", "stah"},
}
MIN_FRAGMENT_LETTERS = 2
NUMBER_WORDS = {"4": "four"}
# 'near': a transcript that starts (or ends) like the keyword and is about as long: "for what",
# "backboard", "awkward", "stopped", "start", where 'other' is unrelated ("thank you", "okay":
# Whisper's usual output for noise or unclear audio)
NEAR_PREFIX = {"forward": "for", "backward": "bac", "stop": "st"}
NEAR_SUFFIXES = {"forward": ("ward", "word", "words", "what", "wood"),
                 "backward": ("ward", "word", "words", "what", "wood", "board", "court")}
NEAR_MIN_LENGTH = 0.7

# Energy measures
FRAME_S = 0.01
ACTIVE_FLOOR_DB = 12.0        # a loud frame is this far above the clip's floor (10th percentile)...
ACTIVE_PEAK_DB = 30.0         # ...and within this of its peak
MERGE_GAP_S = 0.15            # quieter gaps inside a word (the closure in "stop") up to this
EDGE_S = 0.02
EDGE_PEAK_DB = 20.0
SHORT_FRACTION = 0.5
QUIET_SNR_DB = 10.0

FIELDS = ["file", "keyword", "duration_s", "whisper_text", "whisper_class", "no_speech_prob",
          "avg_logprob", "floor_db", "peak_db", "word_start_s", "word_end_s", "word_s",
          "start_rel_db", "end_rel_db", "cut_start", "cut_end", "quiet"]
WHISPER_FLAGS = ["partial", "near", "other", "empty", "extra"]
ENERGY_FLAGS = ["cut_start", "cut_end", "short", "quiet"]

SPOTCHECK_GAP_S = 0.5


# --- Checks ------------------------------------------------------------------

def whisper_class(text, keyword):
    """ok / partial / near / extra / other / empty for a transcript of a clip of keyword."""
    words = re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()
    words = [NUMBER_WORDS.get(w, w) for w in words if re.sub(r"[0-9]", "", w) or w in NUMBER_WORDS]
    if not words:
        return "empty"
    joined = "".join(words)
    variants = VARIANTS[keyword]
    if all(w in variants for w in words) or joined in variants:
        return "ok"
    if any(w in variants for w in words):
        return "extra"
    fragment = (len(joined) >= MIN_FRAGMENT_LETTERS
                and (keyword.startswith(joined) or keyword.endswith(joined)))
    if fragment or joined in PARTIAL_WORDS[keyword]:
        return "partial"
    # As long as the keyword and starting or ending like it: probably all of it, heard differently
    sounds_like = (joined.replace("four", "for", 1).startswith(NEAR_PREFIX[keyword])
                   or joined.endswith(NEAR_SUFFIXES.get(keyword, ())))
    if sounds_like and len(joined) >= NEAR_MIN_LENGTH * len(keyword):
        return "near"
    return "other"


def frame_db(audio):
    n = round(FRAME_S * SAMPLE_RATE_HZ)
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    frames = frames - frames.mean(axis=1, keepdims=True)
    return 10 * np.log10((frames ** 2).mean(axis=1) + 1e-10)


def energy_measures(audio):
    """The word's position and length, and the edge / level flags (see the docstring)."""
    db = frame_db(audio)
    floor, peak = float(np.percentile(db, 10)), float(db.max())
    loud = db > max(floor + ACTIVE_FLOOR_DB, peak - ACTIVE_PEAK_DB)
    # The run of loud frames around the peak, bridging short gaps
    merge = round(MERGE_GAP_S / FRAME_S)
    start = end = int(db.argmax())
    while True:
        before = np.flatnonzero(loud[max(0, start - merge - 1):start])
        after = np.flatnonzero(loud[end + 1:end + merge + 2])
        if not before.size and not after.size:
            break
        if before.size:
            start = max(0, start - merge - 1) + int(before[0])
        if after.size:
            end = end + 1 + int(after[-1])
    edge = max(1, round(EDGE_S / FRAME_S))
    start_rel, end_rel = float(db[:edge].max() - peak), float(db[-edge:].max() - peak)
    strong = max(peak - EDGE_PEAK_DB, floor + ACTIVE_FLOOR_DB)
    return {
        "duration_s": f"{len(audio) / SAMPLE_RATE_HZ:.3f}",
        "floor_db": f"{floor:.1f}", "peak_db": f"{peak:.1f}",
        "word_start_s": f"{start * FRAME_S:.2f}", "word_end_s": f"{(end + 1) * FRAME_S:.2f}",
        "word_s": f"{(end + 1 - start) * FRAME_S:.2f}",
        "start_rel_db": f"{start_rel:.1f}", "end_rel_db": f"{end_rel:.1f}",
        "cut_start": int(db[:edge].max() >= strong), "cut_end": int(db[-edge:].max() >= strong),
        "quiet": int(peak - floor < QUIET_SNR_DB),
    }


class Transcriber:
    """Whisper on the Apple GPU (MLX), one 30 s window per clip, temperature 0, English, no prompt."""

    def __init__(self, model_name):
        import mlx.core as mx
        import mlx_whisper  # noqa: F401  (registers the transcribe module)
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.decoding import DecodingOptions, decode

        self.mx, self.decode = mx, decode
        print(f"Whisper {model_name} on {mx.default_device()}")
        self.model = sys.modules["mlx_whisper.transcribe"].ModelHolder.get_model(model_name, mx.float16)
        self.options = DecodingOptions(language="en", temperature=0.0, without_timestamps=True, fp16=True)
        # As mlx_whisper.transcribe() makes its first window
        self.mel = lambda audio: pad_or_trim(
            log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels, padding=N_SAMPLES)[:N_FRAMES],
            N_FRAMES, axis=-2).astype(mx.float16)

    def __call__(self, clips):
        results = self.decode(self.model, self.mx.stack([self.mel(c) for c in clips]), self.options)
        return [(r.text.strip(), r.no_speech_prob, r.avg_logprob) for r in results]


# --- Results -----------------------------------------------------------------

def keyword_files(data_dir, keywords):
    return [(kw, f) for kw in keywords for f in sorted((data_dir / kw).glob("*.wav"))]


def rel(path):
    return str(path.relative_to(HOSTSW_DIR)) if path.is_relative_to(HOSTSW_DIR) else str(path)


def read_results(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check_clips(data_dir, keywords, model_name, limit=None):
    """Checks every clip not in the results CSV yet, appending as it goes."""
    path = data_dir / RESULTS_NAME
    done = {r["file"] for r in read_results(path)}
    todo = [(kw, f) for kw, f in keyword_files(data_dir, keywords) if rel(f) not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(done)} clips checked already, {len(todo)} to check")
    if not todo:
        return
    transcribe = Transcriber(model_name)
    new = not path.exists()
    t0 = time.time()
    with open(path, "a", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            audio = [sf.read(str(f), dtype="float32")[0] for _, f in batch]
            for (kw, f), a, (text, no_speech, logprob) in zip(batch, audio, transcribe(audio)):
                writer.writerow({"file": rel(f), "keyword": kw, "whisper_text": text,
                                 "whisper_class": whisper_class(text, kw),
                                 "no_speech_prob": f"{no_speech:.3f}", "avg_logprob": f"{logprob:.3f}",
                                 **energy_measures(a)})
            out.flush()
            n = i + len(batch)
            if n % 400 < BATCH_SIZE or n == len(todo):
                rate = (time.time() - t0) / n
                print(f"  {n}/{len(todo)} clips, {rate:.2f} s per clip, ~{rate * (len(todo) - n) / 60:.0f} min left",
                      flush=True)


def add_short_flags(rows):
    """'short': word under SHORT_FRACTION of its keyword's median word length."""
    medians = {kw: float(np.median([float(r["word_s"]) for r in rows if r["keyword"] == kw]))
               for kw in {r["keyword"] for r in rows}}
    for r in rows:
        r["short"] = int(float(r["word_s"]) < SHORT_FRACTION * medians[r["keyword"]])
    return medians


def flags(r):
    """A result row's flags, Whisper's first."""
    return ([r["whisper_class"]] if r["whisper_class"] != "ok" else []) + [f for f in ENERGY_FLAGS if int(r[f])]


def report(rows):
    medians = add_short_flags(rows)
    print(f"\n{len(rows)} clips checked")
    print("Word length (energy) percentiles 2 / 10 / 50 / 90 %, s, and the 'short' limit:")
    for kw in KEYWORDS:
        w = [float(r["word_s"]) for r in rows if r["keyword"] == kw]
        if w:
            p = np.percentile(w, [2, 10, 50, 90])
            print(f"  {kw:9s} {p[0]:.2f} / {p[1]:.2f} / {p[2]:.2f} / {p[3]:.2f}   short < {SHORT_FRACTION * medians[kw]:.2f}")
    print("\nFlags per keyword (a clip can have several):")
    names = WHISPER_FLAGS + ENERGY_FLAGS
    print(f"  {'':9s} {'clips':>6s} " + " ".join(f"{n:>9s}" for n in names) + f" {'any':>9s}")
    for kw in KEYWORDS:
        sub = [r for r in rows if r["keyword"] == kw]
        if not sub:
            continue
        counts = Counter(f for r in sub for f in flags(r))
        flagged = sum(bool(flags(r)) for r in sub)
        print(f"  {kw:9s} {len(sub):6d} " + " ".join(f"{counts[n]:9d}" for n in names)
              + f" {flagged:6d} ({100 * flagged / len(sub):.0f}%)")
    print("\nWhisper class vs energy flags (clips):")
    print(f"  {'':9s} {'clips':>6s} " + " ".join(f"{n:>9s}" for n in ENERGY_FLAGS) + f" {'none':>9s}")
    for cls in ["ok"] + WHISPER_FLAGS:
        sub = [r for r in rows if r["whisper_class"] == cls]
        counts = Counter(f for r in sub for f in ENERGY_FLAGS if int(r[f]))
        none = sum(not any(int(r[f]) for f in ENERGY_FLAGS) for r in sub)
        print(f"  {cls:9s} {len(sub):6d} " + " ".join(f"{counts[n]:9d}" for n in ENERGY_FLAGS) + f" {none:9d}")
    print("\nMost common transcripts that aren't the keyword:")
    for kw in KEYWORDS:
        c = Counter(r["whisper_text"].lower().strip(" .!?,") for r in rows
                    if r["keyword"] == kw and r["whisper_class"] != "ok")
        print(f"  {kw:9s} " + ", ".join(f"'{t}' {n}" for t, n in c.most_common(15)))


def write_spotcheck(rows, data_dir, per_flag, controls, seed):
    """A listening file of flagged clips (per keyword and flag) and unflagged controls."""
    rng = np.random.default_rng(seed)
    picked, seen = [], set()
    for kw in KEYWORDS:
        sub = [r for r in rows if r["keyword"] == kw]
        groups = [(f, [r for r in sub if f in flags(r)]) for f in WHISPER_FLAGS + ENERGY_FLAGS]
        groups.append(("control", [r for r in sub if not flags(r)]))
        for name, group in groups:
            group = [r for r in group if r["file"] not in seen]
            n = controls if name == "control" else per_flag
            for i in sorted(rng.choice(len(group), size=min(n, len(group)), replace=False)) if group else []:
                picked.append((name, group[i]))
                seen.add(group[i]["file"])

    out = data_dir / SPOTCHECK_NAME
    audio, labels, table, t = [], [], [], 0.0
    gap = np.zeros(round(SPOTCHECK_GAP_S * SAMPLE_RATE_HZ), dtype=np.float32)
    for n, (name, r) in enumerate(picked):
        a, _ = sf.read(str(HOSTSW_DIR / r["file"]), dtype="float32")
        all_flags = ", ".join(flags(r)) or "none"
        labels.append((t, t + len(a) / SAMPLE_RATE_HZ,
                       f'{n:03d} {r["keyword"]} [{name}] | whisper "{r["whisper_text"]}" | {all_flags} | '
                       f'{pathlib.Path(r["file"]).name}'))
        table.append({"clip": n, "start_s": f"{t:.2f}", "group": name, **{k: r[k] for k in FIELDS + ["short"]}})
        audio += [a, gap]
        t += (len(a) + len(gap)) / SAMPLE_RATE_HZ
    sf.write(str(out) + ".wav", np.concatenate(audio), SAMPLE_RATE_HZ, subtype="PCM_16")
    with open(str(out) + "_labels.txt", "w") as f:
        f.writelines(f"{a:.3f}\t{b:.3f}\t{text}\n" for a, b, text in labels)
    with open(str(out) + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"\nWrote {out}.wav ({len(picked)} clips, {t / 60:.1f} min), {out.name}_labels.txt (Audacity: "
          f"File > Import > Labels) and {out.name}.csv")
    print("  per group: " + ", ".join(f"{kw} {name} {c}" for (kw, name), c in
                                      Counter((r["keyword"], name) for name, r in picked).items()))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR,
                        help="Speech Commands dir (default: dataset/)")
    parser.add_argument("--keywords", nargs="+", choices=KEYWORDS, default=KEYWORDS)
    parser.add_argument("--model", default=WHISPER_MODEL, help=f"Whisper model (default: {WHISPER_MODEL})")
    parser.add_argument("--limit", type=int, help="check at most this many new clips (for a quick test)")
    parser.add_argument("--report-only", action="store_true", help="don't check new clips, just report")
    parser.add_argument("--per-flag", type=int, default=8, help="spot-check clips per keyword and flag (default: 8)")
    parser.add_argument("--controls", type=int, default=5, help="unflagged spot-check clips per keyword (default: 5)")
    parser.add_argument("--seed", type=int, default=0, help="spot-check sampling seed (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if not args.report_only:
        check_clips(data_dir, args.keywords, args.model, args.limit)
    rows = [r for r in read_results(data_dir / RESULTS_NAME) if r["keyword"] in args.keywords]
    if not rows:
        sys.exit(f"no results in {data_dir / RESULTS_NAME}")
    for r in rows:   # from the stored transcript, so the class rules can change without re-running Whisper
        r["whisper_class"] = whisper_class(r["whisper_text"], r["keyword"])
    report(rows)
    write_spotcheck(rows, data_dir, args.per_flag, args.controls, args.seed)


if __name__ == "__main__":
    main()
