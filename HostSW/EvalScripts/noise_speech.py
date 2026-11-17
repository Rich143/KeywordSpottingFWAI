"""
Find speech in the background-noise datasets (dataset_noise/, download_noise.py), so only
speech-free audio is used as background noise: speech there would be unlabelled 'other' audio,
and could even contain a keyword.

A dataset's own labels decide where it has them, and YAMNet (an AudioSet sound-event classifier,
run here on every file) where it doesn't. Any speech-like class counts (SPEECH_CLASSES: speech,
conversation, narration, whispering, shouting, singing, rapping, chatter / babble, TV, radio):
  Sounds of Home  its PANNs (Cnn14) predictions, <stamp>_light.json: the 7 most likely classes
                  every 0.213 s, made on the original audio. The dataset authors muted the speech
                  they found (to digital silence, ~0.5-1 s beyond the labelled frames), so these
                  show where. A frame is speech if a speech class is among its 7
  MUSAN music     its vocals annotation (manifest 'vocals', used by make_public_eval.py), and YAMNet
                  windows >= 0.5: brief spoken samples in 4.6% of instrumental files
  MUSAN noise     no labels, and speech despite its description (chatter, crowds): YAMNet >= 0.2
  DEMAND          no labels; announcements and passengers on the bus / metro: YAMNet >= 0.5
YAMNet gives a score per 0.96 s window every 0.48 s (YAMNET_THRESHOLDS per dataset). Real speech
scores > 0.3 in ~95% of its windows; instrumental music and noise mostly stay below 0.2.
Checked by ear (Nov 2026): YAMNet's MUSAN detections were speech. In Sounds of Home and DEMAND, its
detections the labels didn't have (up to 1.0 in Sounds of Home, 0.2-0.8 in DEMAND's quieter
environments) were at most faint, unintelligible speech from elsewhere, which is fine as background.
So Sounds of Home goes by its labels alone, and DEMAND only counts confident detections.
A file's speech spans (speech_spans()) are its speech frames / windows, widened by a margin each
side and merged. The eval build cuts Sounds of Home and DEMAND into excerpts between the spans;
MUSAN files are used whole, so one with any span isn't used.

Output: dataset_noise/speech.npz (gitignored, like the audio): per manifest file, the YAMNet
speech score per window (float16) and, for Sounds of Home, the PANNs speech flag per frame.
Re-running only analyses files that aren't in it yet (--redo for all).
YAMNet (14 MB, Apache 2.0) is downloaded once into dataset_noise/yamnet/ and checked against
YAMNET_SHA256.

Usage (from HostSW/):
    python EvalScripts/noise_speech.py
    python EvalScripts/noise_speech.py --summary
"""

import argparse
import csv
import hashlib
import json
import math
import pathlib
import shutil
import sys
import tarfile
import time
import urllib.request

import numpy as np
import scipy.signal
import soundfile as sf

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
HOSTSW_DIR = SCRIPT_DIR.parent
DEFAULT_NOISE_DIR = HOSTSW_DIR / "dataset_noise"
SCORES_NAME = "speech.npz"

YAMNET_URL = "https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed"
YAMNET_SHA256 = "b80da2a1a56926fb0767205051a200dd7b3beaf3ea1ea126c42a53943996e5e0"
YAMNET_RATE_HZ = 16000
YAMNET_HOP_S = 0.48
YAMNET_WINDOW_S = 0.96
# Long files go through YAMNet in chunks of this many windows (10 min), overlapping by one window
# and starting on a window boundary, so the scores are the same as in one pass. In one pass, an
# hour-long file grew TensorFlow's memory until macOS killed the process
YAMNET_CHUNK_WINDOWS = 1250
YAMNET_MIN_SAMPLES = 15600   # one window: 0.96 s + a 25 ms STFT frame - its 10 ms hop
PANNS_HOP_S = 10240 / 48000      # one prediction per 32 frames of 320 samples at 48 kHz
PANNS_WINDOW_S = PANNS_HOP_S

# AudioSet classes that are (or usually carry) speech. Both models use AudioSet's names;
# YAMNet lacks the male / female speech and singing classes
SPEECH_CLASSES = [
    "Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
    "Conversation", "Narration, monologue", "Babbling", "Speech synthesizer", "Whispering",
    "Shout", "Bellow", "Yell", "Children shouting", "Children playing",
    "Singing", "Male singing", "Female singing", "Child singing", "Synthetic singing", "Choir", "Chant",
    "Mantra", "Rapping", "Chatter", "Crowd", "Hubbub, speech noise, speech babble", "Television", "Radio",
]
# Speech if a YAMNet window scores at least this, by manifest file prefix (see the docstring).
# Sounds of Home isn't here: its own labels decide. And the margin added each side of every speech
# window / frame: covers the window edges and the Sounds of Home muting, which reaches up to ~1 s
# past the labelled frames
YAMNET_THRESHOLDS = {"musan/noise/": 0.2, "musan/music/": 0.5, "demand/": 0.5}
SPEECH_MARGIN_S = 2.0


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def yamnet_model(noise_dir):
    """Loads YAMNet, downloading it first if needed. Returns (model, class names)."""
    model_dir = noise_dir / "yamnet"
    if not (model_dir / "saved_model.pb").exists():
        archive = noise_dir / "yamnet.tar.gz"
        print(f"Downloading YAMNet to {model_dir}...")
        # Kaggle (where tfhub.dev redirects) answers Python's default User-Agent with an error
        request = urllib.request.Request(YAMNET_URL, headers={"User-Agent": "noise_speech.py"})
        with urllib.request.urlopen(request, timeout=120) as response, open(archive, "wb") as f:
            shutil.copyfileobj(response, f)
        digest = file_sha256(archive)
        if digest != YAMNET_SHA256:
            archive.unlink()
            sys.exit(f"YAMNet download has SHA-256 {digest}, expected {YAMNET_SHA256}: not using it")
        model_dir.mkdir(exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(model_dir, filter="data")
        archive.unlink()
    import tensorflow as tf   # only needed here, and slow to import
    tf.config.set_visible_devices([], "GPU")   # CPU: fast enough, and its memory use stays flat
    model = tf.saved_model.load(str(model_dir))
    with open(model_dir / "assets" / "yamnet_class_map.csv", newline="") as f:
        names = [r["display_name"] for r in csv.DictReader(f)]
    return model, names


def yamnet_speech(model, class_index, path):
    """YAMNet speech score per window: the highest of the speech classes."""
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != YAMNET_RATE_HZ:
        g = math.gcd(rate, YAMNET_RATE_HZ)
        audio = scipy.signal.resample_poly(audio, YAMNET_RATE_HZ // g, rate // g).astype(np.float32)
    hop = round(YAMNET_HOP_S * YAMNET_RATE_HZ)
    chunk = YAMNET_CHUNK_WINDOWS * hop
    parts = []
    for start in range(0, max(1, len(audio)), chunk):
        # Two hops more than the chunk: its last window needs 0.975 s (0.96 s + one STFT frame)
        scores = model(audio[start:start + chunk + 2 * hop])[0].numpy()
        parts.append(scores[:YAMNET_CHUNK_WINDOWS])   # the rest start in the next chunk
    # YAMNet pads the end up to a whole window: as many windows as one pass over the file gives
    windows = 1 + max(0, math.ceil((len(audio) - YAMNET_MIN_SAMPLES) / hop))
    return np.concatenate(parts)[:windows, class_index].max(axis=1).astype(np.float16)


def panns_speech(labels_path):
    """1 per PANNs frame with a speech class among its listed predictions, else 0."""
    speech = set(SPEECH_CLASSES)
    frames = [f for f in json.loads(labels_path.read_text()) if "time" in f]
    return np.array([any(p["class"] in speech for p in f["predictions"]) for f in frames], dtype=np.uint8)


def read_manifest(noise_dir):
    with open(noise_dir / "manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


def load_scores(noise_dir=DEFAULT_NOISE_DIR):
    """{key: array} from speech.npz, keys '<file>|yamnet' and '<file>|panns'."""
    path = noise_dir / SCORES_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run python EvalScripts/noise_speech.py first")
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


def frame_spans(flags, hop_s, window_s, margin_s, duration_s):
    """Merged [start, end) spans covering every flagged frame plus margin_s each side."""
    spans = []
    for i in np.flatnonzero(flags):
        start = max(0.0, i * hop_s - margin_s)
        end = min(duration_s, i * hop_s + window_s + margin_s)
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    return [tuple(s) for s in spans]


def speech_spans(scores, file, duration_s, margin_s=SPEECH_MARGIN_S):
    """Speech spans of one manifest file, each widened by margin_s, merged: Sounds of Home's PANNs
    frames with a speech class, else YAMNet windows scoring >= the file's YAMNET_THRESHOLDS (MUSAN
    music's vocals annotation is checked by the caller). Raises KeyError if the file hasn't been
    analysed."""
    if f"{file}|panns" in scores:
        spans = frame_spans(scores[f"{file}|panns"] > 0, PANNS_HOP_S, PANNS_WINDOW_S, margin_s, duration_s)
    else:
        threshold = next(t for prefix, t in YAMNET_THRESHOLDS.items() if file.startswith(prefix))
        spans = frame_spans(scores[f"{file}|yamnet"] >= threshold, YAMNET_HOP_S, YAMNET_WINDOW_S,
                            margin_s, duration_s)
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(s) for s in merged]


def summary(noise_dir):
    """Speech-free hours per source / category (music with vocals, or unknown vocals, is speech)."""
    scores = load_scores(noise_dir)
    totals = {}
    for r in read_manifest(noise_dir):
        duration = float(r["duration_s"])
        spans = speech_spans(scores, r["file"], duration)
        if r["vocals"] != "no":
            spans = [(0.0, duration)]
        speech_s = sum(e - s for s, e in spans)
        key = r["category"].rsplit("/", 1)[0] if r["source"] == "soh" else r["category"]
        t = totals.setdefault(key, [0, 0, 0.0, 0.0])
        t[0] += 1
        t[1] += not spans
        t[2] += duration / 3600
        t[3] += (duration - speech_s) / 3600
    print(f"{'category':28s} {'files':>5s} {'no speech':>9s} {'hours':>7s} {'speech-free h':>13s}")
    for key, (n, clean, hours, free) in sorted(totals.items()):
        print(f"{key:28s} {n:5d} {clean:9d} {hours:7.1f} {free:13.1f}")


def main():
    parser = argparse.ArgumentParser(description="Find speech in dataset_noise with YAMNet (+ Sounds of Home's labels).")
    parser.add_argument("-d", "--noise-dir", type=pathlib.Path, default=DEFAULT_NOISE_DIR,
                        help=f"noise dataset dir (default: {DEFAULT_NOISE_DIR})")
    parser.add_argument("--redo", action="store_true", help="analyse every file again")
    parser.add_argument("--summary", action="store_true", help="only print speech-free hours per category")
    args = parser.parse_args()
    if args.summary:
        summary(args.noise_dir)
        return

    rows = read_manifest(args.noise_dir)
    path = args.noise_dir / SCORES_NAME
    scores = {} if args.redo or not path.exists() else load_scores(args.noise_dir)
    todo = [r for r in rows if f"{r['file']}|yamnet" not in scores]
    print(f"{len(rows)} files in the manifest, {len(todo)} to analyse")
    if todo:
        model, names = yamnet_model(args.noise_dir)
        class_index = [names.index(c) for c in SPEECH_CLASSES if c in names]
        t0 = time.time()
        for n, r in enumerate(todo, 1):
            audio = args.noise_dir / r["file"]
            scores[f"{r['file']}|yamnet"] = yamnet_speech(model, class_index, audio)
            labels = audio.with_name(audio.stem + "_light.json")
            if r["source"] == "soh" and labels.exists():
                scores[f"{r['file']}|panns"] = panns_speech(labels)
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} files, {time.time() - t0:.0f} s", flush=True)
                np.savez_compressed(path, **scores)   # so an interrupted run keeps its progress
        print(f"Wrote {path}")
    summary(args.noise_dir)


if __name__ == "__main__":
    main()
