"""
Play a WAV file while animating the KWS model output from run_model_on_wav.py,
so you can hear the audio and check the predictions against it.

Three panels share a scrolling time axis, with a playback cursor:
  1. waveform
  2. predicted label per window (y axis = class name)
  3. per-class softmax probabilities
Each window's prediction is plotted at the window's centre (start + 0.5 s).
If a ground-truth CSV is available (from make_test_wav.py), the true clip
spans are shaded on the waveform and drawn as a band on the true label's row,
so a correct prediction is the line running through the band.

Sync: the sounddevice output callback advances the playback position, and the
animation just reads it (minus the output latency), so the cursor follows what
is actually heard and can't drift from it.

Controls: space pause/resume, left/right seek 1 s, click a plot to seek, home restart.

Usage:
    python view_results.py -w capture.wav                     # reads capture.csv (+ capture_truth.csv if present)
    python view_results.py -w capture.wav -r results.csv -t truth.csv --span 10

Requirements:
    pip install numpy matplotlib sounddevice
"""

import argparse
import csv
import pathlib
import sys
import threading
import time
import wave
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

WINDOW_LEN_S = 1.0
ENVELOPE_BIN_S = 0.01   # waveform drawn as a min/max envelope per 10 ms
FRAME_INTERVAL_MS = 33  # ~30 fps
SEEK_STEP_S = 1.0

# Samples per audio callback. With the default (0, let the host choose), CoreAudio
# asks for ~14 samples per callback at 16 kHz, i.e. >1000 Python callbacks/s. Each
# needs the GIL, which the main thread holds while matplotlib redraws (~30-40 ms
# per frame), so callbacks are missed and ~20% of the audio is dropped (garbled
# playback). 1024 samples = 64 ms per callback leaves plenty of headroom.
AUDIO_BLOCKSIZE = 1024

HELP_TEXT = "space: pause/resume    ←/→: seek 1 s    click: seek    home: restart"


@dataclass
class Results:
    labels: list          # class names, in model output order
    start_s: np.ndarray
    end_s: np.ndarray
    label_idx: np.ndarray
    probs: np.ndarray     # (n_windows, n_labels)

    @property
    def centre_s(self):
        return self.start_s + WINDOW_LEN_S / 2


def load_wav(path):
    """Returns (float32 samples in [-1, 1), sample rate). Multi-channel files use channel 0."""
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        channels = wav.getnchannels()
        raw = wav.readframes(wav.getnframes())
    if sample_width != 2:
        sys.exit(f"{path}: only 16-bit PCM is supported, got {sample_width * 8}-bit")
    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)[:, 0]
    return samples.astype(np.float32) / 32768.0, sample_rate


def load_results(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        prob_cols = [c for c in reader.fieldnames if c.startswith("p_")]
        rows = list(reader)
    if not rows:
        sys.exit(f"{path}: no result rows")
    return Results(
        labels=[c[len("p_"):] for c in prob_cols],
        start_s=np.array([float(r["start_s"]) for r in rows]),
        end_s=np.array([float(r["end_s"]) for r in rows]),
        label_idx=np.array([int(r["label_idx"]) for r in rows]),
        probs=np.array([[float(r[c]) for c in prob_cols] for r in rows]),
    )


def load_truth(path):
    """Returns a list of (start_s, end_s, label)."""
    with open(path, newline="") as f:
        return [(float(r["start_s"]), float(r["end_s"]), r["label"]) for r in csv.DictReader(f)]


class ResultsView:
    """The figure. draw_at(t) updates it for playback time t; it knows nothing
    about audio, so it can be driven by the player or rendered headless."""

    def __init__(self, signal, sample_rate, results, truth=None, span_s=6.0):
        self.results = results
        self.truth = truth or []
        self.span_s = span_s
        self.duration_s = len(signal) / sample_rate
        self.centres = results.centre_s

        n_labels = len(results.labels)
        cmap = matplotlib.colormaps["tab10"]
        colours = [cmap(k % 10) for k in range(n_labels)]

        self.fig, (self.ax_wave, self.ax_label, self.ax_prob) = plt.subplots(
            3, 1, sharex=True, figsize=(13, 8), height_ratios=[1, 1.4, 1.2]
        )

        # 1. Waveform envelope
        bin_len = max(1, round(ENVELOPE_BIN_S * sample_rate))
        n_bins = len(signal) // bin_len
        binned = signal[: n_bins * bin_len].reshape(n_bins, bin_len)
        t_bins = (np.arange(n_bins) + 0.5) * bin_len / sample_rate
        self.ax_wave.fill_between(t_bins, binned.min(axis=1), binned.max(axis=1), color="0.3", lw=0)
        self.ax_wave.set_ylim(-1, 1)
        self.ax_wave.set_ylabel("amplitude")

        # 2. Predicted label, with truth bands on each clip's true-label row
        for start_s, end_s, label in self.truth:
            if label not in results.labels:
                print(f"[warn] truth label '{label}' is not a model label, not drawn")
                continue
            k = results.labels.index(label)
            self.ax_wave.axvspan(start_s, end_s, color=colours[k], alpha=0.2, lw=0)
            self.ax_label.add_patch(
                Rectangle((start_s, k - 0.4), end_s - start_s, 0.8, color=colours[k], alpha=0.3, lw=0)
            )
        self.ax_label.step(self.centres, results.label_idx, where="mid", color="k", lw=1.5)
        self.ax_label.scatter(
            self.centres, results.label_idx, s=12, zorder=3,
            c=[colours[k] for k in results.label_idx],
        )
        self.ax_label.set_yticks(range(n_labels), results.labels)
        self.ax_label.set_ylim(n_labels - 0.5, -0.5)   # first label at the top
        self.ax_label.set_ylabel("predicted label")
        self.ax_label.grid(axis="y", alpha=0.3)

        # 3. Per-class probabilities
        prob_lines = [
            self.ax_prob.plot(self.centres, results.probs[:, k], color=colours[k], lw=1.2)[0]
            for k in range(n_labels)
        ]
        self.ax_prob.set_ylim(-0.02, 1.02)
        self.ax_prob.set_ylabel("probability")
        self.ax_prob.set_xlabel("time (s), windows plotted at their centre")
        # Labels passed explicitly: matplotlib hides auto-collected labels that
        # start with '_', which would drop '_background_noise_'
        self.ax_prob.legend(prob_lines, results.labels, loc="upper left", ncol=n_labels, fontsize="small")
        self.ax_prob.grid(alpha=0.3)

        self.cursors = [
            ax.axvline(0, color="red", lw=1.2, zorder=5)
            for ax in (self.ax_wave, self.ax_label, self.ax_prob)
        ]
        self.status = self.fig.suptitle("", family="monospace", fontsize=11)
        self.fig.text(0.5, 0.01, HELP_TEXT, ha="center", fontsize=9, color="0.4")
        self.fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    def window_at(self, t):
        """Index of the window whose centre is nearest t."""
        i = int(np.searchsorted(self.centres, t))
        if i == 0:
            return 0
        if i == len(self.centres):
            return i - 1
        return i if self.centres[i] - t < t - self.centres[i - 1] else i - 1

    def truth_at(self, t):
        for start_s, end_s, label in self.truth:
            if start_s <= t < end_s:
                return label
        return "-"

    def draw_at(self, t):
        for cursor in self.cursors:
            cursor.set_xdata([t, t])
        self.ax_wave.set_xlim(t - self.span_s / 2, t + self.span_s / 2)

        r = self.results
        i = self.window_at(t)
        k = r.label_idx[i]
        status = (
            f"t = {t:6.2f} / {self.duration_s:.2f} s  |  window {r.start_s[i]:.2f}–{r.end_s[i]:.2f} s"
            f" → {r.labels[k]} (p={r.probs[i, k]:.2f})"
        )
        if self.truth:
            status += f"  |  truth: {self.truth_at(t)}"
        self.status.set_text(status)


class Player:
    """Plays a mono float signal through sounddevice. The output callback owns
    the playback position, so time() reflects what has actually been sent to
    the sound card."""

    def __init__(self, signal, sample_rate):
        self.signal = signal
        self.sample_rate = sample_rate
        self.pos = 0
        self.paused = False
        self.lock = threading.Lock()
        # Whether the callback is feeding real audio, and when that last changed.
        # Starts as "feeding since long ago" so the stream start uses the full latency.
        self.feeding = True
        self.feeding_changed_at = 0.0
        self.stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="float32", callback=self._callback,
            blocksize=AUDIO_BLOCKSIZE,
        )

    def _callback(self, outdata, frames, time_info, status):
        with self.lock:
            chunk = self.signal[0:0] if self.paused else self.signal[self.pos : self.pos + frames]
            self.pos += len(chunk)
            if (len(chunk) > 0) != self.feeding:
                self.feeding = len(chunk) > 0
                self.feeding_changed_at = time.monotonic()
        outdata[: len(chunk), 0] = chunk
        outdata[len(chunk) :] = 0

    @property
    def duration_s(self):
        return len(self.signal) / self.sample_rate

    def time(self):
        """Playback time in seconds of the audio currently being heard.

        pos is what has been handed to the sound card; it is heard stream.latency
        later. When feeding stops (pause / end) the already-queued audio still
        drains, so the lag shrinks to 0 over one latency period; when feeding
        resumes, the lag grows back while the queued silence drains."""
        latency = self.stream.latency
        since_change = time.monotonic() - self.feeding_changed_at
        if self.feeding:
            lag = min(latency, since_change)
        else:
            lag = max(0.0, latency - since_change)
        t = self.pos / self.sample_rate - lag
        return min(max(t, 0.0), self.duration_s)

    def seek(self, t):
        with self.lock:
            self.pos = int(min(max(t, 0.0), self.duration_s) * self.sample_rate)

    def toggle_pause(self):
        with self.lock:
            if self.paused and self.pos >= len(self.signal):
                self.pos = 0   # resuming at the end restarts
            self.paused = not self.paused

    def start(self):
        self.stream.start()

    def close(self):
        self.stream.stop()
        self.stream.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play a WAV and animate the KWS model results from run_model_on_wav.py in sync."
    )
    parser.add_argument("-w", "--wav", required=True, help="the .wav file the model was run on")
    parser.add_argument(
        "-r", "--results",
        help="results CSV from run_model_on_wav.py (default: the WAV path with a .csv suffix)",
    )
    parser.add_argument(
        "-t", "--truth",
        help="ground-truth CSV from make_test_wav.py (default: <wav stem>_truth.csv if it exists)",
    )
    parser.add_argument("--span", type=float, default=6.0, help="visible seconds (default: 6)")
    return parser.parse_args()


def main():
    args = parse_args()
    wav_path = pathlib.Path(args.wav)
    results_path = pathlib.Path(args.results) if args.results else wav_path.with_suffix(".csv")

    truth_path = None
    if args.truth:
        truth_path = pathlib.Path(args.truth)
    elif wav_path.with_name(wav_path.stem + "_truth.csv").exists():
        truth_path = wav_path.with_name(wav_path.stem + "_truth.csv")

    signal, sample_rate = load_wav(wav_path)
    results = load_results(results_path)
    truth = load_truth(truth_path) if truth_path else None
    print(f"WAV {wav_path} ({len(signal) / sample_rate:.2f}s), results {results_path} ({len(results.start_s)} windows)"
          + (f", truth {truth_path}" if truth_path else ""))

    # Free up keys that matplotlib binds to its own navigation by default
    for keymap, keys in (("keymap.back", ["left"]), ("keymap.forward", ["right"]), ("keymap.home", ["home"])):
        plt.rcParams[keymap] = [k for k in plt.rcParams[keymap] if k not in keys]

    view = ResultsView(signal, sample_rate, results, truth, span_s=args.span)
    player = Player(signal, sample_rate)

    def on_key(event):
        if event.key == " ":
            player.toggle_pause()
        elif event.key == "left":
            player.seek(player.time() - SEEK_STEP_S)
        elif event.key == "right":
            player.seek(player.time() + SEEK_STEP_S)
        elif event.key == "home":
            player.seek(0)

    def on_click(event):
        toolbar = view.fig.canvas.toolbar
        if toolbar is not None and toolbar.mode:
            return   # zoom/pan tool active
        if event.button == 1 and event.inaxes is not None and event.xdata is not None:
            player.seek(event.xdata)

    def on_frame(_):
        # Start audio on the first frame, once the window is actually up
        if not player.stream.active:
            player.start()
        view.draw_at(player.time())

    view.fig.canvas.mpl_connect("key_press_event", on_key)
    view.fig.canvas.mpl_connect("button_press_event", on_click)
    anim = FuncAnimation(view.fig, on_frame, interval=FRAME_INTERVAL_MS, cache_frame_data=False)  # noqa: F841

    try:
        plt.show()
    finally:
        player.close()


if __name__ == "__main__":
    main()
