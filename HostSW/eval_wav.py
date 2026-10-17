"""
Run the whole host evaluation on one WAV in one go:
  1. run_model_on_wav.py   model over the WAV         -> <wav stem>.csv
  2. score_results.py      recall / false accepts     -> <wav stem>_detections.csv
                           (skipped if there's no <wav stem>_truth.csv)
  3. view_results.py       playback + animated results

Stops at the first step that fails. Each script runs with the same Python
interpreter as this one, so run it from the kws env.

Usage:
    python eval_wav.py -w eval_recordings/eval_01_speech.wav
    python eval_wav.py -w eval_recordings/eval_01_speech.wav -m tflite_models/kws_v7_float32_softmax.tflite
    python eval_wav.py -w eval_recordings/eval_01_speech.wav --threshold 0.6 --avg 1.0 --no-view
"""

import argparse
import pathlib
import subprocess
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "tflite_models" / "kws_v7_int8_softmax.tflite"


def run_step(name, script, script_args):
    print(f"\n=== {name}: {script} {' '.join(script_args)}", flush=True)
    result = subprocess.run([sys.executable, str(SCRIPT_DIR / script)] + script_args)
    if result.returncode != 0:
        sys.exit(f"{name} failed (exit code {result.returncode})")


def parse_args():
    parser = argparse.ArgumentParser(description="Run model -> score -> view on one WAV.")
    parser.add_argument("-w", "--wav", required=True, help="16 kHz 16-bit .wav to evaluate")
    parser.add_argument("-m", "--model", default=str(DEFAULT_MODEL), help=f"tflite model (default: {DEFAULT_MODEL})")
    parser.add_argument("-s", "--stride", help="window stride in seconds (run_model_on_wav.py default if omitted)")
    parser.add_argument("--avg", help="score_results.py smoothing window, s")
    parser.add_argument("--threshold", help="score_results.py detection threshold")
    parser.add_argument("--suppress", help="score_results.py suppression time, s")
    parser.add_argument("--no-sweep", action="store_true", help="skip the threshold sweep table")
    parser.add_argument("--no-view", action="store_true", help="don't open the viewer")
    return parser.parse_args()


def main():
    args = parse_args()
    wav_path = pathlib.Path(args.wav)
    results_path = wav_path.with_suffix(".csv")
    truth_path = wav_path.with_name(wav_path.stem + "_truth.csv")

    if not wav_path.exists():
        sys.exit(f"{wav_path} not found")

    model_args = ["-m", args.model, "-w", str(wav_path), "-o", str(results_path)]
    if args.stride:
        model_args += ["-s", args.stride]
    run_step("Run model", "run_model_on_wav.py", model_args)

    if truth_path.exists():
        score_args = ["-r", str(results_path), "-t", str(truth_path)]
        for flag, value in (("--avg", args.avg), ("--threshold", args.threshold), ("--suppress", args.suppress)):
            if value:
                score_args += [flag, value]
        if not args.no_sweep:
            score_args.append("--sweep")
        run_step("Score", "score_results.py", score_args)
    else:
        print(f"\n=== Score: skipped, no {truth_path} (label the WAV first, see eval_recordings/README.md)")
        stale = wav_path.with_name(wav_path.stem + "_detections.csv")
        if stale.exists():
            print(f"[warn] {stale} is from an earlier run and may not match these results")

    if not args.no_view:
        run_step("View", "view_results.py", ["-w", str(wav_path), "-r", str(results_path)])


if __name__ == "__main__":
    main()
