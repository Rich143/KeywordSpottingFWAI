# Board-mic evaluation recordings

Held-out recordings made on the B-U585I-IOT02A mic. They're used to measure **keyword recall** and
**false accepts per hour** on realistic, continuous audio.

**Never train on these files.** Keep training recordings somewhere else, or the scores stop meaning
anything.

There are two splits:
- **`val/`**: for comparing models and making decisions. The notebook scores it after every training
  run. Both existing recordings are here.
- **`test/`**: held back, and empty so far. It's only scored when asked for by name (`--split test`,
  or `RUN_TEST_EVAL = True` in the notebook). Look at it rarely, e.g. once per finished model, and
  never tune on it, or it stops being an unbiased test.

Put new recordings in `test/` until it has a few sessions' worth, e.g. every third recording session,
in different rooms. Record them as usual; the steps are the same, just in that folder.

Each recording is a pair of files:
- `eval_<nn>_<desc>.wav`: 16 kHz, 16-bit mono, from `receive_and_play.py -o`
- `eval_<nn>_<desc>_truth.csv`: keyword spans, converted from Audacity labels

Generated files (`eval_*.csv` results, `*_detections.csv`) depend on the model and settings. They
can be regenerated at any time, so don't commit them.

## 1. Record

```
python HostSW/receive_and_play.py -p /dev/tty.usbmodemXXXX -o HostSW/eval_recordings/val/eval_02_kitchen.wav
```

Aim for about 5 minutes per file, with roughly 10 of each keyword (`forward`, `backward`, `stop`).
[`val/script_eval_02.md`](val/script_eval_02.md) is a ready-made script to read aloud that covers all of the
following. Mix:
- keywords said on their own, with pauses around them
- keywords inside sentences: "go forward a bit", "please stop there", "back up, backward"
- fast and slow, quiet and loud, at about 0.3 m, 1 m and 2 m from the board
- confusable non-keywords: "four", "follow", "top", "back", "forwarding", "stopped"
- about 30 s of silence (room tone)
- about 1 min with a TV, podcast or music playing in the background
- a variety of recording sessions and rooms across files (other speakers too, if possible)

## 2. Label in Audacity

Label **only** keywords, plus `ignore` where needed. Everything unlabelled counts as negative audio.

| Label | Use for |
|---|---|
| `forward`, `backward`, `stop` | a clearly spoken keyword, from the start of the word to its end. "forwards" / "backwards" count as `forward` / `backward` |
| `ignore` | audio that shouldn't count as a hit or a false accept: a cut-off or mumbled keyword, "stopped" / "stopping", a keyword said by the TV, a cough right on top of a word |

Spans don't need to be exact. Scoring allows about 1 s of slack after the word.

1. Open the WAV in Audacity. For finding word edges, the spectrogram view is easier to read:
   click the track name, then **Spectrogram** (or **Multi-view**).
2. Play it. When you hear a keyword, drag across the word, press **Cmd+B** (Ctrl+B on Windows or
   Linux), type the label and press Enter. Click a label to select its region and replay it; drag a
   label's edges to adjust it.
3. **File → Export → Export Labels…** and save as `eval_<nn>_<desc>_labels.txt` next to the WAV.
4. Convert it to the truth CSV (labels are checked; a typo is rejected with its line number):
   ```
   python HostSW/EvalScripts/audacity_labels_to_truth.py -i HostSW/eval_recordings/val/eval_02_kitchen_labels.txt
   ```
5. To fix labels later: **File → Import → Labels…** with the `_labels.txt` file, edit, then export
   and convert again. `--reverse` rebuilds a `_labels.txt` file from a truth CSV if you've lost it.

## 3. Run the model, score it, look at the results

All three steps in one command (options: `-m` model, `--threshold`/`--avg`/`--suppress`, `--no-view`):
```
cd HostSW
python EvalScripts/eval_wav.py -w eval_recordings/val/eval_02_kitchen.wav
```

Or step by step:
```
cd HostSW
python EvalScripts/run_model_on_wav.py -m tflite_models/kws_v7_int8_softmax.tflite -w eval_recordings/val/eval_02_kitchen.wav
python EvalScripts/score_results.py -r eval_recordings/val/eval_02_kitchen.csv --sweep
python EvalScripts/view_results.py -w eval_recordings/val/eval_02_kitchen.wav
```

## 4. Snapshot a model version

When you have a new model, score it on every labelled recording of a split. The notebook does this
for val after every training run.
```
cd HostSW
python EvalScripts/snapshot_eval.py -v v11 -m tflite_models/kws_v11_int8_softmax.tflite --notes "what changed"
python EvalScripts/snapshot_eval.py -v v11 -m tflite_models/kws_v11_int8_softmax.tflite --split test
```
It converts any `_labels.txt` newer than its `_truth.csv`, runs the model, and scores with the same
code as `score_results.py`.

It writes **one summary row per run** to `<split>/eval_history.csv`, committed so all versions can
be compared, with `recording = ALL`, pooled over the split's recordings. The per-recording results
are printed and saved in MLflow. Rows from before this format also have per-recording rows; they're
kept. Each run also logs the split's whole history to MLflow as sortable tables in the run's
**Artifacts** tab, in `reports/board_<split>_history/`: the default threshold, one per false accept
budget, and run settings (see the public set's README).

Each row also has recall at false accept budgets, as described in the
[public set's README](../eval_recordings_public/README.md#recall-at-a-false-accept-budget). This set
has only about 5.5 minutes of negative audio, so one false accept is about 11 per hour, and the
script marks most of these figures as resting on too few events. Use the public set's numbers for
that comparison, and this set's recall on your own mic alongside them.

To leave a recording out (for example a broken one), add its filename or a glob to
`<split>/eval_ignore.txt`, one per line.

Detection settings (`--avg`, `--threshold`, `--suppress`) are stored with each row. Compare versions
at the same settings, and on the same set of recordings, since adding recordings changes the
numbers. Re-running a version with the same settings is refused unless `--replace`.

**MLflow:** every row has `mlflow_run_id` and `run_type`.
- `training`: the row came from the notebook. The metrics (`board_val_recall`,
  `board_val_fa_per_hour`, …) are in that training run, and the model must be the one the run
  logged.
- `standalone`: the row came from running the script yourself, e.g. to try another threshold. It
  creates its own MLflow eval run, named like `v11 board-val eval` and nested under the v11
  training run.

---

`score_results.py` prints recall per keyword, false accepts per minute and per hour, and the times
of every miss and false accept. `view_results.py` then shows each detection as a triangle (green =
hit, red = false accept) so you can listen to what went wrong.
