# Board-mic evaluation recordings

Held-out test set recorded on the B-U585I-IOT02A mic. It's used to measure **keyword recall** and
**false accepts per minute** on realistic, continuous audio.

**Never train on these files.** Keep training recordings somewhere else, or the scores stop meaning
anything.

Each recording is a pair of files:
- `eval_<nn>_<desc>.wav`: 16 kHz, 16-bit mono, from `receive_and_play.py -o`
- `eval_<nn>_<desc>_truth.csv`: keyword spans, converted from Audacity labels

Generated files (`eval_*.csv` results, `*_detections.csv`) depend on the model and settings. They
can be regenerated at any time, so don't commit them.

## 1. Record

```
python HostSW/receive_and_play.py -p /dev/tty.usbmodemXXXX -o HostSW/eval_recordings/eval_02_kitchen.wav
```

Aim for about 5 minutes per file, with roughly 10 of each keyword (`forward`, `backward`, `stop`).
Mix:
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
| `forward`, `backward`, `stop` | a clearly spoken keyword, from the start of the word to its end |
| `ignore` | audio that shouldn't count as a hit or a false accept: a cut-off or mumbled keyword, "stopped" / "forwards", a keyword said by the TV, a cough right on top of a word |

Spans don't need to be exact. Scoring allows about 1 s of slack after the word.

1. Open the WAV in Audacity. For finding word edges, the spectrogram view is easier to read:
   click the track name, then **Spectrogram** (or **Multi-view**).
2. Play it. When you hear a keyword, drag across the word, press **Cmd+B** (Ctrl+B on Windows or
   Linux), type the label and press Enter. Click a label to select its region and replay it; drag a
   label's edges to adjust it.
3. **File → Export → Export Labels…** and save as `eval_<nn>_<desc>_labels.txt` next to the WAV.
4. Convert it to the truth CSV (labels are checked; a typo is rejected with its line number):
   ```
   python HostSW/audacity_labels_to_truth.py -i HostSW/eval_recordings/eval_02_kitchen_labels.txt
   ```
5. To fix labels later: **File → Import → Labels…** with the `_labels.txt` file, edit, then export
   and convert again. `--reverse` rebuilds a `_labels.txt` file from a truth CSV if you've lost it.

## 3. Run the model, score it, look at the results

All three steps in one command (options: `-m` model, `--threshold`/`--avg`/`--suppress`, `--no-view`):
```
cd HostSW
python eval_wav.py -w eval_recordings/eval_02_kitchen.wav
```

Or step by step:
```
cd HostSW
python run_model_on_wav.py -m tflite_models/kws_v7_int8_softmax.tflite -w eval_recordings/eval_02_kitchen.wav
python score_results.py -r eval_recordings/eval_02_kitchen.csv --sweep
python view_results.py -w eval_recordings/eval_02_kitchen.wav
```

`score_results.py` prints recall per keyword, false accepts per minute and per hour, and the times
of every miss and false accept. `view_results.py` then shows each detection as a triangle (green =
hit, red = false accept) so you can listen to what went wrong.
