# Board-mic training recordings

Training data recorded on the B-U585I-IOT02A mic. It's used alongside Google Speech Commands
(`HostSW/dataset/`, downloaded, not committed) by `TrainModel.ipynb`, so the model also learns this
mic's noise floor, level and DC offset.

The WAVs here are stored with **Git LFS** (see the repo's `.gitattributes`). Install it once with
`brew install git-lfs && git lfs install` **before** committing anything in this folder. Otherwise
the files go in as normal git blobs.

**Never put evaluation recordings here.** `eval_recordings/` is the held-out test set.

## Layout

```
dataset_board/
  _background_noise_/   long recordings with no speech, any length (split into 1 s chunks by the notebook)
  forward/ backward/ stop/  1 s keyword clips      (none yet)
  <word>/               1 s clips of other words, used as the 'other' class (none yet)
```

All files must be 16 kHz, 16-bit mono WAV, as written by `receive_and_play.py -o`.

### `_background_noise_/`
- Name files `board_<place>_<nn>.wav`, e.g. `board_room_01.wav` or `board_kitchen_01.wav`.
- **No speech at all**, not even a distant TV voice: everything in here is labelled background.
- Variety helps more than length: different rooms, appliances, typing, music, fans, traffic.
- The notebook's split cell writes them to `dataset/_background_noise_split_/` as
  `board__<stem>_<n>.wav`. For the train/val/test split, chunks are grouped in 10 s blocks, so
  neighbouring seconds of one recording can't end up in both training and test.

### Word clips (future)
- 1 s clips, word roughly centred and not cut off at either end.
- Name them `<session>_<n>.wav`, e.g. `board03_17.wav`. The part before the first `_` is the split
  group, the same way Speech Commands uses the speaker id, so one session never ends up in both
  train and test.
- Directory names are the label: `forward`, `backward`, `stop`, or any other word, which maps to
  `other`.

## Current contents

| File | Length | Notes |
|---|---|---|
| `_background_noise_/board_room_01.wav` | 226 s | Room ambience, no speech. Level varies from about −54 to −13 dBFS per second; DC offset about −0.06 (board mic, as on target) |
