# Public-dataset evaluation set

Eval clips built from public speech datasets by `EvalScripts/make_public_eval.py`. They're in the same
`<stem>.wav` + `<stem>_truth.csv` format as [`eval_recordings/`](../eval_recordings/README.md), with
hundreds of speakers and hours of non-keyword speech. That gives false accepts per hour enough
events to compare models, which the board recordings alone don't.

Keyword positions come from the datasets' word-level alignments, so nothing is labelled by hand.
The audio isn't the board mic, so treat this set as a complement to `eval_recordings/`, not a
replacement:
- **LibriSpeech** is read audiobooks, with keywords mid-sentence ("went forward to the door"), which
  is harder than a spoken command. Compare its recall between model versions, not against board recall.
- **AMI** is meetings: conversational speech, with interruptions, overlap and fillers. It uses the
  Mix-Headset recording, all participants' headset mics summed into one file.
- **SWC** (Spoken Wikipedia) is about 500 volunteers reading Wikipedia articles on their own home
  mics: the widest range of voices and recording setups.

## Train / val / test split

The speakers are split **50% train / 25% val / 25% test**, by speaker group, so no voice is in two
splits:

| Source | Group |
|---|---|
| LibriSpeech | speaker |
| SWC | reader (Wikipedia user name) |
| AMI | meetings linked by any shared participant |

AMI needs the linking rule because the same four people are in every meeting of a series
(ES2002a–d), and 30 participants are in more than one series. The Idiap IS/IN meetings link 25
series into one group. Groups are assigned greedily, biggest first, so each split gets close to its
share of every keyword and of the utterances or recordings. The rare `backward` is spread the same
way.

| Split | Used for |
|---|---|
| **train** | reserved for training on public data (a later step). Nothing is built from it here. **Only groups with `split = train` in `splits.csv` may ever be trained on.** |
| **val** (`val/`) | comparing models and making decisions: the notebook scores it every run |
| **test** (`test/`) | held back. It's only scored when asked for by name (`--split test`, or `RUN_TEST_EVAL = True` in the notebook). Look at it rarely, e.g. once per finished model, and never tune on it |

`val/` and `test/` each contain the full-length recordings and 1 s clips cut from them (`clips/`).
The clips are for clip-level evaluation and training later:
- **Keyword clips:** one per keyword span of up to 0.7 s, centred on the word. Word timings can be
  off by a few hundred ms, and centring leaves the most room for that on both sides (about 280 ms
  for a typical word). There's no random offset; timing errors already give some variation, and
  time shifts can be added as augmentation when training.
  - **Words near a piece edge:** if the centred window would leave the piece, it's shifted back to
    the edge, but only if that's also the edge of the source audio. The word can't be any further
    out, so nothing is lost. These are about 3% of clips, all LibriSpeech, whose pieces are whole
    files, and shifted by up to 275 ms.
  - **A piece edge inside the source** would need padding instead, so that keyword is skipped and
    `--check` warns. AMI and SWC excerpts keep keywords at least 2 s from their edges, so this
    doesn't happen today. It's the rule to keep when adding a dataset.
  - **Other skips:** a keyword longer than 0.7 s, or a window that overlaps another keyword or
    `ignore` span.
- **`other` clips:** twice as many, as random 1 s windows of speech that overlap no keyword or
  `ignore` span.
- **No background clips:** these sources have no real background audio.
- **The word barely stands out in keyword clips.** It's only ~1–3 dB above the rest of its window,
  because the neighbouring words are in it too. That makes them much harder than Speech Commands'
  isolated words.

## What's committed

| File | |
|---|---|
| `splits.csv` | one row per group: `source`, `group`, `split`, `members` (the speakers / readers / meetings in it), `items`, keyword counts |
| `<split>/ls_<nn>_truth.csv`, `ami_…`, `swc_…` | keyword and `ignore` spans; `file` = source utterance / excerpt |
| `<split>/manifest.csv` | every piece: its position in which WAV, source file and time range, the source's length (`src_duration_s`, which shows whether a piece edge is a source edge), speakers, `gain_db` (static gain), `limit_db` (largest peak reduction by the limiter, 0 if untouched) |
| `<split>/clips.csv` | every clip: file, label, which WAV, piece and time it was cut from, and the word's span |
| `<split>/build_info.json` | each source's build settings and the SHA-256 of every output file, WAVs and clips included |
| `<split>/eval_history.csv` | scores, one row per model version (see below) |

The WAVs are gitignored (~0.1 GB per hour). Rebuild them from `dataset_public/` with the same
command and seed; the output is byte-identical, and the build tells you whether it matched
`build_info.json`.

## 1. Get the data

All raw data goes in `HostSW/dataset_public/` (gitignored):

```
dataset_public/
  LibriSpeech/<split>/<spk>/<chapter>/...                  # openslr.org/12, tar -xf each split here
  librispeech_alignments/<split>/<spk>/<chapter>/*.TextGrid  # zenodo.org/records/2619474
  ami/ami_public_manual_1/                                 # AMI manual annotations
  ami/amicorpus/<meeting>/audio/<meeting>.Mix-Headset.wav  # AMI audio, via download_ami.py
  swc/english/<article>/{aligned.swc,audio.ogg}            # tar -xf swc-2/SWC_English.tar -C swc
```

- **LibriSpeech**: all 7 splits from [openslr.org/12](https://www.openslr.org/12) (~60 GB extracted).
  `backward` is rare, so the train splits matter most:

  | Split | Hours | Speakers | forward | backward | stop |
  |---|---|---|---|---|---|
  | dev + test, all four | 39 | 146 | 36 | 2 | 21 |
  | train-clean-100 | 101 | 251 | 192 | 37 | 122 |
  | train-clean-360 | 367 | 921 | 629 | 118 | 431 |
  | train-other-500 | 525 | 1,166 | 878 | 170 | 605 |

- **LibriSpeech alignments**: `librispeech_alignments.zip` from
  [zenodo.org/records/2619474](https://zenodo.org/records/2619474) (623 MB, Montreal Forced Aligner,
  all splits).
- **AMI annotations**: `ami_public_manual_1` from
  [groups.inf.ed.ac.uk/ami/download](https://groups.inf.ed.ac.uk/ami/download).
- **AMI audio**: on the same page, select the meetings and any recording, and it generates a
  `amiBuild-*.wget.sh` script. Don't run that script itself: with all 16 array mics it's ~163 GB.
  `download_ami.py` takes the meeting list from it and downloads just `Mix-Headset` (171 files,
  ~11 GB), even if the script lists other recordings, with resume, retries and a progress bar.
  From `HostSW/`:
  ```
  python EvalScripts/download_ami.py -s eval_recordings_public/amiBuild-<...>.wget.sh --dry-run
  python EvalScripts/download_ami.py -s eval_recordings_public/amiBuild-<...>.wget.sh
  ```
  If it's interrupted (Ctrl-C, timeout), run the same command again to resume.
  - Why Mix-Headset: the table-top array mic (`Array1-01`, used at first) picks up so much room
    noise and echo that speech is only ~18 dB above the background, so it sounds distant and buried
    even after normalising. Mix-Headset is ~35–43 dB, like LibriSpeech and SWC. `Mix-Lapel` is in
    between. Use `--ami-channel` in `make_public_eval.py` and `--channels` in `download_ami.py` to
    switch.
  - Coverage: 167 meetings, 95 h, 187 participants.
  - Keywords: only 60 `forward`, 14 `backward` and 76 `stop`, so AMI is mostly useful for its
    negatives.
  - Negative excerpts must be at least 30% covered by words, so they contain someone talking rather
    than room noise boosted to speech level.
  - Each meeting gets its own gain, from −9.5 to +28 dB (median +11 dB), because the meetings were
    recorded at levels ~30 dB apart. The quietest meetings are TS3004a/b.
  - One Mix-Headset file (ES2010d) is stereo, with identical channels; it's downmixed like any
    other.
- **SWC**: the swc-2 release from [nats.gitlab.io/swc](https://nats.gitlab.io/swc/).
  - Only `SWC_English.tar` is used (16 GB, 1,339 articles). Extract it into `dataset_public/swc/`.
    The Dutch and German tars aren't needed.
  - Usable: 1,187 articles, 331 h, 507 readers. The rest is skipped:
    - 54 articles whose Ogg file libsndfile can't open.
    - 10 below 16 kHz.
    - Articles split over several audio files, which have no timings.
  - Only about half the words are aligned, so the script is cautious:
    - An untimed keyword becomes an `ignore` span covering the gap where it must be, plus 1 s each
      side. The aligner often leaves no gap at all (e.g. "and stop the" with "and" and "the"
      touching), so without the padding the span would be empty.
    - A keyword that only comes from text normalisation becomes `ignore`. For example, a `/` in a
      URL is read as "forward slash".
    - Negative excerpts must be at least 40% covered by aligned words.

## 2. Build

From `HostSW/`:
```
python EvalScripts/make_public_eval.py --stats      # keyword counts per source and split, writes nothing
python EvalScripts/make_public_eval.py              # split, then build val/ and test/ (a source without data is skipped)
python EvalScripts/make_public_eval.py --sources ami   # re-split and rebuild one source; the others' files are kept
python EvalScripts/make_public_eval.py --check      # check both splits
```

| Option | Default | Effect |
|---|---|---|
| `--max-per-keyword` | no limit | limit on occurrences of a keyword per source and split |
| `--max-per-speaker` | no limit | limit on occurrences of a keyword per speaker |
| `--ls-negative-min`, `--ami-negative-min`, `--swc-negative-min` | 1500, 600, 900 | minutes of pieces with no keyword, 50 h per split in total. The build warns if a source runs out. |
| `--ami-channel` | Mix-Headset | which AMI recording to use |
| `--level-dbfs` | −21 | median active-speech level of each source (the board recordings' level) |
| `--seed` | 0 | changes which pieces are picked and the order they're in |

The settings are only recorded in the command, so if you change them, note it in the snapshot notes.

**Word rules:**

| Words | Treated as |
|---|---|
| `forward(s)` / `backward(s)` / `stop` | keywords |
| any other word containing one (`stopped`, `straightforward`, `forwarded`, a cut-off `stop-`) | `ignore` |
| `Christopher`, `Aristophanes` | negative audio |
| a keyword aligned shorter than 0.2 s (the aligner squeezed it, so its position is unreliable) | `ignore`, widened by 1 s each side |

**Level.** The aim is to match the board on average while keeping natural variation, so that the
set also tests how the model copes with loud and soft speech:
- **Gain:** each piece has its DC removed, then gets one static gain per source. AMI gets one per
  meeting instead, since its meetings were recorded about 30 dB apart and the board's mic gain is
  fixed.
- **Target:** the gain is chosen so the source's median *active-speech level* is `--level-dbfs`.
  Active speech is the loudest 30% of 20 ms frames, so pauses don't count. The default of −21 dBFS
  is the board recordings' active-speech level (−19 to −22.6 dBFS).
- **Spread:** loudness still varies within each source, 5th to 95th percentile:

  | Source | Spread |
  |---|---|
  | LibriSpeech | 9–10 dB |
  | AMI | 12–14 dB |
  | SWC | 18–19 dB |

  Neighbouring pieces are almost always different speakers, separated by a gap, so a level change
  sounds like a new person or someone at a different distance.
- **Limiter:** a look-ahead peak limiter keeps every sample at or below −1 dBFS. It lowers the
  gain only within about 10 ms of a peak, so the speech keeps its level, and pieces without peaks
  are untouched. It acts on 8–11% of pieces, mostly breath pops in the AMI headsets and SWC
  recordings that were made loud. Keywords are rarely affected. The heaviest case is a few loud SWC
  articles in test, such as `Air-tractor_sledge`: up to 22% of a piece's samples are reduced by over
  1 dB, and up to 13 dB at the peaks.
- **Keywords:** they come out around −25 dBFS. That's quieter than on the board (−20), because
  here they're ordinary words in the middle of sentences, not stressed commands.

Excerpts that contain a recording dropout are skipped: a run of exact zeros of 0.5 s or more, or an
RMS below −70 dBFS. ES2012c has 20 s of zeros. The rule uses a run rather than the share of zero
samples because clean recordings like AMI's Mix-Headset have many zero samples in pauses, in runs of
at most ~10 ms.

**Current build** (default settings), per split:

| Split | Source | Files | Minutes | Pieces with keywords (min) | Speakers | forward | backward | stop |
|---|---|---|---|---|---|---|---|---|
| val | LibriSpeech | 338 | 1,733 | 161 | 677 | 400 | 81 | 279 |
| val | AMI | 122 | 627 | 11 | 48 | 15 | 3 | 19 |
| val | SWC | 184 | 946 | 21 | 155 | 25 | 8 | 33 |
| test | LibriSpeech | 338 | 1,732 | 162 | 677 | 403 | 79 | 279 |
| test | AMI | 122 | 627 | 10 | 45 | 15 | 4 | 17 |
| test | SWC | 184 | 945 | 20 | 143 | 23 | 6 | 33 |

Each split is about 55 h. The size is set by the false accept rates it needs to measure: 50 h gives
about 50 events at 1 false accept per hour, or ±14% before allowing for clustering by speaker, and
zero false accepts would show the rate is below about 0.06 per hour. AMI stops at 10 h because its
hours come from only about 12 speaker groups, and its 25 h per split can't be filled by random 20 s
excerpts. All three sources are nearly continuous speech, so these rates say little about music, TV
or a quiet room.

Every keyword of a split's speakers is used, so the classes aren't balanced: `forward` is about half
of the keywords and LibriSpeech about 88%. The pooled recall leans the same way, so also look at the
per-keyword and per-source columns. The only keywords left out are those with no usable alignment,
such as LibriSpeech utterances where the alignment doesn't match the transcript (about 10 per split).

Clips:
- **val:** 408 `forward`, 70 `backward`, 318 `stop`, 1,592 `other`.
- **test:** 413 `forward`, 65 `backward`, 315 `stop`, 1,586 `other`.

The number of clips is tied to the keywords, not the length of the recordings, so `other` stays at
exactly twice the keyword clips however many hours of negatives are built.

## 3. Check

Every build checks each split's output, and `python EvalScripts/make_public_eval.py --check` checks
an existing build.

**Errors** (exit code 1):
- A WAV, truth file or manifest row is missing.
- A WAV isn't 16 kHz mono.
- A file doesn't match the hash in `build_info.json`, e.g. a stale WAV or a hand-edited truth file.
- A piece lies outside its WAV, or two pieces overlap.
- A peak is above −1 dBFS, which would mean the limiter failed.
- A span lies outside the piece it came from.
- A label is unknown.
- A speaker, reader or meeting in a split's manifest isn't in that split in `splits.csv`.
- A clip isn't 1 s of 16 kHz mono, or is outside its piece.
- A keyword clip doesn't contain its whole word, or is neither centred on it nor at a piece edge
  that's also a source edge.
- An `other` clip overlaps a keyword or `ignore` span.

**Warnings**, which are listed but don't fail the build:
- A keyword span shorter than 0.2 s or longer than 2 s.
- A keyword span more than 10 dB quieter than the 1.5 s on each side, within its piece. It may be
  misaligned onto a pause.
- A piece with more than +35 dB of gain, probably near-silence boosted to speech level. The
  quietest AMI meetings need about +28 dB.
- A source's median speech level is more than 1 dB from the target it was built with.
- A keyword is too close to a piece edge inside the source, so the piece needs more padding.

It also prints each source's speech-level spread, and how many pieces the limiter touched and by
how much.

The current build has 6 warnings, all in test. They're keyword spans that are quiet compared with the
audio around them; the ones listened to in earlier builds turned out to be real words.

A build also reports, per source, whether it came out identical to the previous one. If it changed
with the same settings, it warns that either the datasets changed or the build isn't reproducible.

**By ear:**
- To listen to the labels:
  `python EvalScripts/audacity_labels_to_truth.py --reverse -i eval_recordings_public/val/ls_00_truth.csv`,
  then open `val/ls_00.wav` in Audacity and import `val/ls_00_labels.txt`.
- Spot-check a few keyword and `ignore` spans per source.
- Delete the `_labels.txt` afterwards, since `snapshot_eval.py` would convert it over the truth file.

## 4. Score a model

`snapshot_eval.py --dataset public` scores a split. The notebook runs it on val for every model;
test only on request. The ~55 h of a split take a few minutes with `--jobs 6`.
```
python EvalScripts/snapshot_eval.py -v v10 -m tflite_models/kws_v10_int8_softmax.tflite --dataset public --jobs 6
python EvalScripts/snapshot_eval.py -v v10 -m tflite_models/kws_v10_int8_softmax.tflite --dataset public --split test --jobs 6
```

**`<split>/eval_history.csv`** has one row per model version; re-scoring a version replaces its row.
Each row has:
- **Pooled results:** recall (overall and per keyword) and false accepts per hour, from the same
  scoring as the board set.
- **Per-source results:** `recall_ls`, `fa_per_hour_ami`, `keyword_spans_swc`, `negative_min_…`
  and so on.
- **Recall at a false accept budget:** `recall_at_100fa` and so on, described below.
- **MLflow link:** `mlflow_run_id` and `run_type`.
  - `training`: the row came from the notebook, and the id is that training run.
  - `standalone`: the row came from running the script yourself. It made its own MLflow eval run,
    nested under the version's training run.

Every run, including replaced ones, stays in MLflow, with the per-recording results as
`reports/public_<split>_per_recording.csv`. Each run also gets the split's whole history as it
stood after that run, as sortable, filterable tables in the run's **Artifacts** tab, in
`reports/public_<split>_history/`:

| Table | Columns |
|---|---|
| `1_default_threshold` | recall (overall and per keyword), false accepts and counts at the fixed threshold |
| `2_default_threshold_by_source` | the same per source |
| `3_budget_300fa`, `4_budget_100fa`, `5_budget_30fa` | recall, false accepts per hour, threshold and events at that budget, per keyword too. Only rows scored since budgets existed |
| `6_run_settings` | model path and hash, detection settings, notes, MLflow run id |

Each starts with the version and timestamp. The latest run has the latest history. MLflow shows one
table at a time.

### Recall at a false accept budget

This is the main way to compare models. Two models at threshold 0.7 can sit at very different points
of their trade-off, so instead each is held to the same false accept rate, as KWS papers do with
recall at a fixed false alarm rate.

The budgets are 300, 100 and 30 false accepts per hour **for the whole device**, all keywords
together (`FA_BUDGETS` in `snapshot_eval.py`). All keywords share one threshold, as the device runs
today:

1. A threshold sweep gives each keyword's recall and false accepts at each threshold.
2. A budget's threshold is the one with the highest mean recall whose false accepts, all keywords
   together, fit in the budget.

| Column | Meaning |
|---|---|
| `recall_at_<b>fa` | mean of the three keywords' recalls, so `forward` being half the keywords doesn't dominate it. Empty if no threshold reaches the budget |
| `fa_per_hour_at_<b>fa` | the false accepts per hour at that threshold |
| `threshold_at_<b>fa` | the threshold |
| `fa_events_at_<b>fa` | how many false accepts the rate rests on. The script marks fewer than 10 as too few to rely on |
| `recall_at_<b>fa_<kw>`, `fa_events_at_<b>fa_<kw>` | each keyword's recall and false accepts at the same threshold: which keywords miss, and which the false accepts come from |

- **Why one threshold:** it's simple and it's what the firmware will run. A model that's less
  confident on one keyword is penalised for it: v10 gets only 16% `forward` at 100 per hour,
  because a threshold strict enough for `backward` and `stop` silences `forward`. Per-keyword
  thresholds, as in the multi-keyword papers, would hide that. Improving the model should fix it.
- **Test:** the threshold is chosen on val only. A test split is scored at the thresholds val chose
  for the same model and settings, so score val first. Its false accept rate can land above the
  budget, which is the honest result.
- **Curve:** a val run adds `reports/public_val_curve.csv` and `public_val_curve.png` to the MLflow
  run: each keyword's miss rate, and their mean, against the total false accepts per hour, with
  the budgets marked.
- **Changing the budgets:** add lower ones as models improve rather than replacing these. Each
  budget has history columns, and a history with columns the script doesn't know is refused.
- **Limits:** 55 h of negatives resolves a budget down to about 1 per hour, about 55 events. These
  rates are per hour of near-continuous speech, so they don't compare directly with published
  figures.
