# Public-dataset evaluation set

Eval recordings built from public speech datasets, plus speech-free background noise, by
`EvalScripts/make_public_eval.py`. They're in the same `<stem>.wav` + `<stem>_truth.csv` format as
[`eval_recordings/`](../eval_recordings/README.md), with hundreds of speakers and hours of
non-keyword speech and noise. That gives false accepts per hour enough events to compare models,
which the board recordings alone don't, and shows separately how often the model fires on speech
and on everything else.

Keyword positions come from the datasets' word-level alignments, so nothing is labelled by hand.
The audio isn't the board mic, so treat this set as a complement to `eval_recordings/`, not a
replacement:
- **LibriSpeech** is read audiobooks, with keywords mid-sentence ("went forward to the door"), which
  is harder than a spoken command. Compare its recall between model versions, not against board recall.
- **AMI** is meetings: conversational speech, with interruptions, overlap and fillers. It uses the
  Mix-Headset recording, all participants' headset mics summed into one file.
- **SWC** (Spoken Wikipedia) is about 500 volunteers reading Wikipedia articles on their own home
  mics: the widest range of voices and recording setups.
- **Background noise** (see [Background noise](#background-noise)): music without vocals and
  assorted noise from **MUSAN**, environments from **DEMAND**, and household audio from **Sounds
  of Home**. Its truth files are empty: anything that fires in it is a false accept.

## Train / val / test split

The speakers are split **50% train / 25% val / 25% test**, by speaker group, so no voice is in two
splits:

| Source | Group |
|---|---|
| LibriSpeech | speaker |
| SWC | reader (Wikipedia user name) |
| AMI | meetings linked by any shared participant |
| MUSAN music | artist (a performer's tracks sound alike) |
| MUSAN noise | Sound Bible: who recorded it; Free Sound: the file (no names given) |
| DEMAND | environment |
| Sounds of Home | home (its living room and kitchen mics record at the same time) |

AMI needs the linking rule because the same four people are in every meeting of a series
(ES2002a–d), and 30 participants are in more than one series. The Idiap IS/IN meetings link 25
series into one group. Groups are assigned greedily, biggest first, so each split gets close to its
share of every keyword and of the utterances or recordings. The rare `backward` is spread the same
way. Noise groups are balanced on usable (speech-free) hours, per MUSAN folder so each split gets
every genre, and on group count. A noise group keeps the split it has in `splits.csv`, so
downloading more audio never moves one.

| Split | Used for |
|---|---|
| **train** (`train/`) | training: its 1 s clips go into the notebook's train split. Built like the others (its WAVs make the clips reproducible), but **never scored**: `snapshot_eval.py` only accepts val and test. **Only groups with `split = train` in `splits.csv` may ever be trained on.** |
| **val** (`val/`) | comparing models and making decisions: the notebook scores it every run |
| **test** (`test/`) | held back. It's only scored when asked for by name (`--split test`, or `RUN_TEST_EVAL = True` in the notebook). Look at it rarely, e.g. once per finished model, and never tune on it |

`val/`, `test/` and `train/` each contain the full-length recordings and 1 s clips cut from them (`clips/`).
The notebook adds the clips to its splits of the same name: train clips to training, val clips to its
per-epoch validation and per-run metrics, and test clips to the held-back test (only used with
`RUN_TEST_EVAL = True`):
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
- **`_background_noise_` clips** (the model's background class): twice as many as keyword clips, as
  random 1 s windows of the noise pieces, picked in proportion to their length.
- **The word barely stands out in keyword clips.** It's only ~1–3 dB above the rest of its window,
  because the neighbouring words are in it too. That makes them much harder than Speech Commands'
  isolated words.
- **Checked by ear** (Nov 2026, `review_keyword_clips.py --public`, val only): of 138 clips (32 per source at
  random, plus the shifted, longest, shortest and quietest words), 1 didn't contain the whole keyword: a
  LibriSpeech "forward" aligned to 0.23 s. It's still in the val clips.

## Background noise

About as many hours of noise as speech, so false accepts can be read on each (`fa_per_hour_speech`
and `fa_per_hour_noise`). It comes from `dataset_noise/` (`download_noise.py`). Only speech-free
audio is used: speech there would be unlabelled `other`, and could even contain a keyword.
`noise_speech.py` finds the speech:
- **Labels decide where a dataset has them:** Sounds of Home's own sound-event labels (PANNs,
  frames of 0.21 s, made before its authors muted the speech they found), and MUSAN's vocals
  annotation.
- **YAMNet decides where it doesn't:** MUSAN noise at a score of 0.2 (cautious: real speech
  scores above 0.3 in ~95% of its 0.96 s windows), DEMAND at 0.5 (confident speech only). MUSAN
  music also cuts out what YAMNet is confident (≥ 0.5) is speech, despite the vocals label.
- **Checked by ear** (a spot-check file, Nov 2026): YAMNet's MUSAN detections were speech. Its
  extra detections in Sounds of Home (up to 1.0, where the labels had none) and DEMAND's quieter
  environments (0.2–0.8) were at most faint, unintelligible speech, fine as background. So Sounds
  of Home goes by its labels alone, and DEMAND counts only confident speech.
- Every speech window or frame gets 2 s of margin each side.
- MUSAN files without vocals give every speech-free stretch of 5 s or more, cut evenly into pieces
  of at most 5 min, so a long track is several pieces and short sound effects are used too.
  Sounds of Home hours and DEMAND recordings are cut into 60 s excerpts between their speech.
  Every MUSAN piece and DEMAND excerpt of a split goes in, then Sounds of Home fills up to 35 h:
  in val and test that's about 6 h MUSAN, 10 min DEMAND and 29 h Sounds of Home.

| Type | Why | What the labels say | What YAMNet says | Downloaded | Usable, all splits | val / test |
|---|---|---|---|---|---|---|
| **MUSAN music**, instrumental: Free Music Archive (incl. western art), HD Classical, Jamendo, Kevin MacLeod | music and TV-like sound, a common loud background | vocals yes 226 files (14.2 h), unknown 15 (0.9 h), no 419 (27.6 h): only "no" is used | 19 of the 419 instrumental files have a confident speech moment (spoken samples?): cut out with the margin | 660 files, 42.6 h | 419 files, 27.5 h | 5.0 / 4.8 h |
| **MUSAN noise**: Free Sound, Sound Bible | everyday and technical sounds: vehicles, water, animals, phones, tools, bells, footsteps | no speech labels; MUSAN says "no intelligible speech" | speech in 202 of the 843 Free Sound files (crowds, chatter) and 10 of Sound Bible's 87: cut out with the margin | 930 files, 6.2 h (714 under 30 s) | 625 files, 5.0 h | 1.0 / 1.3 h |
| **DEMAND** | real environments, recorded at their natural level: home, office, nature, street, transport | none; the 5 environments with conversation weren't downloaded | confident speech in the bus (most of it), metro, station, traffic and park; 8 environments have none | 13 environments, 1.1 h | 0.8 h of 60 s excerpts from 11 environments (not bus, metro) | val 9 min (field, traffic); test 12 min (kitchen, river, station) |
| **Sounds of Home** | real homes, kitchen and living room mics, over a week: dishes, taps, doors, cooking, appliances, pets, footsteps, music | speech or TV in 95 of the 336 h (muted in the audio); 187 h label-free with the margin | not used (its extra detections were faint, unintelligible speech) | 336 h: 36 h per mic in the val / test homes (2, 3, 4, 7), 8 h per mic in the train homes (1, 5, 6) | 132 h in 60 s excerpts | 28.9 / 28.7 h (val: homes 2, 4; test: 3, 7) |

Sounds of Home fills each split up to the 35 h `--noise-min` asks for, after MUSAN and DEMAND. Its
val / test homes have 36 h per mic downloaded, the train homes 8 h, from:
```
python EvalScripts/download_noise.py --sources soh --soh-mics 03 04 05 06 07 08 13 14 --soh-hours-per-mic 32
python EvalScripts/noise_speech.py
python EvalScripts/make_public_eval.py --sources soh
```
(32 extra hours per mic, ~32 GB to download, ~77 GB on disk as 48 kHz WAVs.) Val uses 29 of its
homes' 55 usable hours and test 29 of 46, so `--noise-min` could go up a little. Training
on Sounds of Home would want more hours of the train homes (mics 01, 02, 09–12). Each home's two
mics record at the same time, so they're one group: they can hear the same sound.

## What's committed

| File | |
|---|---|
| `splits.csv` | one row per group: `source`, `group`, `split`, `members` (the speakers / readers / meetings in it; a noise group's own name), `items`, keyword counts. A noise file's group is in `dataset_noise/manifest.csv` |
| `<split>/ls_<nn>_truth.csv`, `ami_…`, `swc_…` | keyword and `ignore` spans; `file` = source utterance / excerpt |
| `<split>/musan_<nn>_truth.csv`, `demand_…`, `soh_…` | empty (header only): noise |
| `<split>/manifest.csv` | every piece: its position in which WAV, source file (relative to `dataset_public/` or `dataset_noise/`) and time range, the source's length (`src_duration_s`, which shows whether a piece edge is a source edge), speakers, noise `category`, `gain_db` (static gain; 0 for noise), `limit_db` (largest peak reduction by the limiter, 0 if untouched) |
| `<split>/clips.csv` | every clip: file, label, which WAV, piece and time it was cut from, and the word's span |
| `<split>/build_info.json` | each source's build settings and the SHA-256 of every output file, WAVs and clips included |
| `<split>/eval_history.csv` | scores, one row per model version (see below) |

The WAVs are gitignored (~0.1 GB per hour). Rebuild them from `dataset_public/` and
`dataset_noise/` with the same command and seed; the output is byte-identical, and the build tells
you whether it matched `build_info.json`. Noise also depends on `dataset_noise/speech.npz`
(`noise_speech.py`), whose hash is in `build_info.json`.

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
- **Background noise** goes in `HostSW/dataset_noise/` (gitignored), from `HostSW/`:
  ```
  python EvalScripts/download_noise.py      # MUSAN, DEMAND, Sounds of Home (resumable; --dry-run for sizes)
  python EvalScripts/noise_speech.py        # finds the speech in it -> dataset_noise/speech.npz (~15 min)
  ```
  Run `noise_speech.py` again after downloading more: it only analyses new files.

## 2. Build

From `HostSW/`:
```
python EvalScripts/make_public_eval.py --stats      # keyword counts per source and split, writes nothing
python EvalScripts/make_public_eval.py              # split, then build val/, test/ and train/ (a source without data is skipped)
python EvalScripts/make_public_eval.py --splits train   # build one split; the others' files are kept
python EvalScripts/make_public_eval.py --clips-only     # re-cut the clips from the WAVs as built, after changing the clip settings
python EvalScripts/make_public_eval.py --sources ami   # re-split and rebuild one source; the others' files are kept
python EvalScripts/make_public_eval.py --sources soh   # e.g. after downloading more Sounds of Home hours
python EvalScripts/make_public_eval.py --check      # check every built split
```

| Option | Default | Effect |
|---|---|---|
| `--max-per-keyword` | no limit | limit on occurrences of a keyword per source and split |
| `--max-per-speaker` | no limit | limit on occurrences of a keyword per speaker |
| `--ls-negative-min`, `--ami-negative-min`, `--swc-negative-min` | 950, 380, 570 | minutes of speech pieces with no keyword, ~32 h per split in total, ~35 h with the keyword pieces. The build warns if a source runs out. |
| `--noise-min` | 2100 | minutes of background noise per split (35 h, as much as the speech): every MUSAN piece and DEMAND excerpt of the split, then Sounds of Home excerpts up to this. The build warns if Sounds of Home runs out. |
| `--ami-channel` | Mix-Headset | which AMI recording to use |
| `--level-dbfs` | −21 | median active-speech level of each speech source (the board recordings' level) |
| `--noise-dir` | `dataset_noise/` | where `download_noise.py` and `noise_speech.py` put the noise |
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
  are untouched. It acts on about 1% of LibriSpeech pieces, 35–37% of AMI's (breath pops in the
  headsets, raised with the quiet meetings) and 5–23% of SWC's (recordings that were made loud), and
  on almost every MUSAN music piece, by about 1 dB (mastered to full scale). Keywords are rarely
  affected. The heaviest cases are a few loud SWC articles in test, such as `Air-tractor_sledge`:
  up to 22% of a piece's samples are reduced by over 1 dB, and up to 15 dB at the peaks.
- **Keywords:** they come out around −25 dBFS. That's quieter than on the board (−20), because
  here they're ordinary words in the middle of sentences, not stressed commands.
- **Noise keeps its recorded level** (DC removed, no gain; the limiter only touches peaks above
  −1 dBFS). The datasets already range widely: Sounds of Home rooms around −55 dBFS with events up
  to about −20, DEMAND from −56 (living room) to −22 dBFS (car), MUSAN noise −35 to −12. MUSAN music is
  at its mastering level, −18 to −9 dBFS (median −14), 3–12 dB above the speech: it stands for
  loud music in the room, and there's no quiet-music case.

Excerpts that contain a recording dropout are skipped: a run of exact zeros of 0.5 s or more, or an
RMS below −70 dBFS. ES2012c has 20 s of zeros. The rule uses a run rather than the share of zero
samples because clean recordings like AMI's Mix-Headset have many zero samples in pauses, in runs of
at most ~10 ms. Sounds of Home excerpts are also skipped for 0.5 s below −90 dBFS: that's how its
speech removal looks, far below any room's noise floor (about −60 dBFS).

AMI / SWC negative excerpts and Sounds of Home excerpts are picked the same way: each recording's
stretches between keywords / `ignore` words (or speech) are cut into excerpts end to end from a
random offset, and taken round-robin over meetings / readers / hours until there are enough.

**Current build** (default settings), per split:

| Split | Source | Files | Minutes | Pieces with keywords (min) | Speakers / groups | forward | backward | stop |
|---|---|---|---|---|---|---|---|---|
| val | LibriSpeech | 226 | 1,159 | 161 | 677 | 400 | 81 | 279 |
| val | AMI | 78 | 401 | 11 | 48 | 15 | 3 | 19 |
| val | SWC | 118 | 607 | 21 | 154 | 25 | 8 | 33 |
| val | MUSAN | 53 | 360 | | 155 | | | |
| val | DEMAND | 2 | 9 | | 2 | | | |
| val | Sounds of Home | 347 | 1,752 | | 2 | | | |
| test | LibriSpeech | 226 | 1,159 | 162 | 677 | 403 | 79 | 279 |
| test | AMI | 78 | 401 | 10 | 45 | 15 | 4 | 17 |
| test | SWC | 118 | 606 | 20 | 142 | 23 | 6 | 33 |
| test | MUSAN | 57 | 372 | | 176 | | | |
| test | DEMAND | 2 | 12 | | 3 | | | |
| test | Sounds of Home | 344 | 1,737 | | 2 | | | |
| train | LibriSpeech | 263 | 1,352 | 347 | 1,130 | 914 | 163 | 599 |
| train | AMI | 80 | 415 | 24 | 94 | 30 | 7 | 40 |
| train | SWC | 122 | 629 | 43 | 148 | 50 | 15 | 69 |
| train | MUSAN | 190 | 1,231 | | 314 | | | |
| train | DEMAND | 6 | 28 | | 6 | | | |
| train | Sounds of Home | 170 | 858 | | 3 | | | |

Val and test each have 36.1 h of speech and 35.3 h of noise (see [Background noise](#background-noise));
train has 39.9 h and 35.3 h (more keywords, so more keyword pieces). Train gets more MUSAN music (all of
MUSAN's Kevin MacLeod folder is in train) and less Sounds of Home than val and test: the train homes have only
8 h per mic downloaded.
The size is set by the false accept rates it needs to measure: 35 h gives about 35 events at 1
false accept per hour, or ±17% before allowing for clustering by speaker, and zero false accepts
would show the rate is below about 0.09 per hour. AMI stops at about 6.5 h because its hours come
from only about 12 speaker groups.

Every keyword of a split's speakers is used, so the classes aren't balanced: `forward` is about half
of the keywords and LibriSpeech about 88%. The pooled recall leans the same way, so also look at the
per-keyword and per-source columns. The only keywords left out are those with no usable alignment,
such as LibriSpeech utterances where the alignment doesn't match the transcript (about 10 per split).

Clips:
- **val:** 408 `forward`, 70 `backward`, 318 `stop`, 1,592 `other`, 1,592 `_background_noise_`.
- **test:** 413 `forward`, 65 `backward`, 315 `stop`, 1,586 `other`, 1,586 `_background_noise_`.
- **train:** 925 `forward`, 135 `backward`, 670 `stop`, 3,460 `other`, 3,460 `_background_noise_`.

The number of clips is tied to the keywords, not the length of the recordings, so `other` and
`_background_noise_` stay at exactly twice the keyword clips, however many hours are built.

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
- An `other` clip overlaps a keyword or `ignore` span, or comes from noise; a `_background_noise_`
  clip isn't in a noise piece; a clip has an unknown label.
- A noise file has a truth span, or a noise piece overlaps speech found by `noise_speech.py`
  (with its margin).

**Warnings**, which are listed but don't fail the build:
- A keyword span shorter than 0.2 s or longer than 2 s.
- A keyword span more than 10 dB quieter than the 1.5 s on each side, within its piece. It may be
  misaligned onto a pause.
- A piece with more than +35 dB of gain, probably near-silence boosted to speech level. The
  quietest AMI meetings need about +28 dB.
- A source's median speech level is more than 1 dB from the target it was built with.
- A keyword is too close to a piece edge inside the source, so the piece needs more padding.
- A Sounds of Home piece has a muted stretch (0.5 s below −90 dBFS): speech removal the labels
  missed.
- A split has less noise than `--noise-min`.

It also prints each source's level spread, and how many pieces the limiter touched and by
how much.

The current build has 6 keyword warnings in test and 6 in train, none in val. They're keyword spans that are
quiet compared with the audio around them; the ones listened to in earlier builds turned out to be real words.

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
test only on request. A split (~71 h today) takes about 4 minutes with `--jobs 6`.
```
python EvalScripts/snapshot_eval.py -v v10 -m tflite_models/kws_v10_int8_softmax.tflite --dataset public --jobs 6
python EvalScripts/snapshot_eval.py -v v10 -m tflite_models/kws_v10_int8_softmax.tflite --dataset public --split test --jobs 6
```

**`<split>/eval_history.csv`** has one row per model version; re-scoring a version replaces its row.
Each row has:
- **Pooled results:** recall (overall and per keyword) and false accepts per hour, from the same
  scoring as the board set.
- **Speech and noise apart:** `fa_per_hour_speech` / `fa_per_hour_noise`, with `false_accepts_…`
  and `negative_min_…`. Speech is LibriSpeech, AMI and SWC; noise is MUSAN, DEMAND and Sounds of
  Home. Each rate is over its own hours, so the overall rate lies between them.
- **Per-source results:** `recall_ls`, `fa_per_hour_ami`, `keyword_spans_swc`, `negative_min_…`
  and so on; noise sources have `fa_per_hour_…` and `negative_min_…` only.
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
| `1_default_threshold` | recall (overall and per keyword), false accepts per hour (overall, speech, noise) and counts at the fixed threshold |
| `2_default_threshold_by_source` | the same per source |
| `3_budget_300fa`, `4_budget_100fa`, `5_budget_30fa` | recall, false accepts per hour (overall, speech, noise), threshold and events at that budget, per keyword too. Only rows scored since budgets existed |
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
| `fa_per_hour_at_<b>fa_speech`, `_noise`, `fa_events_at_<b>fa_speech`, `_noise` | the same threshold's false accepts on speech only and on noise only, each per its own hours |

- **Why one threshold:** it's simple and it's what the firmware will run. A model that's less
  confident on one keyword is penalised for it: v10 gets only 22% `forward` at 100 per hour,
  because a threshold strict enough for `backward` and `stop` silences `forward`. Per-keyword
  thresholds, as in the multi-keyword papers, would hide that. Improving the model should fix it.
- **Test:** the threshold is chosen on val only. A test split is scored at the thresholds val chose
  for the same model and settings, so score val first. Its false accept rate can land above the
  budget, which is the honest result.
- **Budget on everything together:** the threshold is chosen on all negative hours, speech and
  noise, as the device hears them. With half the hours noise, which fires less, a budget is easier
  to meet than on speech alone, so compare these numbers only with rows scored on the same set.
- **Curve:** a val run adds `reports/public_val_curve.csv` and `public_val_curve.png` to the MLflow
  run: each keyword's miss rate, and their mean, against the total false accepts per hour, with
  the budgets marked. The CSV also has `total_fa_per_hour_speech` and `_noise` per threshold.
- **Changing the budgets:** add lower ones as models improve rather than replacing these. Each
  budget has history columns, and a history with columns the script doesn't know is refused.
- **Limits:** ~70 h of negatives resolves a budget down to about 1 per hour. Published figures are
  usually per hour of mixed everyday audio; the speech rate here is per hour of near-continuous
  speech, so it's harsher, and the noise rate depends on which noise. Neither compares directly.
