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

**Never train on the speakers in `speakers.txt`.** For AMI that means whole meeting series (ES2002
covers ES2002a–d), since the same people are in each meeting. For SWC it means the readers'
Wikipedia user names.

## What's committed

| File | |
|---|---|
| `ls_<nn>_truth.csv`, `ami_<nn>_truth.csv`, `swc_<nn>_truth.csv` | keyword and `ignore` spans; `file` = source utterance / excerpt |
| `manifest.csv` | every piece: its position in which WAV, source file and time range, speakers, `gain_db` (static gain), `limit_db` (largest peak reduction by the limiter, 0 if untouched) |
| `speakers.txt` | speakers (and AMI meeting series) to keep out of training |
| `build_info.json` | each source's build settings and the SHA-256 of every output file, WAVs included |

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
python EvalScripts/make_public_eval.py --stats      # keyword counts per split / source, writes nothing
python EvalScripts/make_public_eval.py              # build ls_*, ami_* and swc_* (a source without data is skipped)
python EvalScripts/make_public_eval.py --sources ami   # rebuild one source; the other's files are kept
```

| Option | Default | Effect |
|---|---|---|
| `--max-per-keyword` | 150 | keyword occurrences per keyword per source |
| `--max-per-speaker` | 2 | occurrences of one keyword per speaker |
| `--ls-negative-min`, `--ami-negative-min`, `--swc-negative-min` | 120, 60, 60 | minutes of pieces with no keyword |
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
  | LibriSpeech | 9 dB |
  | AMI | 12 dB |
  | SWC | 20 dB |

  Neighbouring pieces are almost always different speakers, separated by a gap, so a level change
  sounds like a new person or someone at a different distance.
- **Limiter:** a look-ahead peak limiter keeps every sample at or below −1 dBFS. It lowers the
  gain only within about 10 ms of a peak, so the speech keeps its level, and pieces without peaks
  are untouched. It acts on about 6% of pieces, mostly breath pops in the AMI headsets and SWC
  recordings that were made loud.
- **Keywords:** they come out around −25 dBFS. That's quieter than on the board (−20), because
  here they're ordinary words in the middle of sentences, not stressed commands.

Excerpts that contain a recording dropout are skipped: a run of exact zeros of 0.5 s or more, or an
RMS below −70 dBFS. ES2012c has 20 s of zeros. The rule uses a run rather than the share of zero
samples because clean recordings like AMI's Mix-Headset have many zero samples in pauses, in runs of
at most ~10 ms.

**Current build** (default settings):

| Source | Files | Minutes | Pieces with keywords (min) | Speakers | forward | backward | stop |
|---|---|---|---|---|---|---|---|
| LibriSpeech | 43 | 219 | 91 | 871 | 150 | 153 | 151 |
| AMI | 20 | 102 | 40 | 176 | 53 | 14 | 70 |
| SWC | 24 | 122 | 59 | 235 | 66 | 25 | 90 |
| **Total** | 87 | 443 | 190 | ~1,280 | 269 | 192 | 311 |

## 3. Check

Every build checks its own output, and `python EvalScripts/make_public_eval.py --check` checks an
existing build (about 2 s).

**Errors** (exit code 1):
- A WAV, truth file or manifest row is missing.
- A WAV isn't 16 kHz mono.
- A file doesn't match the hash in `build_info.json`, e.g. a stale WAV or a hand-edited truth file.
- A piece lies outside its WAV, or two pieces overlap.
- A peak is above −1 dBFS, which would mean the limiter failed.
- A span lies outside the piece it came from.
- A label is unknown.

**Warnings**, which are listed but don't fail the build:
- A keyword span shorter than 0.2 s or longer than 2 s.
- A keyword span more than 10 dB quieter than the 1.5 s on each side. It may be misaligned onto a pause.
- A piece with more than +35 dB of gain, probably near-silence boosted to speech level. The
  quietest AMI meetings need about +28 dB.
- A source's median speech level is more than 1 dB from the target it was built with.

It also prints each source's speech-level spread, and how many pieces the limiter touched and by
how much.

The current build has 7 warnings, all quiet keyword spans that are real words: one LibriSpeech
`backward` (`ls_06.wav`), and 6 AMI keywords said softly in conversation, often while someone else
is louder.

A build also reports, per source, whether it came out identical to the previous one. If it changed
with the same settings, it warns that either the datasets changed or the build isn't reproducible.

**By ear:**
- To listen to the labels:
  `python EvalScripts/audacity_labels_to_truth.py --reverse -i eval_recordings_public/ls_00_truth.csv`,
  then open `ls_00.wav` in Audacity and import `ls_00_labels.txt`.
- Spot-check a few keyword and `ignore` spans per source.
- Scoring works the same way as for the board recordings:
  `snapshot_eval.py --eval-dir eval_recordings_public`. It writes its own `eval_history.csv` here,
  so the numbers don't pool with the board set.
