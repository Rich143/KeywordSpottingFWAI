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
- **AMI** is meetings recorded with a far-field table mic: conversational speech, closer to how the
  board hears people.
- **SWC** (Spoken Wikipedia) is about 500 volunteers reading Wikipedia articles on their own home
  mics: the widest range of voices and recording setups.

**Never train on the speakers in `speakers.txt`.** For AMI that means whole meeting series (ES2002
covers ES2002a–d), since the same people are in each meeting. For SWC it means the readers'
Wikipedia user names.

## What's committed

| File | |
|---|---|
| `ls_<nn>_truth.csv`, `ami_<nn>_truth.csv`, `swc_<nn>_truth.csv` | keyword and `ignore` spans; `file` = source utterance / excerpt |
| `manifest.csv` | every piece: its position in which WAV, source file and time range, speakers, gain |
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
  ami/amicorpus/<meeting>/audio/<meeting>.Array1-01.wav    # AMI audio, via download_ami.py
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
- **AMI audio**: on the same page, select the meetings and the array mics, and it generates a
  `amiBuild-*.wget.sh` script. Don't run that script itself: with all 16 mics it's ~163 GB.
  `download_ami.py` takes just `Array1-01` (169 files, ~11 GB), with resume, retries and a
  progress bar. From `HostSW/`:
  ```
  python EvalScripts/download_ami.py -s eval_recordings_public/amiBuild-<...>.wget.sh --dry-run
  python EvalScripts/download_ami.py -s eval_recordings_public/amiBuild-<...>.wget.sh
  ```
  If it's interrupted (Ctrl-C, timeout), run the same command again to resume.
  - Coverage: 169 meetings, 94 h, 187 participants.
  - Keywords: only 60 `forward`, 14 `backward` and 76 `stop`, so AMI is mostly useful for its
    negatives.
  - Negative excerpts must be at least 30% covered by words, so they contain someone talking rather
    than room noise boosted to speech level.
  - AMI's far-field mic is quiet, so pieces typically get about +21 dB of gain.
- **SWC**: the swc-2 release from [nats.gitlab.io/swc](https://nats.gitlab.io/swc/).
  - Only `SWC_English.tar` is used (16 GB, 1,339 articles). Extract it into `dataset_public/swc/`.
    The Dutch and German tars aren't needed.
  - Usable: 1,187 articles, 331 h, 507 readers. The rest is skipped:
    - 54 articles whose Ogg file libsndfile can't open.
    - 10 below 16 kHz.
    - Articles split over several audio files, which have no timings.
  - Only about half the words are aligned, so the script is cautious:
    - An untimed keyword becomes an `ignore` span covering the gap where it must be.
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
| `--level-dbfs` | −26 | RMS level each piece is normalised to |
| `--seed` | 0 | changes which pieces are picked and the order they're in |

The settings are only recorded in the command, so if you change them, note it in the snapshot notes.

**Word rules:**

| Words | Treated as |
|---|---|
| `forward(s)` / `backward(s)` / `stop` | keywords |
| any other word containing one (`stopped`, `straightforward`, `forwarded`, a cut-off `stop-`) | `ignore` |
| `Christopher`, `Aristophanes` | negative audio |
| a keyword aligned shorter than 0.2 s (the aligner squeezed it, so its position is unreliable) | `ignore`, widened by 1 s each side |

Excerpts that are recording dropouts (more than 5% exact zeros, or quieter than −70 dBFS) are
skipped. ES2012c has one.

**Current build** (default settings):

| Source | Files | Minutes | Pieces with keywords (min) | Speakers | forward | backward | stop |
|---|---|---|---|---|---|---|---|
| LibriSpeech | 43 | 219 | 91 | 871 | 150 | 153 | 151 |
| AMI | 20 | 102 | 40 | 176 | 53 | 14 | 70 |
| SWC | 24 | 122 | 59 | 234 | 68 | 24 | 89 |
| **Total** | 87 | 443 | 190 | ~1,280 | 271 | 191 | 310 |

## 3. Check

Every build checks its own output, and `python EvalScripts/make_public_eval.py --check` checks an
existing build (about 2 s).

**Errors** (exit code 1):
- A WAV, truth file or manifest row is missing.
- A WAV isn't 16 kHz mono.
- A file doesn't match the hash in `build_info.json`, e.g. a stale WAV or a hand-edited truth file.
- A piece lies outside its WAV, or two pieces overlap.
- A span lies outside the piece it came from.
- A label is unknown.

**Warnings**, which are listed but don't fail the build:
- A keyword span shorter than 0.2 s or longer than 2 s.
- A keyword span more than 10 dB quieter than the 1.5 s on each side. It may be misaligned onto a pause.
- A piece with more than +30 dB of gain, probably near-silence boosted to speech level.

The current build has one warning: a LibriSpeech `backward` in `ls_06.wav`, said quietly.

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
