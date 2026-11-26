"""
Build a keyword-spotting eval set from public speech datasets, plus speech-free background
noise, in the same <stem>.wav + <stem>_truth.csv format as eval_recordings/, so many more
speakers and hours of non-keyword audio can be scored than the board recordings give.

Background noise (dataset_noise/, download_noise.py), about as many hours as the speech
(--noise-min), so false accepts can be read separately on speech and on noise:
  MUSAN           music without vocals, and assorted noise (Free Sound, Sound Bible)
  DEMAND          environments: home, office, nature, street, transport
  Sounds of Home  7 homes, living room and kitchen mics
Only audio without speech is used: noise_speech.py finds it from the datasets' own labels (Sounds
of Home's sound-event labels, MUSAN's vocals annotation) and with YAMNet (MUSAN, DEMAND). MUSAN
files without vocals give every speech-free stretch of at least NOISE_MIN_PIECE_S, cut evenly into
pieces of at most NOISE_MAX_PIECE_S (files with vocals, or unknown vocals, aren't used). Sounds of
Home hours and DEMAND recordings are cut into NOISE_EXCERPT_S excerpts between their speech spans,
spread round-robin over the files. All of a split's MUSAN pieces and DEMAND excerpts go in; Sounds
of Home fills up to --noise-min. Noise keeps its recorded level (see Level).

Keyword positions come from the datasets' word-level time alignments, so
nothing is labelled by hand:
  LibriSpeech  read audiobooks, ~2,400 speakers; word times from the Montreal
               Forced Aligner alignments (zenodo.org/records/2619474)
  AMI          meetings; by default the Mix-Headset recording (all participants'
               close-talk headsets summed: the table-top array mics pick up too much
               room noise and echo); word times from the manual annotations
               (ami_public_manual_1)
  SWC          Spoken Wikipedia, English: ~500 volunteers reading articles on home
               mics; word times from the corpus' own alignment (aligned.swc), which
               covers about half the words

Expected layout (see eval_recordings_public/README.md for where to get it):
  dataset_public/LibriSpeech/<split>/<spk>/<chapter>/<utt>.flac + <spk>-<chapter>.trans.txt
  dataset_public/librispeech_alignments/<split>/<spk>/<chapter>/<utt>.TextGrid
  dataset_public/ami/ami_public_manual_1/{words,corpusResources}/
  dataset_public/ami/amicorpus/<meeting>/audio/<meeting>.<--ami-channel>.wav  (download_ami.py)
  dataset_public/swc/english/<article>/{aligned.swc,audio.ogg}   (SWC_English.tar from swc-2)

Word classes (uppercased, punctuation removed):
  FORWARD(S) / BACKWARD(S) / STOP   keyword span forward / backward / stop
  any other word containing one     'ignore' span (STOPPED, STRAIGHTFORWARD...):
                                    neither a hit nor a false accept
  NOT_KEYWORD (CHRISTOPHER...)      negative audio: spelt with "stop", not said
  everything else                   negative audio
Alignments that can't be trusted become 'ignore' too:
  - a keyword aligned shorter than MIN_KEYWORD_S (squeezed by the aligner), widened
  - SWC: a keyword / ignore word with no time, over the gap between its timed neighbours
  - SWC: a keyword produced by text normalisation ('/' is read as "forward slash")

Pieces:
  LibriSpeech  whole utterances (~13 s). Every keyword utterance of the split's speakers
               is used (--max-per-speaker / --max-per-keyword can limit them). Negative
               utterances have no keyword and no ignore word, at most
               LS_NEGATIVES_PER_SPEAKER per speaker, up to --ls-negative-min minutes.
  AMI, SWC     20 s excerpts. Each keyword gets an excerpt that contains it at a
               random offset (overlapping excerpts are merged), with the same limits.
               Negative excerpts contain no keyword or ignore word, up to
               --ami-negative-min / --swc-negative-min minutes, spread round-robin
               over meetings / readers. SWC negatives must be at least
               SWC_MIN_COVERAGE covered by aligned words. SWC audio is resampled to
               16 kHz and downmixed; articles below 16 kHz, split over several audio
               files (no timings) or that libsndfile can't open are skipped.

Each source's pieces are shuffled together and packed into ~5 minute files
(ls_<nn>.wav, ami_<nn>.wav, swc_<nn>.wav, musan_<nn>.wav, demand_<nn>.wav, soh_<nn>.wav),
separated by short gaps of very quiet noise. The same arguments always give the same files.
Noise files have no truth spans: everything that fires in them is a false accept.

Level: each piece has its DC removed. Speech pieces then get one static gain per source (AMI:
per meeting, since its meetings were recorded ~30 dB apart), chosen so the source's median
active-speech level (loudest 30% of 20 ms frames) is --level-dbfs, the board's level. The
natural loudness spread inside a source is kept. Noise keeps the level it was recorded (or, for
MUSAN, mastered) at: the datasets already range from quiet rooms to buses and loud music.
A peak limiter then keeps every sample at or below -1 dBFS by lowering the gain only around
the peaks.

Train / val / test split (by speaker, so no voice is in two splits): every group goes to
one split, ~50 / 25 / 25% of each keyword and of the utterances / recordings, assigned
greedily biggest group first. Groups:
  LibriSpeech  speaker
  SWC          reader (Wikipedia user name)
  AMI          meetings linked by any shared participant (connected components): the same
               people are in every meeting of a series, and 30 people are in more than one
  noise        dataset_noise/manifest.csv's group: MUSAN music by artist, Sound Bible noise by
               recorder, Free Sound noise by file, a DEMAND environment, a Sounds of Home home.
               Balanced on usable (speech-free) hours and group count. A noise group keeps the
               split it has in splits.csv, so downloading more audio never moves a group
Val and test are eval splits. Train is built the same way for training (its clips go into the
notebook's training split), and is never scored: snapshot_eval.py only accepts val and test.

Outputs in --out (default eval_recordings_public/):
  splits.csv            one row per group: source, group, split, members (the speakers /
                        readers / meetings in it; a noise group's own name), items, keyword
                        counts. Only groups with split = train may ever be trained on
  <split>/              val/, test/ and train/, each with:
    <stem>.wav          16 kHz, 16-bit mono (gitignored: rebuild instead)
    <stem>_truth.csv    start_s, end_s, label, file   (file = source utterance / excerpt id)
    manifest.csv        one row per piece: where it is in which WAV, and where it came from
    clips/<label>/      1 s clips cut from the WAVs: one per keyword (centred on the word,
                        or shifted to the piece's edge where that's the source's edge),
                        CLIPS_OTHER_PER_KEYWORD x as many 'other' (speech, no keyword) and
                        CLIPS_BACKGROUND_PER_KEYWORD x as many '_background_noise_' (from the
                        noise pieces, the model's background class)
    clips.csv           one row per clip: file, label, where it's from (and the word's span)
    build_info.json     settings of each source's build and the SHA-256 of every output file

After a build (or with --check, on an existing build) each split is checked:
  errors    a WAV / truth file / manifest row missing, not 16 kHz mono, files that
            don't match build_info.json, pieces outside their WAV or overlapping,
            a peak above PEAK_LIMIT_DBFS, empty spans, spans outside the piece
            they came from, unknown labels, a speaker that isn't in this split in
            splits.csv, a clip that isn't 1 s or is outside its piece, a keyword clip
            without its whole word, or neither centred on it nor at a piece edge that's
            a source edge, an 'other' clip overlapping a keyword / ignore span, a
            background clip outside a noise piece, a truth span in a noise file, a noise
            piece overlapping speech found by noise_speech.py, a noise group that isn't in
            this split -> exit code 1
  warnings  keyword spans shorter than MIN_KEYWORD_S or longer than MAX_KEYWORD_S,
            or QUIET_SPAN_DB quieter than the audio around them (misaligned?), and
            pieces with more than HIGH_GAIN_DB of gain (boosted near-silence?), a
            source median speech level more than 1 dB from the target, keywords
            too close to a piece edge inside the source (the piece needs more padding),
            a Sounds of Home excerpt with a muted stretch (missed speech removal?), and a
            source with less noise than requested
It also prints each source's level spread and how many pieces the limiter touched.
A build also reports, per source, whether it came out identical to the previous
build (build_info.json), and warns if it didn't with the same settings.

Usage (from HostSW/):
    python EvalScripts/noise_speech.py          # once, after download_noise.py
    python EvalScripts/make_public_eval.py --stats
    python EvalScripts/make_public_eval.py
    python EvalScripts/make_public_eval.py --check
    python EvalScripts/make_public_eval.py --clips-only     # re-cut the clips from the WAVs as built
    python EvalScripts/make_public_eval.py --splits val --sources ls --ls-negative-min 60
    python EvalScripts/make_public_eval.py --sources soh --noise-min 2400
"""

import argparse
import csv
import hashlib
import json
import math
import pathlib
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile as sf

import noise_speech

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
HOSTSW_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = HOSTSW_DIR / "dataset_public"
DEFAULT_NOISE_DIR = noise_speech.DEFAULT_NOISE_DIR
DEFAULT_OUT_DIR = HOSTSW_DIR / "eval_recordings_public"

SAMPLE_RATE_HZ = 16000
KEYWORDS = ["forward", "backward", "stop"]
IGNORE_LABEL = "ignore"
# Transcript word -> keyword label. Speakers often add an 's' to forward / backward,
# and the model should treat that the same.
KEYWORD_WORDS = {"FORWARD": "forward", "FORWARDS": "forward",
                 "BACKWARD": "backward", "BACKWARDS": "backward",
                 "STOP": "stop"}
KEYWORD_STEMS = ["FORWARD", "BACKWARD", "STOP"]
# Contain a keyword's letters but not its sound: negative audio, not 'ignore'
NOT_KEYWORD = {"CHRISTOPHER", "CHRISTOPHERS", "ARISTOPHANES", "ARISTOPHANIC"}

# Speech sources, then background noise. New sources go at the end: a source's index seeds its
# random generator, so inserting one would change the others' picks
SPEECH_SOURCES = ["ls", "ami", "swc"]
NOISE_SOURCES = ["musan", "demand", "soh"]
SOURCES = SPEECH_SOURCES + NOISE_SOURCES
SOURCE_NAMES = {"ls": "librispeech", "ami": "ami", "swc": "swc",
                "musan": "musan", "demand": "demand", "soh": "sounds_of_home"}
NOISE_SOURCE_OF = {"musan": "musan", "demand": "demand", "soh": "soh"}   # dataset_noise manifest 'source'

# Background noise. A MUSAN file without vocals gives its speech-free stretches of at least
# NOISE_MIN_PIECE_S (short sound effects included), cut evenly into pieces of at most
# NOISE_MAX_PIECE_S, so a long track is several pieces. Sounds of Home and DEMAND (EXCERPT_NOISE) give NOISE_EXCERPT_S
# excerpts between speech spans. A Sounds of Home excerpt with a muted stretch (MUTED_DBFS for
# MUTED_S: far below any room's noise floor, ~-60 dBFS) is skipped: speech removal that the
# labels missed. Defaults give ~35 h of noise per split, about as much as the speech
EXCERPT_NOISE = {"demand", "soh"}
NOISE_MIN_PIECE_S = 5.0
NOISE_MAX_PIECE_S = 300.0
NOISE_EXCERPT_S = 60.0
MUTED_DBFS = -90.0
MUTED_S = 0.5
DEFAULT_NOISE_MIN = 2100


# Speaker-group split. Every group (LibriSpeech speaker, SWC reader, AMI meetings linked by a
# shared participant) goes to one split, so no speaker is in two. Groups are assigned greedily,
# biggest first, each to the split that keeps every keyword's share and the share of items
# (utterances / recordings) closest to SPLIT_FRACTIONS. Each is built into <out>/<split>.
# A split's index seeds its random generators, so new splits go at the end: train first would
# change val and test
SPLITS = ["val", "test", "train"]
SPLIT_FRACTIONS = {"train": 0.5, "val": 0.25, "test": 0.25}
SPLITS_CSV = "splits.csv"

# 1 s clips cut from each split's built WAVs (same gain / limiter), for clip-level eval later:
# one per keyword span of at most CLIP_MAX_KEYWORD_S, centred on the word: word timings can be
# off by a few hundred ms, and centring leaves the most room on both sides for that (a longer
# word has too little). A window that would leave the piece is shifted back to the piece's edge,
# but only if that edge is also the edge of the source audio: there the word can't be any further
# out, so nothing is lost. A piece edge inside the source needs padding instead (excerpts have
# EXCERPT_EDGE_S), so such a keyword is skipped. Also skipped if the window overlaps another
# keyword / ignore span. Plus CLIPS_OTHER_PER_KEYWORD x as many 'other' clips: windows overlapping
# no keyword / ignore span, with speech in them (active level within CLIP_OTHER_MIN_REL_DB of
# their piece's)
CLIP_S = 1.0
CLIP_MAX_KEYWORD_S = 0.7
SOURCE_EDGE_TOL_S = 0.002   # the manifest's source times have 3 decimals
# Truth times have 6 decimals and file offsets aren't round, so a word exactly on a limit lands
# either side of it depending on where it is in the WAV. Clip rules allow this much for that
SPAN_ROUNDING_S = 1e-5
CLIPS_OTHER_PER_KEYWORD = 2
CLIP_OTHER_MIN_REL_DB = -15.0
OTHER_LABEL = "other"
# Background clips: random 1 s windows of the noise pieces, chosen in proportion to their length
CLIPS_BACKGROUND_PER_KEYWORD = 2
BACKGROUND_LABEL = "_background_noise_"
CLIP_LABELS = KEYWORDS + [OTHER_LABEL, BACKGROUND_LABEL]
CLIP_FIELDS = ["file", "label", "wav", "start_s", "end_s", "piece_id", "source", "span_start_s", "span_end_s"]
# Seeds the clips' random generator (the source count when there were only the speech sources,
# so their picks didn't change when noise was added)
CLIPS_RNG_KEY = 3

LS_NEGATIVES_PER_SPEAKER = 12
EXCERPT_S = 20.0         # AMI / SWC excerpt length
EXCERPT_EDGE_S = 2.0     # keyword at least this far inside its excerpt (padding for its 1 s clip)
# AMI recording to use (--ami-channel). Mix-Headset: speech ~43 dB above the background,
# like LibriSpeech; the Array1-01 table-top mic only ~18 dB
DEFAULT_AMI_CHANNEL = "Mix-Headset"
# Timings known to be incomplete or wrong (ami_public_manual_1/00README_MANUAL.txt)
AMI_BAD_MEETINGS = {"EN2002a", "EN2002c", "EN2003a", "TS3009c"}
# SWC aligns only about half the words: a negative excerpt needs aligned words covering at
# least this share of it, so it's read article text, not an unaligned stretch
SWC_MIN_COVERAGE = 0.4
# AMI words are all timed; this keeps negatives to stretches where someone is talking,
# rather than near-silent room noise that normalisation would boost to speech level
AMI_MIN_COVERAGE = 0.3
# An excerpt with a run of exact zeros at least this long, or quieter than DEAD_RMS_DBFS,
# contains a recording dropout (ES2012c has 20 s of zeros), not real audio: it's skipped.
# A run, not the share of zeros: clean recordings (AMI Mix-Headset) have many zero samples
# in pauses, but in runs of at most ~10 ms
DEAD_ZERO_RUN_S = 0.5
DEAD_RMS_DBFS = -70.0
# A keyword aligned shorter than this was squeezed by the aligner (spoken keywords take
# 0.25 s or more), so its position can't be trusted: it becomes an 'ignore' span widened
# by MISALIGNED_PAD_S each side. LibriSpeech has very few; SWC about 6% of its keywords.
MIN_KEYWORD_S = 0.2
MISALIGNED_PAD_S = 1.0
SWC_USER = re.compile(r"User(?:[ _]talk)?:([^|\]/]+)", re.I)

FILE_TARGET_S = 300.0
GAP_S = (0.2, 0.8)
GAP_NOISE_DBFS = -70.0

# Level. Each source gets one gain (AMI: one per meeting, since its meetings were recorded
# at levels ~30 dB apart), set so the source's median active-speech level is --level-dbfs.
# That keeps the natural loudness spread inside a source while every source's average
# matches the board. Default: the board recordings' active-speech level (-19..-22.6 dBFS).
DEFAULT_LEVEL_DBFS = -21.0
LEVEL_PER_RECORDING = {"ami"}
# Active-speech level: mean power of the loudest ACTIVE_TOP_SHARE of ACTIVE_FRAME_S frames,
# DC removed per frame, so pauses (and the board's drifting DC offset) don't count
ACTIVE_FRAME_S = 0.02
ACTIVE_TOP_SHARE = 0.3
# Peak limiter: where the gain would push a sample above PEAK_LIMIT_DBFS, the gain is
# lowered only around that peak (look-ahead window +-LIMITER_WINDOW_S, smoothed over the same),
# instead of clipping it or turning the whole piece down
PEAK_LIMIT_DBFS = -1.0
LIMITER_WINDOW_S = 0.005

OUTPUT_NAME = re.compile(rf"^({'|'.join(SOURCES)})_\d+(\.wav|_truth\.csv)$")
MANIFEST_FIELDS = ["wav", "start_s", "end_s", "source", "piece_id", "group", "speakers", "keywords", "category",
                   "src_file", "src_start_s", "src_end_s", "src_duration_s", "gain_db", "limit_db"]


def word_label(raw):
    """Keyword label, IGNORE_LABEL, or None (negative audio) for one transcript word."""
    word = re.sub(r"[^A-Z]", "", raw.upper())
    if raw.rstrip().endswith("-") and any(stem in word for stem in KEYWORD_STEMS):
        return IGNORE_LABEL  # AMI marks cut-off words with a trailing '-'
    if word in KEYWORD_WORDS:
        return KEYWORD_WORDS[word]
    if word in NOT_KEYWORD:
        return None
    if any(stem in word for stem in KEYWORD_STEMS):
        return IGNORE_LABEL
    return None


@dataclass
class Piece:
    """A stretch of source audio that goes into an eval file."""
    source: str
    piece_id: str
    speakers: list           # keyword speakers (or all speakers for a negative piece)
    group: str               # LibriSpeech split / AMI meeting
    audio_path: pathlib.Path
    start_s: float           # within the source audio
    end_s: float
    spans: list = field(default_factory=list)  # (start_s, end_s, label), relative to start_s
    category: str = ""       # noise: dataset_noise manifest category (musan/music/jamendo...)

    @property
    def duration_s(self):
        return self.end_s - self.start_s

    @property
    def keywords(self):
        return [label for _, _, label in self.spans if label in KEYWORDS]


# --- LibriSpeech -------------------------------------------------------------

@dataclass
class LsUtterance:
    utt_id: str
    speaker: str
    split: str
    words: list
    labels: list             # word_label() per word

    @property
    def has_keyword(self):
        return any(l in KEYWORDS for l in self.labels)

    @property
    def has_ignore(self):
        return IGNORE_LABEL in self.labels


def ls_read_transcripts(ls_root):
    """Returns {split: [LsUtterance]} for every split dir found."""
    splits = {}
    for split_dir in sorted(p for p in ls_root.iterdir() if p.is_dir()):
        utts = []
        for trans in sorted(split_dir.glob("*/*/*.trans.txt")):
            for line in trans.read_text().splitlines():
                utt_id, text = line.split(" ", 1)
                words = text.split()
                utts.append(LsUtterance(utt_id, utt_id.split("-")[0], split_dir.name, words,
                                        [word_label(w) for w in words]))
        if utts:
            splits[split_dir.name] = utts
    return splits


def ls_utt_path(root, utt, suffix):
    spk, chapter, _ = utt.utt_id.split("-")
    return root / utt.split / spk / chapter / f"{utt.utt_id}{suffix}"


def read_textgrid_words(path):
    """[(start_s, end_s, word)] from a TextGrid's 'words' tier, silences dropped."""
    text = path.read_text()
    tier = re.search(r'name = "words"(.*?)(?:item \[\d+\]:|\Z)', text, re.S)
    if not tier:
        raise ValueError(f"{path}: no 'words' tier")
    intervals = re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"', tier.group(1))
    return [(float(a), float(b), w) for a, b, w in intervals if w.strip()]


def ls_aligned_spans(align_root, utt):
    """Keyword and ignore spans of one utterance from its alignment, or None if the
    alignment is missing or doesn't match the transcript."""
    path = ls_utt_path(align_root, utt, ".TextGrid")
    if not path.exists():
        return None
    words = read_textgrid_words(path)
    # The aligner writes <unk> for words not in its dictionary (mostly names)
    if len(words) != len(utt.words) or any(
            w.upper() not in (word, "<UNK>") for (_, _, w), word in zip(words, utt.words)):
        return None
    return [(start, end, label) for (start, end, w), label in zip(words, utt.labels) if label]


def warn_if_short(source, negatives_s, negative_min, hint=""):
    """FA/hour rests on the negative audio, so a source that ran out shouldn't pass unnoticed."""
    if negatives_s < negative_min * 60:
        print(f"  WARNING: {SOURCE_NAMES[source]}: only {negatives_s / 60:.0f} of the "
              f"{negative_min:.0f} negative minutes requested{hint}")


def round_robin(items, key, rng):
    """Shuffles items so each group (e.g. speaker) gets its 1st pick before any gets a 2nd."""
    order = rng.permutation(len(items))
    rank_in_group = Counter()
    ranked = []
    for i in order:
        group = key(items[i])
        ranked.append((rank_in_group[group], len(ranked), items[i]))
        rank_in_group[group] += 1
    return [item for _, _, item in sorted(ranked, key=lambda r: r[:2])]


def select_keyword_pieces(candidates, max_per_keyword, max_per_speaker, rng):
    """Picks keyword pieces, rarest keyword first, round-robin over speakers. By default
    (limits None) that's every piece; a piece's keywords all count towards any limits."""
    per_keyword = Counter()
    per_speaker = Counter()
    chosen = []
    taken = set()
    totals = Counter(kw for p in candidates for kw in set(p.keywords))
    for kw in sorted(KEYWORDS, key=lambda k: totals[k]):
        pool = [p for p in candidates if kw in p.keywords and p.piece_id not in taken]
        for piece in round_robin(pool, lambda p: p.speakers[0], rng):
            if max_per_keyword is not None and per_keyword[kw] >= max_per_keyword:
                break
            if max_per_speaker is not None and per_speaker[(piece.speakers[0], kw)] >= max_per_speaker:
                continue
            chosen.append(piece)
            taken.add(piece.piece_id)
            for k in piece.keywords:
                per_keyword[k] += 1
                per_speaker[(piece.speakers[0], k)] += 1
    return chosen


def build_librispeech(utts, data_dir, args, rng):
    ls_root = data_dir / "LibriSpeech"
    align_root = data_dir / "librispeech_alignments"

    candidates = []
    unaligned = 0
    for utt in utts:
        if not utt.has_keyword:
            continue
        spans = ls_aligned_spans(align_root, utt)
        if spans is None:
            unaligned += 1
            continue
        flac = ls_utt_path(ls_root, utt, ".flac")
        duration = sf.info(str(flac)).duration
        spans = [(max(0.0, start - MISALIGNED_PAD_S), min(duration, end + MISALIGNED_PAD_S), IGNORE_LABEL)
                 if label in KEYWORDS and end - start < MIN_KEYWORD_S else (start, end, label)
                 for start, end, label in spans]
        candidates.append(Piece("ls", utt.utt_id, [utt.speaker], utt.split, flac, 0.0, duration, spans))
    if unaligned:
        print(f"  LibriSpeech: skipped {unaligned} keyword utterances with no matching alignment")
    pieces = select_keyword_pieces(candidates, args.max_per_keyword, args.max_per_speaker, rng)

    negatives_s = 0.0
    per_speaker = Counter()
    pool = [u for u in utts if not u.has_keyword and not u.has_ignore]
    for utt in round_robin(pool, lambda u: u.speaker, rng):
        if negatives_s >= args.ls_negative_min * 60:
            break
        if per_speaker[utt.speaker] >= LS_NEGATIVES_PER_SPEAKER:
            continue
        flac = ls_utt_path(ls_root, utt, ".flac")
        duration = sf.info(str(flac)).duration
        pieces.append(Piece("ls", utt.utt_id, [utt.speaker], utt.split, flac, 0.0, duration))
        per_speaker[utt.speaker] += 1
        negatives_s += duration
    warn_if_short("ls", negatives_s, args.ls_negative_min)
    return pieces


# --- Long recordings (AMI meetings, SWC articles) ----------------------------

@dataclass
class TimedWord:
    start_s: float | None
    end_s: float | None
    word: str
    label: str | None
    speaker: str


@dataclass
class Recording:
    """A long recording that excerpts are cut from."""
    rec_id: str
    audio_path: pathlib.Path
    duration_s: float
    words: list              # TimedWord, every one with a time
    group_key: str           # negatives are spread round-robin over these (meeting / reader)
    min_coverage: float = 0.0  # share of a negative excerpt aligned words must cover

    def __post_init__(self):
        timed = [w for w in self.words if w.start_s is not None]
        self.word_starts = np.array([w.start_s for w in timed])
        self.word_ends = np.array([w.end_s for w in timed])
        labelled = [w for w in timed if w.label]
        self.blocked_starts = np.array([w.start_s for w in labelled])
        self.blocked_ends = np.array([w.end_s for w in labelled])


def excerpt(source, rec, start_s, end_s):
    """A Piece for [start_s, end_s) of a recording. A keyword cut by the excerpt edge
    becomes an 'ignore' span."""
    spans, speakers = [], []
    for w in rec.words:
        if not w.label or w.end_s <= start_s or w.start_s >= end_s:
            continue
        label = w.label if start_s <= w.start_s and w.end_s <= end_s else IGNORE_LABEL
        spans.append((max(w.start_s, start_s) - start_s, min(w.end_s, end_s) - start_s, label))
        if label in KEYWORDS:
            speakers.append(w.speaker)
    if not speakers:
        speakers = sorted({w.speaker for w in rec.words if start_s <= w.start_s < end_s}) or ["?"]
    return Piece(source, f"{rec.rec_id}_{start_s:.2f}", speakers, rec.rec_id, rec.audio_path,
                 start_s, end_s, spans)


def read_mono(path, start_s, end_s):
    """[start_s, end_s) of an audio file, downmixed, at its own rate: (float32 audio, rate)."""
    rate = sf.info(str(path)).samplerate
    audio, _ = sf.read(str(path), start=round(start_s * rate), stop=round(end_s * rate),
                       dtype="float32", always_2d=True)
    return audio.mean(axis=1), rate


def frame_powers(audio, rate=SAMPLE_RATE_HZ):
    """Mean power of each ACTIVE_FRAME_S frame, DC removed per frame."""
    n = round(ACTIVE_FRAME_S * rate)
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    return ((frames - frames.mean(axis=1, keepdims=True)) ** 2).mean(axis=1)


def longest_run(mask):
    """Length of the longest run of True."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int((ends - starts).max()) if starts.size else 0


def muted_run_s(audio, rate=SAMPLE_RATE_HZ):
    """Longest stretch, in seconds, of ACTIVE_FRAME_S frames below MUTED_DBFS."""
    return longest_run(frame_powers(audio, rate) < 10 ** (MUTED_DBFS / 10)) * ACTIVE_FRAME_S


def is_dead_audio(audio_path, start_s, end_s, muted=False):
    """True for a stretch with a dropout, or near-silence (RMS below DEAD_RMS_DBFS). A dropout is
    DEAD_ZERO_RUN_S of digital zeros, or with muted, MUTED_S below MUTED_DBFS: how Sounds of Home's
    speech removal looks. Not that for speech: clean recordings' pauses can be that quiet."""
    audio, rate = read_mono(audio_path, start_s, end_s)
    if audio.size == 0:
        return True
    rms_dbfs = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-12)
    dropout = (muted_run_s(audio, rate) >= MUTED_S if muted
               else longest_run(audio == 0) >= DEAD_ZERO_RUN_S * rate)
    return dropout or rms_dbfs < DEAD_RMS_DBFS


def free_stretches(blocked, duration_s):
    """The [start, end) stretches of [0, duration_s) outside the blocked (start, end) spans."""
    stretches, last = [], 0.0
    for start, end in sorted(blocked) + [(duration_s, duration_s)]:
        if start > last:
            stretches.append((last, min(start, duration_s)))
        last = max(last, end)
    return stretches


def pick_excerpts(recordings, excerpt_s, target_s, accept, rng):
    """Excerpts of excerpt_s between blocked spans: [(key, start_s)]. recordings are
    (key, group, duration_s, blocked spans). Each free stretch holds as many excerpts as fit, end
    to end from a random offset; they're taken round-robin over the groups (each group's 1st
    before any group's 2nd, in random order), if accept(key, start_s, end_s), up to target_s."""
    slots = []
    for key, group, duration_s, blocked in recordings:
        for a, b in free_stretches(blocked, duration_s):
            n = int((b - a) // excerpt_s)
            if n:
                offset = a + float(rng.uniform(0, (b - a) - n * excerpt_s))
                slots += [(group, key, offset + k * excerpt_s) for k in range(n)]
    picked = []
    for _, key, start in round_robin(slots, lambda s: s[0], rng):
        if len(picked) * excerpt_s >= target_s:
            break
        if accept(key, start, start + excerpt_s):
            picked.append((key, start))
    return picked


def overlap_s(starts, ends, start_s, end_s):
    """Total overlap of the intervals with [start_s, end_s)."""
    return float(np.clip(np.minimum(ends, end_s) - np.maximum(starts, start_s), 0, None).sum())


def build_excerpts(source, recordings, negative_min, args, rng):
    """Keyword excerpts (one per keyword, the keyword at a random offset, overlaps merged)
    and negative excerpts (no keyword or ignore word) from long recordings."""
    candidates = []
    keyword_ranges = {}
    for rec in recordings:
        ranges = []
        for w in rec.words:
            if w.label not in KEYWORDS:
                continue
            room = max(0.0, EXCERPT_S - 2 * EXCERPT_EDGE_S - (w.end_s - w.start_s))
            start = w.start_s - EXCERPT_EDGE_S - rng.uniform(0, room)
            start = min(max(0.0, start), max(0.0, rec.duration_s - EXCERPT_S))
            ranges.append([start, min(rec.duration_s, start + EXCERPT_S)])
        merged = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        keyword_ranges[rec.rec_id] = merged
        candidates += [excerpt(source, rec, s, e) for s, e in merged
                       if not is_dead_audio(rec.audio_path, s, e)]
    pieces = select_keyword_pieces(candidates, args.max_per_keyword, args.max_per_speaker, rng)

    # Negative excerpts (pick_excerpts), round-robin over groups (meetings / readers): no keyword
    # or ignore word, no overlap with a keyword excerpt, enough of it covered by aligned words
    by_id = {rec.rec_id: rec for rec in recordings}

    def accept(rec_id, start, end):
        rec = by_id[rec_id]
        return (overlap_s(rec.word_starts, rec.word_ends, start, end) >= rec.min_coverage * (end - start)
                and not is_dead_audio(rec.audio_path, start, end))

    blocked = [(rec.rec_id, rec.group_key, rec.duration_s,
                [tuple(r) for r in keyword_ranges[rec.rec_id]] + list(zip(rec.blocked_starts, rec.blocked_ends)))
               for rec in recordings]
    negatives = pick_excerpts(blocked, EXCERPT_S, negative_min * 60, accept, rng)
    pieces += [excerpt(source, by_id[key], start, start + EXCERPT_S) for key, start in negatives]
    warn_if_short(source, len(negatives) * EXCERPT_S, negative_min)
    return pieces


def distrust_short_keywords(words):
    """Turns implausibly short keyword alignments into wider 'ignore' spans."""
    out = []
    for w in words:
        if w.label in KEYWORDS and w.end_s - w.start_s < MIN_KEYWORD_S:
            w = TimedWord(max(0.0, w.start_s - MISALIGNED_PAD_S), w.end_s + MISALIGNED_PAD_S,
                          w.word, IGNORE_LABEL, w.speaker)
        out.append(w)
    return out


# --- AMI ---------------------------------------------------------------------

def ami_speaker_names(manual_root):
    """{(meeting, nxt_agent letter): participant global name} from meetings.xml."""
    names = {}
    root = ET.parse(manual_root / "corpusResources" / "meetings.xml").getroot()
    for meeting in root.iter("meeting"):
        for spk in meeting.iter("speaker"):
            names[(meeting.get("observation"), spk.get("nxt_agent"))] = spk.get("global_name")
    return names


def ami_read_words(manual_root):
    """Returns {meeting: [TimedWord]} (all speakers, sorted by start time)."""
    names = ami_speaker_names(manual_root)
    meetings = defaultdict(list)
    for path in sorted((manual_root / "words").glob("*.words.xml")):
        meeting, agent = path.name.split(".")[:2]
        speaker = names.get((meeting, agent), f"{meeting}.{agent}")
        for w in ET.parse(path).getroot().iter("w"):
            if w.get("punc"):
                continue
            def time(attr):
                value = w.get(attr)
                return float(value) if value not in (None, "") else None
            text = w.text or ""
            meetings[meeting].append(TimedWord(time("starttime"), time("endtime"), text, word_label(text), speaker))
    for words in meetings.values():
        words.sort(key=lambda w: (w.start_s is None, w.start_s or 0.0))
    return dict(meetings)


def ami_audio_path(ami_root, meeting, channel):
    return ami_root / "amicorpus" / meeting / "audio" / f"{meeting}.{channel}.wav"


def ami_recordings(ami_root, channel):
    """Meetings with audio, good timings, and a time on every keyword / ignore word."""
    recordings = []
    for meeting, words in sorted(ami_read_words(ami_root / "ami_public_manual_1").items()):
        audio = ami_audio_path(ami_root, meeting, channel)
        if meeting in AMI_BAD_MEETINGS or not audio.exists():
            continue
        if any(w.label and (w.start_s is None or w.end_s is None) for w in words):
            print(f"  AMI: skipping {meeting}, a keyword or ignore word has no timing")
            continue
        timed = [w for w in words if w.start_s is not None and w.end_s is not None]
        recordings.append(Recording(meeting, audio, sf.info(str(audio)).duration,
                                    distrust_short_keywords(timed), meeting, AMI_MIN_COVERAGE))
    return recordings


def ami_speaker_groups(manual_root):
    """{meeting: group}: meetings that share any participant are in one group (connected
    components), so no person can end up in two splits. The same four people are in every
    meeting of a series (ES2002a-d), and 30 participants are in more than one series (the
    Idiap IS / IN meetings link 25 series). A group is named after its first series."""
    root = ET.parse(manual_root / "corpusResources" / "meetings.xml").getroot()
    people = defaultdict(set)
    for meeting in root.iter("meeting"):
        people[meeting.get("observation")] |= {spk.get("global_name") for spk in meeting.iter("speaker")}
    parent = {m: m for m in people}

    def find(m):
        while parent[m] != m:
            parent[m] = parent[parent[m]]
            m = parent[m]
        return m

    by_person = defaultdict(list)
    for meeting in sorted(people):
        for person in people[meeting]:
            by_person[person].append(meeting)
    for meetings in by_person.values():
        for meeting in meetings[1:]:
            parent[find(meeting)] = find(meetings[0])
    components = defaultdict(list)
    for meeting in sorted(people):
        components[find(meeting)].append(meeting)
    return {m: ami_series(min(ms)) for ms in components.values() for m in ms}


def ami_series(meeting):
    """ES2002a -> ES2002 (the meetings of a series have the same participants)."""
    return meeting[:-1] if meeting[-1].isalpha() else meeting


def build_ami(recordings, data_dir, args, rng):
    return build_excerpts("ami", recordings, args.ami_negative_min, args, rng)


# --- Spoken Wikipedia (SWC) --------------------------------------------------

def swc_reader(root, article):
    """Reader's Wikipedia user name from the alignment metadata (wiki markup), lowercased,
    spaces as '_' (manifest speakers are space-separated)."""
    name = next((p.get("value") for p in root.find("meta") if p.get("key") == "reader.name"), "") or ""
    m = SWC_USER.search(name)
    reader = m.group(1) if m else re.sub(r"<[^>]*>|\[|\]|\{|\}", "", name)
    reader = re.sub(r"\s+", "_", reader.strip().lower())
    return reader or f"unknown-{article}"


def swc_read_words(path, article):
    """Returns (reader, [TimedWord] in reading order). Untimed words keep None times."""
    root = ET.parse(path).getroot()
    reader = swc_reader(root, article)
    words = []

    def walk(el):
        if el.tag == "ignored":   # text that wasn't read (tables, references...)
            return
        if el.tag == "t":
            token_label = word_label(el.text or "")
            for n in el.findall("n"):
                pron = n.get("pronunciation") or ""
                label = word_label(pron)
                if label in KEYWORDS and token_label != label:
                    # A keyword produced by text normalisation, e.g. '/' -> "forward slash"
                    label = IGNORE_LABEL
                start, end = n.get("start"), n.get("end")
                words.append(TimedWord(int(start) / 1000 if start else None,
                                       int(end) / 1000 if end else None, pron, label, reader))
            return
        for child in el:
            walk(child)

    walk(root.find("d"))
    return reader, words


def fill_untimed(words, duration_s):
    """Drops untimed words, except keyword / ignore words: those can't be placed, so they
    become an 'ignore' span covering the gap between their timed neighbours, padded by
    MISALIGNED_PAD_S each side. The neighbours often touch (the aligner left no room for
    the word), so without the padding the span could be empty."""
    prev_end = []
    last = 0.0
    for w in words:
        prev_end.append(last)
        if w.end_s is not None:
            last = w.end_s
    next_start = [0.0] * len(words)
    nxt = duration_s
    for i in range(len(words) - 1, -1, -1):
        next_start[i] = nxt
        if words[i].start_s is not None:
            nxt = words[i].start_s
    filled = []
    for i, w in enumerate(words):
        if w.start_s is not None and w.end_s is not None:
            filled.append(w)
        elif w.label:
            start = max(0.0, prev_end[i] - MISALIGNED_PAD_S)
            end = min(duration_s, max(prev_end[i], next_start[i]) + MISALIGNED_PAD_S)
            filled.append(TimedWord(start, end, w.word, IGNORE_LABEL, w.speaker))
    filled.sort(key=lambda w: w.start_s)
    return filled


def swc_articles(swc_root):
    """(article dir, alignment, audio) for articles with one audio file: multi-part
    articles have no word timings."""
    for article in sorted(p for p in swc_root.iterdir() if p.is_dir()):
        audio, aligned = article / "audio.ogg", article / "aligned.swc"
        if audio.exists() and aligned.exists():
            yield article, aligned, audio


def swc_recordings(swc_root):
    recordings = []
    unreadable = low_rate = 0
    for article, aligned, audio in swc_articles(swc_root):
        try:
            info = sf.info(str(audio))
        except sf.LibsndfileError:
            unreadable += 1   # e.g. Ogg with a Skeleton stream, which libsndfile can't open
            continue
        if info.samplerate < SAMPLE_RATE_HZ:
            low_rate += 1
            continue
        reader, words = swc_read_words(aligned, article.name)
        words = distrust_short_keywords(fill_untimed(words, info.duration))
        words.sort(key=lambda w: w.start_s)
        recordings.append(Recording(article.name, audio, info.duration, words, reader, SWC_MIN_COVERAGE))
    print(f"  SWC: {len(recordings)} articles, skipped {unreadable} unreadable and "
          f"{low_rate} below {SAMPLE_RATE_HZ} Hz")
    return recordings


def build_swc(recordings, data_dir, args, rng):
    return build_excerpts("swc", recordings, args.swc_negative_min, args, rng)


# --- Background noise (dataset_noise) ----------------------------------------

@dataclass
class NoiseFile:
    """A file of dataset_noise/manifest.csv, with its speech spans (noise_speech.py)."""
    source: str              # this script's source: musan / demand / soh
    file: str                # relative to the noise dir
    path: pathlib.Path
    category: str
    group: str
    vocals: str
    duration_s: float
    speech: list             # [(start_s, end_s)], widened by noise_speech's margin

    @property
    def kind(self):
        """Split balance dimension: a MUSAN folder (music/jamendo, noise/free-sound...), so each
        split gets its share of every one, or the source for DEMAND / Sounds of Home."""
        return self.category.split("/", 1)[1] if self.source == "musan" else self.source

    def piece_spans(self):
        """MUSAN: [(start_s, end_s)] of its pieces: each speech-free stretch of at least
        NOISE_MIN_PIECE_S, cut evenly into pieces of at most NOISE_MAX_PIECE_S. None with vocals
        (or unknown vocals)."""
        if self.vocals != "no":
            return []
        spans = []
        for a, b in free_stretches(self.speech, self.duration_s):
            if b - a >= NOISE_MIN_PIECE_S:
                n = math.ceil((b - a) / NOISE_MAX_PIECE_S)
                spans += [(a + k * (b - a) / n, a + (k + 1) * (b - a) / n) for k in range(n)]
        return spans

    def usable_s(self):
        """Seconds of eval noise this file can give: excerpts between speech (EXCERPT_NOISE), or
        its MUSAN pieces."""
        if self.source in EXCERPT_NOISE:
            return sum((b - a) // NOISE_EXCERPT_S * NOISE_EXCERPT_S
                       for a, b in free_stretches(self.speech, self.duration_s))
        return sum(b - a for a, b in self.piece_spans())


_speech_scores = {}


def speech_scores(noise_dir):
    """noise_speech.py's scores, loaded once."""
    if noise_dir not in _speech_scores:
        _speech_scores[noise_dir] = noise_speech.load_scores(noise_dir)
    return _speech_scores[noise_dir]


def load_noise_source(source, noise_dir):
    """A noise source's files with their speech spans, or None if the data isn't there."""
    name = SOURCE_NAMES[source]
    manifest = noise_dir / "manifest.csv"
    if not manifest.exists():
        print(f"  {name}: {manifest} not found, skipping (run download_noise.py)")
        return None
    try:
        scores = speech_scores(noise_dir)
    except FileNotFoundError as e:
        print(f"  {name}: {e}, skipping")
        return None
    files, unanalysed = [], 0
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] != NOISE_SOURCE_OF[source]:
                continue
            duration = float(r["duration_s"])
            try:
                speech = noise_speech.speech_spans(scores, r["file"], duration)
            except KeyError:
                unanalysed += 1
                continue
            files.append(NoiseFile(source, r["file"], noise_dir / r["file"], r["category"], r["group"],
                                   r["vocals"], duration, speech))
    if unanalysed:
        print(f"  WARNING: {name}: {unanalysed} files not checked for speech yet, left out "
              f"(run python EvalScripts/noise_speech.py)")
    if not files:
        print(f"  {name}: no files in {manifest}, skipping")
        return None
    groups = {id(f): f.group for f in files}
    return SourceData(source, files, groups, {g: [g] for g in set(groups.values())})


def noise_piece(source, f, start_s, end_s):
    folder = pathlib.PurePath(f.file).parent.name   # DEMAND files are all ch01, Sounds of Home stems repeat per mic
    return Piece(source, f"{folder}_{f.path.stem}_{start_s:.2f}", [], f.group, f.path, start_s, end_s,
                 category=f.category)


def build_musan_pieces(source, files):
    """Every file's MUSAN pieces (NoiseFile.piece_spans)."""
    return [noise_piece(source, f, a, b) for f in files for a, b in f.piece_spans()]


def build_noise_excerpts(source, files, target_s, rng):
    """NOISE_EXCERPT_S excerpts between speech spans (pick_excerpts), round-robin over the files,
    up to target_s, without dropouts (Sounds of Home: without muted stretches either)."""
    by_file = {f.file: f for f in files}
    muted = source == "soh"
    picked = pick_excerpts([(f.file, f.file, f.duration_s, f.speech) for f in files], NOISE_EXCERPT_S, target_s,
                           lambda key, start, end: not is_dead_audio(by_file[key].path, start, end, muted), rng)
    return [noise_piece(source, by_file[key], start, start + NOISE_EXCERPT_S) for key, start in picked]


def fixed_noise_s(noise, split):
    """Seconds of MUSAN and DEMAND noise a split gets (all it has; Sounds of Home fills the rest)."""
    total = 0.0
    for source in ("musan", "demand"):
        if source in noise:
            loaded, split_of = noise[source]
            total += sum(f.usable_s() for f in loaded.items if split_of[f.group] == split)
    return total


# --- Sources and the train / val / test split --------------------------------

@dataclass
class SourceData:
    """A loaded source: its items (LsUtterance or Recording), the split group of each item,
    and the members of each group (what a training loader holds out / selects)."""
    source: str
    items: list
    group: dict              # id(item) -> group
    members: dict            # group -> sorted list of member ids (speaker / reader / meeting)

    def keywords(self, item):
        if isinstance(item, LsUtterance):
            return [l for l in item.labels if l in KEYWORDS]
        if isinstance(item, NoiseFile):
            return []
        return [w.label for w in item.words if w.label in KEYWORDS]


def load_source(source, data_dir, args):
    """Loads a source's utterances / recordings, or None if its data isn't there."""
    if source == "ls":
        ls_root = data_dir / "LibriSpeech"
        if not ls_root.is_dir():
            print(f"  LibriSpeech: {ls_root} not found, skipping")
            return None
        utts = [u for split in ls_read_transcripts(ls_root).values() for u in split]
        groups = {id(u): u.speaker for u in utts}
        members = defaultdict(set)
        for u in utts:
            members[u.speaker].add(u.speaker)
    elif source == "ami":
        recordings = ami_recordings(data_dir / "ami", args.ami_channel)
        if not recordings:
            print(f"  AMI: no meetings with {args.ami_channel} audio found, skipping (run download_ami.py)")
            return None
        meeting_group = ami_speaker_groups(data_dir / "ami" / "ami_public_manual_1")
        groups = {id(r): meeting_group[r.rec_id] for r in recordings}
        members = defaultdict(set)
        for meeting, group in meeting_group.items():
            members[group].add(meeting)
        utts = recordings
    else:
        swc_root = data_dir / "swc" / "english"
        if not swc_root.is_dir():
            print(f"  SWC: {swc_root} not found, skipping")
            return None
        recordings = swc_recordings(swc_root)
        groups = {id(r): r.group_key for r in recordings}
        members = defaultdict(set)
        for r in recordings:
            members[r.group_key].add(r.group_key)
        utts = recordings
    return SourceData(source, utts, groups, {g: sorted(m) for g, m in members.items()})


def group_sizes(data):
    """{group: Counter} of what a split balances: per speech group its keyword counts and items
    (utterances / recordings), per noise group its usable hours per kind (NoiseFile.kind) and 1 group."""
    sizes = defaultdict(Counter)
    for item in data.items:
        size = sizes[data.group[id(item)]]
        if data.source in NOISE_SOURCES:
            size[item.kind] += item.usable_s() / 3600
            size["groups"] = 1
        else:
            size.update(data.keywords(item))
            size["items"] += 1
    return sizes


def split_dims(data):
    """What a source's split shows and balances, apart from its group / item count: keywords, or
    noise kinds."""
    return sorted({f.kind for f in data.items}) if data.source in NOISE_SOURCES else KEYWORDS


def assign_splits(data, seed, existing=None):
    """({group: 'train' | 'val' | 'test'}, group_sizes()) for one source. Groups in existing
    ({group: split}, noise sources: from splits.csv) keep their split, so downloading more audio
    never moves a group whose audio may already have been trained or evaluated on. The rest go
    greedily in order of size (keywords / usable hours, then items; ties in seeded random order),
    each to the split that minimises the squared distance of every dimension's split shares from
    SPLIT_FRACTIONS. Big groups go first so the many small ones can even out the rest."""
    sizes = group_sizes(data)
    dims = split_dims(data) + (["groups"] if data.source in NOISE_SOURCES else ["items"])
    totals = {d: sum(c[d] for c in sizes.values()) or 1 for d in dims}
    filled = {split: Counter() for split in SPLIT_FRACTIONS}
    split_of = {}
    for group, split in (existing or {}).items():
        split_of[group] = split
        filled[split] += sizes.get(group, Counter())

    def cost(split, size):
        total = 0.0
        for d in dims:
            for sp, frac in SPLIT_FRACTIONS.items():
                share = (filled[sp][d] + (size[d] if sp == split else 0)) / totals[d]
                total += (share - frac) ** 2
        return total

    rng = np.random.default_rng(seed)
    groups = sorted(g for g in sizes if g not in split_of)
    groups = [groups[i] for i in rng.permutation(len(groups))]
    groups.sort(key=lambda g: (-sum(sizes[g][d] for d in dims[:-1]), -sizes[g][dims[-1]]))   # stable: ties stay random
    for group in groups:
        best = min(SPLIT_FRACTIONS, key=lambda sp: cost(sp, sizes[group]))
        filled[best] += sizes[group]
        split_of[group] = best
    return split_of, sizes


def existing_splits(out_dir, source):
    """{group: split} of a source in splits.csv ({} if there's none)."""
    path = out_dir / SPLITS_CSV
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["group"]: r["split"] for r in csv.DictReader(f) if r["source"] == SOURCE_NAMES[source]}


def write_splits(out_dir, assignments):
    """Writes splits.csv: one row per group, keeping rows of sources not reassigned this time.
    It's what the training step reads: only 'train' groups may be trained on. assignments:
    {source: (SourceData, split_of, group_sizes())}."""
    path = out_dir / SPLITS_CSV
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["source"] not in {SOURCE_NAMES[s] for s in assignments}]
    for source, (data, split_of, sizes) in assignments.items():
        items_per_group = Counter(data.group[id(i)] for i in data.items)
        for group in sorted(split_of):
            rows.append({"source": SOURCE_NAMES[source], "group": group, "split": split_of[group],
                         "members": " ".join(data.members.get(group, [group])),
                         "items": items_per_group[group],
                         **{f"n_{kw}": sizes.get(group, Counter())[kw] for kw in KEYWORDS}})
    rows.sort(key=lambda r: (SOURCES.index(next(k for k, v in SOURCE_NAMES.items() if v == r["source"])), r["group"]))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}: groups per split, and their keywords (speech) or usable hours (noise)")
    for source, (data, split_of, sizes) in assignments.items():
        noise = source in NOISE_SOURCES
        dims = split_dims(data)
        per_split = {split: Counter() for split in SPLIT_FRACTIONS}
        for group, split in split_of.items():
            per_split[split] += sizes.get(group, Counter())
        n_groups = Counter(split_of.values())
        for split, filled in per_split.items():
            cells = [f"{d} {filled[d]:6.1f} h" if noise else f"{d} {int(filled[d]):4d}" for d in dims]
            shares = [100 * filled[d] / (sum(c[d] for c in per_split.values()) or 1) for d in dims]
            print(f"  {SOURCE_NAMES[source]:14s} {split:5s} {n_groups[split]:5d} groups  "
                  + "  ".join(f"{c} ({p:3.0f}%)" for c, p in zip(cells, shares)))


def read_splits(out_dir):
    """{(source name, member): split} from splits.csv, or None if there isn't one."""
    path = out_dir / SPLITS_CSV
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return {(r["source"], m): r["split"] for r in csv.DictReader(f) for m in r["members"].split()}


# --- Assembly ----------------------------------------------------------------

def load_piece(piece):
    """The piece's audio as 16 kHz mono float32 (resampled / downmixed if needed)."""
    audio, rate = read_mono(piece.audio_path, piece.start_s, piece.end_s)
    if rate != SAMPLE_RATE_HZ:
        g = math.gcd(rate, SAMPLE_RATE_HZ)
        audio = scipy.signal.resample_poly(audio, SAMPLE_RATE_HZ // g, rate // g).astype(np.float32)
    return audio


def active_speech_db(audio):
    """Speech level ignoring pauses: mean power of the loudest ACTIVE_TOP_SHARE of
    ACTIVE_FRAME_S frames, with DC removed per frame."""
    power = np.sort(frame_powers(audio))
    if power.size == 0:
        return -240.0
    top = power[-max(1, round(ACTIVE_TOP_SHARE * len(power))):]
    return float(10 * np.log10(np.mean(top) + 1e-12))


def limit_peaks(audio, limit_dbfs=PEAK_LIMIT_DBFS):
    """Look-ahead peak limiter: keeps every sample at or below limit_dbfs by lowering the
    gain only around the peaks. The gain each sample needs is minimum-filtered over
    +-LIMITER_WINDOW_S (so it drops before a peak arrives), then averaged over the same
    window (a smooth ramp). Every value the average uses is at most the gain the peak
    needs, so it can't overshoot. Returns (audio, largest reduction in dB)."""
    limit = 10 ** (limit_dbfs / 20)
    need = np.minimum(1.0, limit / np.maximum(np.abs(audio), 1e-12))
    if need.min() >= 1.0:
        return audio, 0.0
    width = 2 * round(LIMITER_WINDOW_S * SAMPLE_RATE_HZ) + 1
    gain = scipy.ndimage.uniform_filter1d(scipy.ndimage.minimum_filter1d(need, width, mode="nearest"),
                                          width, mode="nearest")
    gain = np.minimum(gain, need)   # guards against float rounding in the average
    return (audio * gain).astype(np.float32), float(-20 * np.log10(gain.min()))


def piece_gains(source, pieces, level_dbfs):
    """Static gain in dB per piece_id: one per source, or one per recording for sources in
    LEVEL_PER_RECORDING (from the whole recording's active-speech level), shifted so the
    source's median piece active-speech level is level_dbfs. Noise keeps its recorded level: 0 dB."""
    if source in NOISE_SOURCES:
        return {p.piece_id: 0.0 for p in pieces}
    levels = {}
    for piece in pieces:
        audio = load_piece(piece)
        levels[piece.piece_id] = active_speech_db(audio - audio.mean())
    offsets = {}
    if source in LEVEL_PER_RECORDING:
        recording_db = {}
        for piece in pieces:
            if piece.audio_path not in recording_db:
                audio, _ = sf.read(str(piece.audio_path), dtype="float32", always_2d=True)
                recording_db[piece.audio_path] = active_speech_db(audio.mean(axis=1))
            offsets[piece.piece_id] = -recording_db[piece.audio_path]
    else:
        offsets = {p.piece_id: 0.0 for p in pieces}
    shift = level_dbfs - float(np.median([levels[k] + offsets[k] for k in levels]))
    return {k: offsets[k] + shift for k in levels}


def assemble(source, pieces, out_dir, args, rng):
    """Shuffles the pieces into ~FILE_TARGET_S files. Returns (manifest rows, file count)."""
    order = [pieces[i] for i in rng.permutation(len(pieces))]
    files, current, current_s = [], [], 0.0
    for piece in order:
        current.append(piece)
        current_s += piece.duration_s + sum(GAP_S) / 2
        if current_s >= FILE_TARGET_S:
            files.append(current)
            current, current_s = [], 0.0
    if current:
        if files and current_s < FILE_TARGET_S / 2:
            files[-1] += current  # fold a short remainder into the last file
        else:
            files.append(current)

    gains_db = piece_gains(source, pieces, args.level_dbfs)
    noise_sigma = 10 ** (GAP_NOISE_DBFS / 20)
    manifest = []
    for n, file_pieces in enumerate(files):
        stem = f"{source}_{n:02d}"

        def gap():
            return rng.normal(0, noise_sigma, round(rng.uniform(*GAP_S) * SAMPLE_RATE_HZ)).astype(np.float32)

        chunks = [gap()]
        pos = len(chunks[0])
        truth = []
        for piece in file_pieces:
            audio = load_piece(piece)
            gain_db = gains_db[piece.piece_id]
            audio, limit_db = limit_peaks((audio - audio.mean()) * np.float32(10 ** (gain_db / 20)))
            offset_s = pos / SAMPLE_RATE_HZ
            for start_s, end_s, label in piece.spans:
                truth.append((offset_s + start_s, offset_s + end_s, label, piece.piece_id))
            manifest.append({
                "wav": f"{stem}.wav", "start_s": f"{offset_s:.6f}",
                "end_s": f"{offset_s + len(audio) / SAMPLE_RATE_HZ:.6f}",
                "source": SOURCE_NAMES[source], "piece_id": piece.piece_id, "group": piece.group,
                "speakers": " ".join(sorted(set(piece.speakers))),
                "keywords": " ".join(piece.keywords),
                "category": piece.category,
                # relative to its dataset dir: dataset_public/, or dataset_noise/ for noise
                "src_file": str(piece.audio_path.relative_to(
                    args.noise_dir if source in NOISE_SOURCES else args.data_dir)),
                "src_start_s": f"{piece.start_s:.3f}", "src_end_s": f"{piece.end_s:.3f}",
                "src_duration_s": f"{sf.info(str(piece.audio_path)).duration:.3f}",
                "gain_db": f"{gain_db:.2f}", "limit_db": f"{limit_db:.2f}",
            })
            g = gap()
            chunks += [audio, g]
            pos += len(audio) + len(g)

        signal = np.clip(np.concatenate(chunks), -1.0, 32767 / 32768)
        sf.write(str(out_dir / f"{stem}.wav"), signal, SAMPLE_RATE_HZ, subtype="PCM_16")
        with open(out_dir / f"{stem}_truth.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["start_s", "end_s", "label", "file"])
            for start_s, end_s, label, piece_id in sorted(truth):
                # 6 dp so times are sample-exact (1 sample = 62.5 us at 16 kHz)
                writer.writerow([f"{start_s:.6f}", f"{end_s:.6f}", label, piece_id])
    return manifest, len(files)


def read_split_build(out_dir):
    """(manifest rows grouped by WAV, truth spans per WAV) of a built split."""
    with open(out_dir / "manifest.csv", newline="") as f:
        manifest = list(csv.DictReader(f))
    by_wav = defaultdict(list)
    for r in manifest:
        by_wav[r["wav"]].append(r)
    spans = {}
    for wav in by_wav:
        with open(out_dir / wav.replace(".wav", "_truth.csv"), newline="") as f:
            spans[wav] = [(float(r["start_s"]), float(r["end_s"]), r["label"], r["file"]) for r in csv.DictReader(f)]
    return by_wav, spans


def at_source_start(piece):
    """True if a manifest row's piece starts where its source audio starts."""
    return float(piece["src_start_s"]) <= SOURCE_EDGE_TOL_S


def at_source_end(piece):
    """True if a manifest row's piece ends where its source audio ends."""
    return float(piece["src_end_s"]) >= float(piece["src_duration_s"]) - SOURCE_EDGE_TOL_S


def place_keyword_clip(start_s, end_s, piece):
    """Start of the 1 s clip for a keyword span, in the piece's WAV: centred on the word, or
    shifted back to the piece's edge if that's also the source's edge. Returns (start_s, None),
    or (None, the reason there's no clip)."""
    piece_start, piece_end = float(piece["start_s"]), float(piece["end_s"])
    start = (start_s + end_s - CLIP_S) / 2
    if start < piece_start:
        if not at_source_start(piece):
            return None, "too close to a piece start inside the source (needs padding)"
        start = piece_start
    elif start + CLIP_S > piece_end:
        if not at_source_end(piece):
            return None, "too close to a piece end inside the source (needs padding)"
        start = piece_end - CLIP_S
    if start < piece_start or start + CLIP_S > piece_end:
        return None, "piece shorter than a clip"
    return start, None


def clip_length_ok(start_s, end_s):
    """True if a keyword span is short enough to get a clip."""
    return end_s - start_s <= CLIP_MAX_KEYWORD_S + SPAN_ROUNDING_S


def overlaps_any(spans, start_s, end_s, skip=None):
    return any(s < end_s and start_s < e for s, e, label, pid in spans if (s, e, label, pid) != skip)


def write_clips(out_dir, rng):
    """Cuts 1 s clips from a split's built WAVs into <out_dir>/clips/<label>/ and writes
    clips.csv: keyword clips (see place_keyword_clip) and CLIPS_OTHER_PER_KEYWORD x
    as many 'other' clips (speech overlapping no keyword / ignore span). Rebuilt from scratch
    from the WAVs on disk, so it always matches them."""
    clips_dir = out_dir / "clips"
    if clips_dir.exists():
        shutil.rmtree(clips_dir)
    by_wav, spans = read_split_build(out_dir)
    n = round(CLIP_S * SAMPLE_RATE_HZ)
    rows = []
    skipped = Counter()

    def save(pcm, wav, start, label, piece, span=None):
        i = round(start * SAMPLE_RATE_HZ)
        name = f"{wav[:-4]}_{i:08d}.wav"
        (clips_dir / label).mkdir(parents=True, exist_ok=True)
        sf.write(str(clips_dir / label / name), pcm[i:i + n], SAMPLE_RATE_HZ, subtype="PCM_16")
        rows.append({"file": f"clips/{label}/{name}", "label": label, "wav": wav,
                     "start_s": f"{i / SAMPLE_RATE_HZ:.6f}", "end_s": f"{(i + n) / SAMPLE_RATE_HZ:.6f}",
                     "piece_id": piece["piece_id"], "source": piece["source"],
                     "span_start_s": f"{span[0]:.6f}" if span else "", "span_end_s": f"{span[1]:.6f}" if span else ""})

    # Keyword clips first: their count sets how many 'other' clips each WAV gets
    keyword_clips = Counter()
    shifted = 0
    for wav in sorted(by_wav):
        pcm, _ = sf.read(str(out_dir / wav), dtype="int16")
        pieces = {p["piece_id"]: p for p in by_wav[wav]}
        for span in spans[wav]:
            start_s, end_s, label, pid = span
            if label not in KEYWORDS:
                continue
            if not clip_length_ok(start_s, end_s):
                skipped[f"keyword longer than {CLIP_MAX_KEYWORD_S} s"] += 1
                continue
            piece = pieces[pid]
            start, problem = place_keyword_clip(start_s, end_s, piece)
            if problem:
                skipped[problem] += 1
                continue
            if overlaps_any(spans[wav], start + SPAN_ROUNDING_S, start + CLIP_S - SPAN_ROUNDING_S, skip=span):
                skipped["another keyword / ignore span in the window"] += 1
                continue
            save(pcm, wav, start, label, piece, span)
            keyword_clips[wav] += 1
            shifted += abs(start - (start_s + end_s - CLIP_S) / 2) > 1 / SAMPLE_RATE_HZ

    # 'other' clips: spread over the WAVs in proportion to their keyword clips, round-robin
    # over each WAV's pieces
    for wav in sorted(by_wav):
        need = CLIPS_OTHER_PER_KEYWORD * keyword_clips[wav]
        if not need:
            continue
        pcm, _ = sf.read(str(out_dir / wav), dtype="int16")
        audio = pcm.astype(np.float32) / 32768
        pieces = [p for p in sorted(by_wav[wav], key=lambda p: float(p["start_s"]))
                  if float(p["end_s"]) - float(p["start_s"]) > CLIP_S]
        level = {p["piece_id"]: active_speech_db(audio[round(float(p["start_s"]) * SAMPLE_RATE_HZ):
                                                       round(float(p["end_s"]) * SAMPLE_RATE_HZ)]) for p in pieces}
        taken, tries = [], 0
        while len(taken) < need and pieces and tries < 50 * need:
            for piece in [pieces[i] for i in rng.permutation(len(pieces))]:
                tries += 1
                start = rng.uniform(float(piece["start_s"]), float(piece["end_s"]) - CLIP_S)
                end = start + CLIP_S
                if overlaps_any(spans[wav], start, end) or any(a < end and start < b for a, b in taken):
                    continue
                window = audio[round(start * SAMPLE_RATE_HZ):round(start * SAMPLE_RATE_HZ) + n]
                if active_speech_db(window) < level[piece["piece_id"]] + CLIP_OTHER_MIN_REL_DB:
                    continue   # mostly pause, not speech
                taken.append((start, end))
                save(pcm, wav, start, OTHER_LABEL, piece)
                if len(taken) >= need:
                    break
        if len(taken) < need:
            skipped["'other' clips short (not enough free speech)"] += need - len(taken)

    # Background clips: CLIPS_BACKGROUND_PER_KEYWORD x the keyword clips, random 1 s windows of
    # the noise pieces (a piece chosen in proportion to its length), not overlapping each other.
    # Positions first, then each WAV is read once
    noise_names = {SOURCE_NAMES[s] for s in NOISE_SOURCES}
    noise_pieces = [(wav, p) for wav in sorted(by_wav) for p in sorted(by_wav[wav], key=lambda p: float(p["start_s"]))
                    if p["source"] in noise_names and float(p["end_s"]) - float(p["start_s"]) > CLIP_S]
    need = CLIPS_BACKGROUND_PER_KEYWORD * sum(keyword_clips.values())
    if need and noise_pieces:
        room = np.array([float(p["end_s"]) - float(p["start_s"]) - CLIP_S for _, p in noise_pieces])
        chosen = defaultdict(list)
        for _ in range(50 * need):
            if sum(len(v) for v in chosen.values()) >= need:
                break
            wav, piece = noise_pieces[rng.choice(len(noise_pieces), p=room / room.sum())]
            start = float(rng.uniform(float(piece["start_s"]), float(piece["end_s"]) - CLIP_S))
            if not any(s < start + CLIP_S and start < s + CLIP_S for s, _ in chosen[wav]):
                chosen[wav].append((start, piece))
        for wav in sorted(chosen):
            pcm, _ = sf.read(str(out_dir / wav), dtype="int16")
            for start, piece in chosen[wav]:
                save(pcm, wav, start, BACKGROUND_LABEL, piece)
        got = sum(len(v) for v in chosen.values())
        if got < need:
            skipped["background clips short"] += need - got
    elif need:
        skipped["background clips (no noise in this split)"] += need

    rows.sort(key=lambda r: r["file"])
    with open(out_dir / "clips.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CLIP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(r["label"] for r in rows)
    print(f"Wrote {len(rows)} clips to {clips_dir}: " + ", ".join(f"{k} {counts[k]}" for k in CLIP_LABELS)
          + f" ({shifted} keyword clips shifted to a source edge)"
          + ("; skipped " + ", ".join(f"{v} {k}" for k, v in skipped.items()) if skipped else ""))


def clips_digest(out_dir):
    """One SHA-256 over every clip file (path + content), for build_info.json."""
    h = hashlib.sha256()
    for path in sorted((out_dir / "clips").rglob("*.wav")):
        h.update(str(path.relative_to(out_dir)).encode())
        h.update(file_sha256(path).encode())
    return h.hexdigest()


def summarise(source, pieces, n_files):
    if source in NOISE_SOURCES:
        minutes = Counter()
        for p in pieces:
            minutes[p.category.split("/", 1)[1]] += p.duration_s / 60
        print(f"{SOURCE_NAMES[source]}: {n_files} files, {len(pieces)} pieces, {len({p.group for p in pieces})} "
              f"groups, {sum(minutes.values()):.1f} min")
        print("  " + ", ".join(f"{c} {m:.0f}" for c, m in sorted(minutes.items())) + " (min)")
        return
    keyword_pieces = [p for p in pieces if p.keywords]
    counts = Counter(kw for p in pieces for kw in p.keywords)
    ignores = sum(label == IGNORE_LABEL for p in pieces for _, _, label in p.spans)
    speakers = {kw: {s for p in pieces if kw in p.keywords for s in p.speakers} for kw in KEYWORDS}
    all_speakers = {s for p in pieces for s in p.speakers}
    kw_min = sum(p.duration_s for p in keyword_pieces) / 60
    neg_min = sum(p.duration_s for p in pieces if not p.keywords) / 60
    print(f"{SOURCE_NAMES[source]}: {n_files} files, {len(pieces)} pieces, {len(all_speakers)} speakers")
    print("  keywords: " + ", ".join(f"{kw} {counts[kw]} ({len(speakers[kw])} spk)" for kw in KEYWORDS)
          + f", ignore spans {ignores}")
    print(f"  {kw_min:.1f} min of pieces with keywords, {neg_min:.1f} min without, "
          f"{kw_min + neg_min:.1f} min total")


# --- Stats -------------------------------------------------------------------

def print_counts(name, hours, speakers, kw_counts, kw_speakers, ignores):
    print(f"  {name:<18} {hours:7.1f} h {speakers:5d} spk  "
          + "  ".join(f"{kw} {kw_counts[kw]:4d} ({len(kw_speakers[kw]):3d} spk)" for kw in KEYWORDS)
          + f"  ignore {ignores:4d}")


def stats(data_dir, ami_channel, out_dir, noise_dir):
    ls_root = data_dir / "LibriSpeech"
    if ls_root.is_dir():
        print("LibriSpeech (transcripts; hours from SPEAKERS.TXT / file count, approximate):")
        for split, utts in ls_read_transcripts(ls_root).items():
            kw_counts, kw_speakers = Counter(), defaultdict(set)
            for u in utts:
                for label in u.labels:
                    if label in KEYWORDS:
                        kw_counts[label] += 1
                        kw_speakers[label].add(u.speaker)
            ignores = sum(l == IGNORE_LABEL for u in utts for l in u.labels)
            # ~12.7 s per utterance on average; avoids opening every file
            print_counts(split, len(utts) * 12.7 / 3600, len({u.speaker for u in utts}),
                         kw_counts, kw_speakers, ignores)
        align_root = data_dir / "librispeech_alignments"
        print(f"  alignments: {'found' if align_root.is_dir() else 'MISSING'} ({align_root})")
    else:
        print(f"LibriSpeech: not found ({ls_root})")

    manual_root = data_dir / "ami" / "ami_public_manual_1"
    if manual_root.is_dir():
        words_by_meeting = ami_read_words(manual_root)
        with_audio = {m for m in words_by_meeting if ami_audio_path(data_dir / "ami", m, ami_channel).exists()}
        print(f"AMI ({len(words_by_meeting)} meetings in the annotations, {len(with_audio)} with "
              f"{ami_channel} audio, {len(AMI_BAD_MEETINGS)} left out for bad timings):")
        for name, meetings in [("all annotated", set(words_by_meeting)), ("with audio", with_audio)]:
            meetings = meetings - AMI_BAD_MEETINGS
            kw_counts, kw_speakers = Counter(), defaultdict(set)
            ignores = 0
            hours = 0.0
            for m in meetings:
                words = words_by_meeting[m]
                hours += max((w.end_s or 0.0) for w in words) / 3600
                for w in words:
                    if w.label in KEYWORDS:
                        kw_counts[w.label] += 1
                        kw_speakers[w.label].add(w.speaker)
                    ignores += w.label == IGNORE_LABEL
            speakers = len({w.speaker for m in meetings for w in words_by_meeting[m]})
            print_counts(name, hours, speakers, kw_counts, kw_speakers, ignores)
    else:
        print(f"AMI: not found ({manual_root})")

    swc_root = data_dir / "swc" / "english"
    if swc_root.is_dir():
        print("SWC English (timed words only; untimed keywords become 'ignore'):")
        recordings = swc_recordings(swc_root)
        kw_counts, kw_speakers = Counter(), defaultdict(set)
        for rec in recordings:
            for w in rec.words:
                if w.label in KEYWORDS:
                    kw_counts[w.label] += 1
                    kw_speakers[w.label].add(w.speaker)
        ignores = sum(w.label == IGNORE_LABEL for rec in recordings for w in rec.words)
        print_counts("single-audio", sum(r.duration_s for r in recordings) / 3600,
                     len({r.group_key for r in recordings}), kw_counts, kw_speakers, ignores)
    else:
        print(f"SWC: not found ({swc_root})")

    path = out_dir / SPLITS_CSV
    if path.exists():
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        print(f"Split ({path}): groups and keyword occurrences per split")
        for name in [SOURCE_NAMES[s] for s in SPEECH_SOURCES]:
            for split in ["train", "val", "test"]:
                r = [x for x in rows if x["source"] == name and x["split"] == split]
                if r:
                    print(f"  {name:14s} {split:5s} {len(r):5d} groups  "
                          + "  ".join(f"{k} {sum(int(x[f'n_{k}']) for x in r):5d}" for k in KEYWORDS))
    noise_stats(noise_dir, out_dir)


def noise_stats(noise_dir, out_dir):
    """Usable (speech-free) hours of background noise per source, kind and split."""
    print(f"Background noise ({noise_dir}): files, hours, and usable hours (speech-free; see the docstring)")
    for source in NOISE_SOURCES:
        loaded = load_noise_source(source, noise_dir)
        if loaded is None:
            continue
        split_of = existing_splits(out_dir, source)
        table = defaultdict(lambda: [0, 0.0, 0.0])
        for f in loaded.items:
            t = table[(split_of.get(f.group, "(new)"), f.kind)]
            t[0] += 1
            t[1] += f.duration_s / 3600
            t[2] += f.usable_s() / 3600
        for (split, kind), (n, hours, usable) in sorted(table.items()):
            print(f"  {SOURCE_NAMES[source]:14s} {split:5s} {kind:21s} {n:5d} files {hours:6.1f} h  {usable:6.1f} h usable")


# --- Checks ------------------------------------------------------------------

BUILD_INFO_NAME = "build_info.json"
MAX_KEYWORD_S = 2.0       # a longer keyword span is probably misaligned too
QUIET_SPAN_DB = -10.0     # keyword span this much quieter than its surroundings: misaligned?
QUIET_CONTEXT_S = 1.5     # surroundings = this much audio either side of the span, within its piece
# More gain than this: probably near-silence boosted to speech level. The quietest AMI
# meetings (TS3004a/b, speech peaks ~-45 dBFS) legitimately need ~32 dB
HIGH_GAIN_DB = 35.0
MAX_LISTED = 10           # problems listed per check (the rest are only counted)
# Settings that change the output (written to build_info.json per source)
BUILD_SETTINGS = ["max_per_keyword", "max_per_speaker", "ls_negative_min", "ami_channel",
                  "ami_negative_min", "swc_negative_min", "level_dbfs", "seed"]
NOISE_BUILD_SETTINGS = ["noise_min", "seed"]


def build_settings(source, args):
    """What a source's build depends on, apart from this script (build_info.json). Noise also
    depends on noise_speech.py's detection settings and output, and the noise manifest's groups."""
    if source not in NOISE_SOURCES:
        return {k: getattr(args, k) for k in BUILD_SETTINGS}
    return {**{k: getattr(args, k) for k in NOISE_BUILD_SETTINGS},
            "speech_thresholds": noise_speech.YAMNET_THRESHOLDS,
            "speech_margin_s": noise_speech.SPEECH_MARGIN_S,
            "speech_scores_sha256": file_sha256(args.noise_dir / noise_speech.SCORES_NAME),
            "noise_manifest_sha256": file_sha256(args.noise_dir / "manifest.csv")}


def output_files(out_dir):
    """Every file a build writes, apart from build_info.json itself and the clip WAVs
    (those are covered by one combined hash, CLIPS_KEY)."""
    files = sorted(p for p in out_dir.iterdir() if OUTPUT_NAME.match(p.name))
    return files + [p for p in (out_dir / "manifest.csv", out_dir / "clips.csv") if p.exists()]


CLIPS_KEY = "clips/*.wav"


def output_hashes(out_dir):
    files = {p.name: file_sha256(p) for p in output_files(out_dir)}
    if (out_dir / "clips").is_dir():
        files[CLIPS_KEY] = clips_digest(out_dir)
    return files


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_build_info(out_dir):
    path = out_dir / BUILD_INFO_NAME
    return json.loads(path.read_text()) if path.exists() else {"sources": {}, "files": {}}


def update_build_info(out_dir, args, built):
    """Records the settings of the sources just built and the hash of every output file,
    and reports whether the rebuilt sources came out the same as the previous build."""
    old = read_build_info(out_dir)
    files = output_hashes(out_dir)
    settings = {source: build_settings(source, args) for source in built}
    script = file_sha256(pathlib.Path(__file__))

    print("\nCompared with the previous build:")
    for source in built:
        prefix = f"{source}_"
        before = {n: h for n, h in old["files"].items() if n.startswith(prefix)}
        after = {n: h for n, h in files.items() if n.startswith(prefix)}
        if not before:
            print(f"  {SOURCE_NAMES[source]}: no previous build")
            continue
        changed = sorted(n for n in before.keys() & after.keys() if before[n] != after[n])
        added, removed = sorted(after.keys() - before.keys()), sorted(before.keys() - after.keys())
        previous = old["sources"].get(source, {})
        if not (changed or added or removed):
            print(f"  {SOURCE_NAMES[source]}: identical ({len(after)} files)")
            continue
        print(f"  {SOURCE_NAMES[source]}: {len(changed)} changed, {len(added)} added, {len(removed)} removed files")
        if previous.get("settings") != settings[source]:
            print("    (settings changed)")
        elif previous.get("script_sha256") != script:
            print("    (make_public_eval.py changed since the previous build)")
        else:
            print("    WARNING: same settings and script as before, so either the datasets changed "
                  "or the build isn't reproducible")

    clips_same = old["files"].get(CLIPS_KEY) == files.get(CLIPS_KEY)
    print(f"  clips: {'identical' if clips_same else 'changed' if CLIPS_KEY in old['files'] else 'no previous build'}")

    sources = dict(old["sources"])
    for source in built:
        sources[source] = {"settings": settings[source], "script_sha256": script}
    info = {"sources": dict(sorted(sources.items())), "files": files}
    (out_dir / BUILD_INFO_NAME).write_text(json.dumps(info, indent=1) + "\n")


def rms_db(audio):
    return 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-12) if audio.size else -240.0


class CheckReport:
    """Collects check results: errors make the build invalid, warnings are suspicious spans."""

    def __init__(self):
        self.problems = defaultdict(list)    # (kind, check) -> [message]

    def add(self, kind, check, message):
        self.problems[(kind, check)].append(message)

    def count(self, kind):
        return sum(len(v) for (k, _), v in self.problems.items() if k == kind)

    def print(self):
        for (kind, check), messages in sorted(self.problems.items()):
            print(f"  {kind.upper()} {check}: {len(messages)}")
            for m in messages[:MAX_LISTED]:
                print(f"    {m}")
            if len(messages) > MAX_LISTED:
                print(f"    ... and {len(messages) - MAX_LISTED} more")


def check_build(out_dir, noise_dir=DEFAULT_NOISE_DIR):
    """Checks the files in out_dir. Prints a report; returns the number of errors."""
    print(f"\nChecking {out_dir}...")
    report = CheckReport()
    noise_names = {SOURCE_NAMES[s] for s in NOISE_SOURCES}
    manifest_path = out_dir / "manifest.csv"
    if not manifest_path.exists():
        print(f"  ERROR: no {manifest_path}")
        return 1
    with open(manifest_path, newline="") as f:
        manifest = list(csv.DictReader(f))

    # Files on disk must be the ones build_info.json recorded (catches a WAV left over from
    # another build, or one edited by hand)
    recorded = read_build_info(out_dir)["files"]
    if not recorded:
        report.add("error", "no build_info.json", f"rebuild to create {BUILD_INFO_NAME}")
    for name, digest in output_hashes(out_dir).items():
        if recorded and recorded.get(name) != digest:
            report.add("error", "file doesn't match build_info.json", name)

    pieces_by_wav = defaultdict(list)
    for r in manifest:
        pieces_by_wav[r["wav"]].append(r)
    truth_wavs = {p.name.replace("_truth.csv", ".wav") for p in out_dir.glob("*_truth.csv")
                  if OUTPUT_NAME.match(p.name)}

    per_source = defaultdict(Counter)
    levels = defaultdict(list)
    for wav_name in sorted(set(pieces_by_wav) | truth_wavs):
        source = wav_name.split("_")[0]
        wav, truth = out_dir / wav_name, out_dir / wav_name.replace(".wav", "_truth.csv")
        if not wav.exists() or not truth.exists() or wav_name not in pieces_by_wav:
            missing = [n for n, ok in [(wav.name, wav.exists()), (truth.name, truth.exists()),
                                       ("manifest rows", wav_name in pieces_by_wav)] if not ok]
            report.add("error", "missing file or manifest rows", f"{wav_name}: no {', '.join(missing)}")
            continue
        audio, rate = sf.read(str(wav), dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE_HZ or audio.shape[1] != 1:
            report.add("error", "wrong WAV format", f"{wav_name}: {rate} Hz, {audio.shape[1]} channels")
            continue
        audio = audio[:, 0]
        duration_s = len(audio) / rate
        per_source[source]["files"] += 1
        per_source[source]["seconds"] += duration_s

        # Pieces: inside the WAV and not overlapping each other
        pieces = sorted(pieces_by_wav[wav_name], key=lambda r: float(r["start_s"]))
        for prev, piece in zip([None] + pieces, pieces):
            start, end = float(piece["start_s"]), float(piece["end_s"])
            if start < 0 or end > duration_s + 1e-6:
                report.add("error", "piece outside its WAV", f"{wav_name} {piece['piece_id']} {start:.2f}-{end:.2f}s")
            if prev and start < float(prev["end_s"]) - 1e-6:
                report.add("error", "pieces overlap", f"{wav_name} {prev['piece_id']} / {piece['piece_id']}")
            # Level: active speech for the per-source spread; peaks must be below the limiter's ceiling
            x = audio[round(start * rate):round(end * rate)]
            levels[source].append(active_speech_db(x))
            peak_db = 20 * np.log10(np.abs(x).max() + 1e-12) if x.size else -240.0
            if peak_db > PEAK_LIMIT_DBFS + 0.01:
                report.add("error", f"peak above {PEAK_LIMIT_DBFS:g} dBFS (limiter failed?)",
                           f"{wav_name} {piece['piece_id']}: {peak_db:+.2f} dBFS")
            # Sounds of Home: a muted stretch is speech removal the labels missed
            if source == "soh" and muted_run_s(x, rate) >= MUTED_S:
                report.add("warning", f"Sounds of Home piece with a muted stretch (>= {MUTED_S:g} s below "
                           f"{MUTED_DBFS:g} dBFS)", f"{wav_name} {piece['piece_id']}")

        with open(truth, newline="") as f:
            spans = list(csv.DictReader(f))
        for span in spans:
            start, end, label = float(span["start_s"]), float(span["end_s"]), span["label"]
            where = f"{wav_name} @{start:.2f}s {label} ({span['file']})"
            if source in NOISE_SOURCES:
                report.add("error", "truth span in a noise file", where)
                continue
            if end <= start:
                report.add("error", "empty or reversed span", f"{where}: {start:.6f}-{end:.6f}s")
            # Inside the piece it says it came from
            if not any(p["piece_id"] == span["file"] and float(p["start_s"]) - 1e-6 <= start
                       and end <= float(p["end_s"]) + 1e-6 for p in pieces):
                report.add("error", "span outside its piece", where)
            if label == IGNORE_LABEL:
                per_source[source]["ignore"] += 1
                continue
            if label not in KEYWORDS:
                report.add("error", "unknown label", where)
                continue
            per_source[source][label] += 1
            # A keyword near a piece edge needs that edge to be the source's, or padding
            piece = next((p for p in pieces if p["piece_id"] == span["file"]), None)
            centred_start = (start + end - CLIP_S) / 2
            if piece and clip_length_ok(start, end) and (
                    (centred_start < float(piece["start_s"]) and not at_source_start(piece))
                    or (centred_start + CLIP_S > float(piece["end_s"]) and not at_source_end(piece))):
                report.add("warning", "keyword too close to a piece edge inside the source (needs padding)", where)
            # Plausible keyword length, and audible: a misaligned span can land on a pause
            if not MIN_KEYWORD_S <= end - start <= MAX_KEYWORD_S:
                report.add("warning", f"keyword span not {MIN_KEYWORD_S}-{MAX_KEYWORD_S} s long",
                           f"{where}: {end - start:.2f}s")
            span_db = rms_db(audio[round(start * rate):round(end * rate)])
            context_start, context_end = start - QUIET_CONTEXT_S, end + QUIET_CONTEXT_S
            if piece:  # its own piece only: the next piece is another speaker at another level
                context_start = max(context_start, float(piece["start_s"]))
                context_end = min(context_end, float(piece["end_s"]))
            context = audio[max(0, round(context_start * rate)):round(context_end * rate)]
            relative_db = span_db - rms_db(context)
            if relative_db < QUIET_SPAN_DB:
                report.add("warning", f"keyword span >{-QUIET_SPAN_DB:g} dB quieter than the "
                           f"{QUIET_CONTEXT_S:g} s around it", f"{where}: {relative_db:+.1f} dB")

    # Noise pieces must be clear of the speech noise_speech.py found (and its margin)
    noise_rows = [r for r in manifest if r["source"] in noise_names]
    if noise_rows:
        try:
            scores = speech_scores(noise_dir)
        except FileNotFoundError as e:
            scores = None
            report.add("warning", "noise not checked for speech", str(e))
        for r in noise_rows if scores is not None else []:
            try:
                speech = noise_speech.speech_spans(scores, r["src_file"], float(r["src_duration_s"]))
            except KeyError:
                report.add("error", "noise piece's file not in the speech scores", f"{r['wav']} {r['src_file']}")
                continue
            # the manifest's source times have 3 decimals
            if any(s < float(r["src_end_s"]) - 1e-3 and float(r["src_start_s"]) + 1e-3 < e for s, e in speech):
                report.add("error", "noise piece overlaps speech", f"{r['wav']} {r['piece_id']} ({r['src_file']})")

    # Split membership: every speaker / meeting / noise group must belong to this split in splits.csv
    split_of = read_splits(out_dir.parent)
    if out_dir.name in SPLITS and split_of is not None:
        for r in manifest:
            members = ([r["group"]] if r["source"] == "ami" or r["source"] in noise_names
                       else [m for m in r["speakers"].split() if m != "?"])
            for member in members:
                got = split_of.get((r["source"], member))
                if got != out_dir.name:
                    report.add("error", f"speaker not in the {out_dir.name} split of {SPLITS_CSV}",
                               f"{r['wav']} {r['piece_id']}: {r['source']} {member} is in '{got}'")
    elif out_dir.name in SPLITS:
        report.add("error", f"no {SPLITS_CSV}", f"{out_dir.parent / SPLITS_CSV} is missing")

    # Clips: 1 s and inside their piece; keyword clips contain their word and are centred on it
    # or shifted to a piece edge that's also a source edge; 'other' clips avoid keyword / ignore spans
    clip_counts = Counter()
    clip_shifted = 0
    if (out_dir / "clips.csv").exists():
        pieces_by_id = {(r["wav"], r["piece_id"]): r for r in manifest}
        tol = 2 / SAMPLE_RATE_HZ
        spans_by_wav = {}
        with open(out_dir / "clips.csv", newline="") as f:
            for c in csv.DictReader(f):
                clip_counts[c["label"]] += 1
                path = out_dir / c["file"]
                if not path.exists():
                    report.add("error", "clip missing", c["file"])
                    continue
                info = sf.info(str(path))
                if info.frames != round(CLIP_S * SAMPLE_RATE_HZ) or info.samplerate != SAMPLE_RATE_HZ or info.channels != 1:
                    report.add("error", "clip not 1 s of 16 kHz mono", f"{c['file']}: {info.frames} frames")
                if c["wav"] not in spans_by_wav:
                    with open(out_dir / c["wav"].replace(".wav", "_truth.csv"), newline="") as tf:
                        spans_by_wav[c["wav"]] = [(float(t["start_s"]), float(t["end_s"]), t["label"], t["file"])
                                                  for t in csv.DictReader(tf)]
                start, end = float(c["start_s"]), float(c["end_s"])
                piece = pieces_by_id.get((c["wav"], c["piece_id"]))
                if piece is None:
                    report.add("error", "clip's piece isn't in the manifest", c["file"])
                    continue
                piece_start, piece_end = float(piece["start_s"]), float(piece["end_s"])
                if start < piece_start - tol or end > piece_end + tol:
                    report.add("error", "clip outside its piece", c["file"])
                if c["label"] in KEYWORDS:
                    span_start, span_end = float(c["span_start_s"]), float(c["span_end_s"])
                    if not (start <= span_start and span_end <= end):
                        report.add("error", "keyword clip doesn't contain its word", c["file"])
                    if abs((start + end) - (span_start + span_end)) / 2 > tol:
                        clip_shifted += 1
                        if not ((abs(start - piece_start) <= tol and at_source_start(piece))
                                or (abs(end - piece_end) <= tol and at_source_end(piece))):
                            report.add("error", "keyword clip neither centred nor at a piece edge "
                                       "that's a source edge", c["file"])
                elif c["label"] == OTHER_LABEL:
                    if overlaps_any(spans_by_wav[c["wav"]], start, end):
                        report.add("error", "'other' clip overlaps a keyword / ignore span", c["file"])
                    if piece["source"] in noise_names:
                        report.add("error", "'other' clip from a noise piece", c["file"])
                elif c["label"] == BACKGROUND_LABEL:
                    if piece["source"] not in noise_names:
                        report.add("error", "background clip outside a noise piece", c["file"])
                else:
                    report.add("error", "unknown clip label", f"{c['file']}: {c['label']}")

    # Gain: large values mean a quiet source; very large ones near-silence boosted to speech level
    gains, limits = defaultdict(list), defaultdict(list)
    for r in manifest:
        gains[r["wav"].split("_")[0]].append(float(r["gain_db"]))
        limits[r["wav"].split("_")[0]].append(float(r.get("limit_db") or 0))
        if float(r["gain_db"]) > HIGH_GAIN_DB:
            report.add("warning", f"gain above {HIGH_GAIN_DB:g} dB", f"{r['wav']} {r['piece_id']}: {r['gain_db']} dB")
    # Each speech source's median speech level should be the target it was built with
    built_with = read_build_info(out_dir)["sources"]
    for source, lv in levels.items():
        target = built_with.get(source, {}).get("settings", {}).get("level_dbfs")
        if target is not None and abs(np.median(lv) - target) > 1.0:
            report.add("warning", "source median speech level more than 1 dB from target",
                       f"{SOURCE_NAMES.get(source, source)}: {np.median(lv):.1f} dBFS, built for {target:g}")
    # As much noise as was asked for (Sounds of Home fills up to it)
    noise_min = built_with.get("soh", {}).get("settings", {}).get("noise_min")
    noise_s = sum(per_source[s]["seconds"] for s in NOISE_SOURCES if s in per_source)
    piece_s = sum(float(r["end_s"]) - float(r["start_s"]) for r in noise_rows)
    if noise_min is not None and piece_s < noise_min * 60 - NOISE_EXCERPT_S:
        report.add("warning", "less noise than requested", f"{piece_s / 60:.0f} of {noise_min:g} min "
                   f"(download more Sounds of Home hours for this split's homes)")

    width = max(len(SOURCE_NAMES[s]) for s in SOURCES)
    print(f"  {'source':<{width}} {'files':>5} {'min':>6}  " + "  ".join(f"{k:>8}" for k in KEYWORDS + [IGNORE_LABEL])
          + "   gain dB: min / median / max")
    for source in sorted(per_source, key=lambda s: SOURCES.index(s) if s in SOURCES else len(SOURCES)):
        c, g = per_source[source], np.array(gains[source])
        print(f"  {SOURCE_NAMES.get(source, source):<{width}} {c['files']:5d} {c['seconds'] / 60:6.1f}  "
              + "  ".join(f"{c[k]:8d}" for k in KEYWORDS + [IGNORE_LABEL])
              + (f"   {g.min():+.1f} / {np.median(g):+.1f} / {g.max():+.1f}" if g.size else ""))
    speech_s = sum(per_source[s]["seconds"] for s in SPEECH_SOURCES if s in per_source)
    print(f"  speech {speech_s / 3600:.1f} h, noise {noise_s / 3600:.1f} h (WAV time, gaps included)")
    print(f"  {'source':<{width}} active level dBFS 5% / 50% / 95% (spread)   pieces limited, largest reduction")
    for source in sorted(levels, key=lambda s: SOURCES.index(s) if s in SOURCES else len(SOURCES)):
        p5, p50, p95 = np.percentile(levels[source], [5, 50, 95])
        lim = np.array(limits[source])
        print(f"  {SOURCE_NAMES.get(source, source):<{width}} {p5:7.1f} / {p50:6.1f} / {p95:6.1f}  ({p95 - p5:4.1f} dB)"
              f"{'':9s}{int(np.sum(lim > 0)):4d} of {len(lim)}, {lim.max() if lim.size else 0:.1f} dB")
    if clip_counts:
        print("  clips: " + ", ".join(f"{k} {clip_counts[k]}" for k in CLIP_LABELS)
              + f" ({clip_shifted} keyword clips shifted to a source edge)")
    report.print()
    errors, warnings = report.count("error"), report.count("warning")
    print(f"  {errors} errors, {warnings} warnings" + ("" if errors else " (OK)"))
    return errors


# --- Main --------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build eval WAVs + truth CSVs from LibriSpeech, AMI and Spoken Wikipedia word alignments, "
                    "plus speech-free background noise (MUSAN, DEMAND, Sounds of Home)."
    )
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR,
                        help=f"public datasets dir (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--noise-dir", type=pathlib.Path, default=DEFAULT_NOISE_DIR,
                        help=f"background noise dir, from download_noise.py + noise_speech.py (default: {DEFAULT_NOISE_DIR})")
    parser.add_argument("-o", "--out", type=pathlib.Path, default=DEFAULT_OUT_DIR,
                        help=f"output dir (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--stats", action="store_true", help="only print keyword counts per split / source")
    parser.add_argument("--check", action="store_true",
                        help="only check the existing build in --out (a build always checks itself too)")
    parser.add_argument("--clips-only", action="store_true",
                        help="only re-cut the clips from the existing WAVs (the recordings aren't touched), "
                             "after a change to the clip settings")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=SOURCES,
                        help=f"sources to build (default: {' '.join(SOURCES)})")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS,
                        help=f"splits to build / check (default: {' '.join(SPLITS)})")
    parser.add_argument("--max-per-keyword", type=int,
                        help="limit on occurrences of a keyword per source and split (default: no limit)")
    parser.add_argument("--max-per-speaker", type=int,
                        help="limit on occurrences of a keyword per speaker (default: no limit)")
    parser.add_argument("--ls-negative-min", type=float, default=950,
                        help="minutes of LibriSpeech utterances without keywords (default: 950)")
    parser.add_argument("--ami-channel", default=DEFAULT_AMI_CHANNEL,
                        help=f"AMI recording to use, e.g. Mix-Headset, Mix-Lapel, Array1-01 "
                             f"(default: {DEFAULT_AMI_CHANNEL})")
    parser.add_argument("--ami-negative-min", type=float, default=380,
                        help="minutes of AMI excerpts without keywords (default: 380)")
    parser.add_argument("--swc-negative-min", type=float, default=570,
                        help="minutes of SWC excerpts without keywords (default: 570)")
    parser.add_argument("--noise-min", type=float, default=DEFAULT_NOISE_MIN,
                        help=f"minutes of background noise per split: every usable MUSAN and DEMAND file of the "
                             f"split, then Sounds of Home excerpts up to this (default: {DEFAULT_NOISE_MIN})")
    parser.add_argument("--level-dbfs", type=float, default=DEFAULT_LEVEL_DBFS,
                        help=f"median active-speech level of each source (default: {DEFAULT_LEVEL_DBFS:g}, "
                             f"the board recordings' level)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.noise_dir = args.noise_dir.resolve()
    if args.stats:
        stats(args.data_dir, args.ami_channel, args.out, args.noise_dir)
        return
    if args.check:
        dirs = [args.out / split for split in args.splits if (args.out / split).is_dir()]
        if not dirs:
            sys.exit(f"no built splits in {args.out}")
        sys.exit(1 if sum(check_build(d, args.noise_dir) for d in dirs) else 0)

    if args.clips_only:
        dirs = [args.out / split for split in args.splits if (args.out / split).is_dir()]
        if not dirs:
            sys.exit(f"no built splits in {args.out}")
        sys.exit(1 if sum(recut_clips(d, args) for d in dirs) else 0)

    args.out.mkdir(parents=True, exist_ok=True)
    data, assignments = {}, {}
    for source in [s for s in args.sources if s in SPEECH_SOURCES]:
        print(f"Loading {SOURCE_NAMES[source]}...")
        loaded = load_source(source, args.data_dir, args)
        if loaded is None:
            continue
        split_of, sizes = assign_splits(loaded, args.seed)
        data[source] = (loaded, split_of)
        assignments[source] = (loaded, split_of, sizes)
    # Noise: every noise source is loaded and split when any is built (its split never changes,
    # see assign_splits), as Sounds of Home fills up to --noise-min after MUSAN and DEMAND
    noise = {}
    if any(s in NOISE_SOURCES for s in args.sources):
        for source in NOISE_SOURCES:
            print(f"Loading {SOURCE_NAMES[source]}...")
            loaded = load_noise_source(source, args.noise_dir)
            if loaded is None:
                continue
            split_of, sizes = assign_splits(loaded, args.seed, existing_splits(args.out, source))
            noise[source] = (loaded, split_of)
            assignments[source] = (loaded, split_of, sizes)
            if source in args.sources:
                data[source] = noise[source]
    if not data:
        sys.exit("no source data found")
    write_splits(args.out, assignments)

    builders = {"ls": build_librispeech, "ami": build_ami, "swc": build_swc}
    errors = 0
    for split in args.splits:
        out_dir = args.out / split
        out_dir.mkdir(exist_ok=True)
        manifest, built = [], []
        for source, (loaded, split_of) in data.items():
            items = [i for i in loaded.items if split_of[loaded.group[id(i)]] == split]
            groups = {loaded.group[id(i)] for i in items}
            kind = "utterances" if source == "ls" else "files" if source in NOISE_SOURCES else "recordings"
            print(f"\n[{split}] Selecting {SOURCE_NAMES[source]} from {len(items)} {kind} of {len(groups)} groups...")
            # One generator per source and split, so building some gives the same picks as building all
            rng = np.random.default_rng([args.seed, SOURCES.index(source), SPLITS.index(split)])
            if source == "soh":
                target_s = args.noise_min * 60 - fixed_noise_s(noise, split)
                pieces = build_noise_excerpts(source, items, target_s, rng)
                warn_if_short(source, len(pieces) * NOISE_EXCERPT_S, target_s / 60,
                              " (Sounds of Home fills up to --noise-min: download more hours of this split's homes)")
            elif source in EXCERPT_NOISE:
                pieces = build_noise_excerpts(source, items, math.inf, rng)
            elif source in NOISE_SOURCES:
                pieces = build_musan_pieces(source, items)
            else:
                pieces = builders[source](items, args.data_dir, args, rng)
            if not pieces:
                continue
            # Remove this source's files from an earlier build, so none are left over
            for old in out_dir.iterdir():
                if OUTPUT_NAME.match(old.name) and old.name.startswith(f"{source}_"):
                    old.unlink()
            rows, n_files = assemble(source, pieces, out_dir, args, rng)
            manifest += rows
            built.append(source)
            summarise(source, pieces, n_files)
        if not manifest:
            print(f"[{split}] nothing built")
            continue
        write_manifest(out_dir, manifest, {SOURCE_NAMES[s] for s in built})
        write_clips(out_dir, np.random.default_rng([args.seed, CLIPS_RNG_KEY, SPLITS.index(split)]))
        update_build_info(out_dir, args, built)
        errors += check_build(out_dir, args.noise_dir)
    sys.exit(1 if errors else 0)


def recut_clips(out_dir, args):
    """Cuts a built split's clips again (write_clips) without touching its recordings, which must still
    match build_info.json. Returns the number of check errors."""
    print(f"\n[{out_dir.name}] Re-cutting the clips...")
    recorded = read_build_info(out_dir)["files"]
    files = [p for p in output_files(out_dir) if p.name != "clips.csv"]
    changed = [p.name for p in files if recorded.get(p.name) != file_sha256(p)]
    if changed or not files:
        sys.exit(f"{out_dir}: {len(changed)} files don't match {BUILD_INFO_NAME} (e.g. {changed[:3]}): rebuild instead")
    write_clips(out_dir, np.random.default_rng([args.seed, CLIPS_RNG_KEY, SPLITS.index(out_dir.name)]))
    update_build_info(out_dir, args, [])
    return check_build(out_dir, args.noise_dir)


def write_manifest(out_dir, rows, built_sources):
    """Writes manifest.csv, keeping an earlier build's rows for sources not rebuilt this time."""
    path = out_dir / "manifest.csv"
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["source"] not in built_sources] + rows
    with open(path, "w", newline="") as f:
        # Rows kept from a build before a column was added get it empty
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} pieces)")


if __name__ == "__main__":
    main()
