"""
Build a keyword-spotting eval set from public speech datasets, in the same
<stem>.wav + <stem>_truth.csv format as eval_recordings/, so many more
speakers and hours of non-keyword speech can be scored than the board
recordings give.

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
  LibriSpeech  whole utterances (~13 s). Keyword utterances are picked
               round-robin over speakers, at most --max-per-speaker per speaker
               and --max-per-keyword per keyword. Negative utterances have no
               keyword and no ignore word, at most 3 per speaker, up to
               --ls-negative-min minutes.
  AMI, SWC     20 s excerpts. Each keyword gets an excerpt that contains it at a
               random offset (overlapping excerpts are merged), with the same caps.
               Negative excerpts contain no keyword or ignore word, up to
               --ami-negative-min / --swc-negative-min minutes, spread round-robin
               over meetings / readers. SWC negatives must be at least
               SWC_MIN_COVERAGE covered by aligned words. SWC audio is resampled to
               16 kHz and downmixed; articles below 16 kHz, split over several audio
               files (no timings) or that libsndfile can't open are skipped.

Each source's pieces are shuffled together and packed into ~5 minute files
(ls_<nn>.wav, ami_<nn>.wav, swc_<nn>.wav), separated by short gaps of very quiet
noise. The same arguments always give the same files.

Level: each piece has its DC removed, then gets one static gain per source (AMI: per
meeting, since its meetings were recorded ~30 dB apart), chosen so the source's median
active-speech level (loudest 30% of 20 ms frames) is --level-dbfs, the board's level.
The natural loudness spread inside a source is kept. A peak limiter then keeps every
sample at or below -1 dBFS by lowering the gain only around the peaks.

Outputs in --out (default eval_recordings_public/):
  <stem>.wav          16 kHz, 16-bit mono (gitignored: rebuild instead)
  <stem>_truth.csv    start_s, end_s, label, file   (file = source utterance / excerpt id)
  manifest.csv        one row per piece: where it is in which WAV, and where it came from
  build_info.json     settings of each source's build and the SHA-256 of every output file
  speakers.txt        every speaker used. Never train on these speakers (AMI: whole
                      meeting series, since the same people are in each meeting;
                      SWC: Wikipedia user names of the readers)

After a build (or with --check, on an existing build) the output is checked:
  errors    a WAV / truth file / manifest row missing, not 16 kHz mono, files that
            don't match build_info.json, pieces outside their WAV or overlapping,
            a peak above PEAK_LIMIT_DBFS, empty spans, spans outside the piece
            they came from, unknown labels  -> exit code 1
  warnings  keyword spans shorter than MIN_KEYWORD_S or longer than MAX_KEYWORD_S,
            or QUIET_SPAN_DB quieter than the audio around them (misaligned?), and
            pieces with more than HIGH_GAIN_DB of gain (boosted near-silence?), and a
            source median speech level more than 1 dB from the target
It also prints each source's speech-level spread and how many pieces the limiter touched.
A build also reports, per source, whether it came out identical to the previous
build (build_info.json), and warns if it didn't with the same settings.

Usage (from HostSW/):
    python EvalScripts/make_public_eval.py --stats
    python EvalScripts/make_public_eval.py
    python EvalScripts/make_public_eval.py --check
    python EvalScripts/make_public_eval.py --sources ls --ls-negative-min 60 --seed 1
"""

import argparse
import csv
import hashlib
import json
import math
import pathlib
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile as sf

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
HOSTSW_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = HOSTSW_DIR / "dataset_public"
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

SOURCES = ["ls", "ami", "swc"]
SOURCE_NAMES = {"ls": "librispeech", "ami": "ami", "swc": "swc"}

LS_NEGATIVES_PER_SPEAKER = 3
EXCERPT_S = 20.0         # AMI / SWC excerpt length
EXCERPT_EDGE_S = 2.0     # keyword at least this far inside its excerpt
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

OUTPUT_NAME = re.compile(r"^(ls|ami|swc)_\d+(\.wav|_truth\.csv)$")


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
    """Picks keyword pieces, rarest keyword first, round-robin over speakers.
    A piece's keywords all count towards the caps."""
    per_keyword = Counter()
    per_speaker = Counter()
    chosen = []
    taken = set()
    totals = Counter(kw for p in candidates for kw in set(p.keywords))
    for kw in sorted(KEYWORDS, key=lambda k: totals[k]):
        pool = [p for p in candidates if kw in p.keywords and p.piece_id not in taken]
        for piece in round_robin(pool, lambda p: p.speakers[0], rng):
            if per_keyword[kw] >= max_per_keyword:
                break
            if per_speaker[(piece.speakers[0], kw)] >= max_per_speaker:
                continue
            chosen.append(piece)
            taken.add(piece.piece_id)
            for k in piece.keywords:
                per_keyword[k] += 1
                per_speaker[(piece.speakers[0], k)] += 1
    return chosen


def build_librispeech(data_dir, args, rng):
    ls_root = data_dir / "LibriSpeech"
    align_root = data_dir / "librispeech_alignments"
    splits = ls_read_transcripts(ls_root)
    utts = [u for split in splits.values() for u in split]

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


def longest_zero_run(audio):
    """Length in samples of the longest run of exact zeros."""
    edges = np.diff(np.concatenate([[0], (audio == 0).astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int((ends - starts).max()) if starts.size else 0


def is_dead_audio(audio_path, start_s, end_s):
    """True for a stretch with a dropout (a long run of digital zeros) or near-silence."""
    rate = sf.info(str(audio_path)).samplerate
    audio, _ = sf.read(str(audio_path), start=round(start_s * rate), stop=round(end_s * rate),
                       dtype="float32", always_2d=True)
    if audio.size == 0:
        return True
    audio = audio.mean(axis=1)
    rms_dbfs = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-12)
    return longest_zero_run(audio) >= DEAD_ZERO_RUN_S * rate or rms_dbfs < DEAD_RMS_DBFS


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

    # Negative excerpts: round-robin over groups (meetings / readers), a random recording
    # of the group and a random position in it. No keyword or ignore word, no overlap with
    # a keyword excerpt, and enough of it covered by aligned words.
    by_group = defaultdict(list)
    for rec in recordings:
        if rec.duration_s > EXCERPT_S:
            by_group[rec.group_key].append(rec)
    groups = sorted(by_group)
    taken = {rec.rec_id: [tuple(r) for r in keyword_ranges[rec.rec_id]] for rec in recordings}
    negatives_s = 0.0
    tries = 0
    while groups and negatives_s < negative_min * 60 and tries < 50 * len(recordings):
        for group in [groups[i] for i in rng.permutation(len(groups))]:
            tries += 1
            recs = by_group[group]
            rec = recs[rng.integers(len(recs))]
            start = rng.uniform(0, rec.duration_s - EXCERPT_S)
            end = start + EXCERPT_S
            if any(s < end and start < e for s, e in taken[rec.rec_id]):
                continue
            if overlap_s(rec.blocked_starts, rec.blocked_ends, start, end) > 0:
                continue
            if overlap_s(rec.word_starts, rec.word_ends, start, end) < rec.min_coverage * EXCERPT_S:
                continue
            if is_dead_audio(rec.audio_path, start, end):
                continue
            taken[rec.rec_id].append((start, end))
            pieces.append(excerpt(source, rec, start, end))
            negatives_s += EXCERPT_S
            if negatives_s >= negative_min * 60:
                break
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


def build_ami(data_dir, args, rng):
    recordings = ami_recordings(data_dir / "ami", args.ami_channel)
    if not recordings:
        print(f"  AMI: no meetings with {args.ami_channel} audio found, skipping (run download_ami.py)")
        return []
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


def build_swc(data_dir, args, rng):
    swc_root = data_dir / "swc" / "english"
    if not swc_root.is_dir():
        print(f"  SWC: {swc_root} not found, skipping")
        return []
    return build_excerpts("swc", swc_recordings(swc_root), args.swc_negative_min, args, rng)


# --- Assembly ----------------------------------------------------------------

def load_piece(piece):
    """The piece's audio as 16 kHz mono float32 (resampled / downmixed if needed)."""
    rate = sf.info(str(piece.audio_path)).samplerate
    audio, _ = sf.read(str(piece.audio_path), start=round(piece.start_s * rate),
                       stop=round(piece.end_s * rate), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE_HZ:
        g = math.gcd(rate, SAMPLE_RATE_HZ)
        audio = scipy.signal.resample_poly(audio, SAMPLE_RATE_HZ // g, rate // g).astype(np.float32)
    return audio


def active_speech_db(audio):
    """Speech level ignoring pauses: mean power of the loudest ACTIVE_TOP_SHARE of
    ACTIVE_FRAME_S frames, with DC removed per frame."""
    n = round(ACTIVE_FRAME_S * SAMPLE_RATE_HZ)
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    if frames.size == 0:
        return -240.0
    power = np.sort(((frames - frames.mean(axis=1, keepdims=True)) ** 2).mean(axis=1))
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
    source's median piece active-speech level is level_dbfs."""
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
                "src_file": str(piece.audio_path.relative_to(args.data_dir)),
                "src_start_s": f"{piece.start_s:.3f}", "src_end_s": f"{piece.end_s:.3f}",
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


def summarise(source, pieces, n_files):
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


def stats(data_dir, ami_channel):
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


# --- Checks ------------------------------------------------------------------

BUILD_INFO_NAME = "build_info.json"
MAX_KEYWORD_S = 2.0       # a longer keyword span is probably misaligned too
QUIET_SPAN_DB = -10.0     # keyword span this much quieter than its surroundings: misaligned?
QUIET_CONTEXT_S = 1.5     # surroundings = this much audio either side of the span
# More gain than this: probably near-silence boosted to speech level. The quietest AMI
# meetings (TS3004a/b, speech peaks ~-45 dBFS) legitimately need ~32 dB
HIGH_GAIN_DB = 35.0
MAX_LISTED = 10           # problems listed per check (the rest are only counted)
# Settings that change the output (written to build_info.json per source)
BUILD_SETTINGS = ["max_per_keyword", "max_per_speaker", "ls_negative_min", "ami_channel",
                  "ami_negative_min", "swc_negative_min", "level_dbfs", "seed"]


def output_files(out_dir):
    """Every file a build writes, apart from build_info.json itself."""
    files = sorted(p for p in out_dir.iterdir() if OUTPUT_NAME.match(p.name))
    return files + [p for p in (out_dir / "manifest.csv", out_dir / "speakers.txt") if p.exists()]


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
    files = {p.name: file_sha256(p) for p in output_files(out_dir)}
    settings = {k: getattr(args, k) for k in BUILD_SETTINGS}
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
        if previous.get("settings") != settings:
            print("    (settings changed)")
        elif previous.get("script_sha256") != script:
            print("    (make_public_eval.py changed since the previous build)")
        else:
            print("    WARNING: same settings and script as before, so either the datasets changed "
                  "or the build isn't reproducible")

    sources = dict(old["sources"])
    for source in built:
        sources[source] = {"settings": settings, "script_sha256": script}
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


def check_build(out_dir):
    """Checks the files in out_dir. Prints a report; returns the number of errors."""
    print(f"\nChecking {out_dir}...")
    report = CheckReport()
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
    for path in output_files(out_dir):
        if recorded and recorded.get(path.name) != file_sha256(path):
            report.add("error", "file doesn't match build_info.json", path.name)

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

        with open(truth, newline="") as f:
            spans = list(csv.DictReader(f))
        for span in spans:
            start, end, label = float(span["start_s"]), float(span["end_s"]), span["label"]
            where = f"{wav_name} @{start:.2f}s {label} ({span['file']})"
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
            # Plausible keyword length, and audible: a misaligned span can land on a pause
            if not MIN_KEYWORD_S <= end - start <= MAX_KEYWORD_S:
                report.add("warning", f"keyword span not {MIN_KEYWORD_S}-{MAX_KEYWORD_S} s long",
                           f"{where}: {end - start:.2f}s")
            span_db = rms_db(audio[round(start * rate):round(end * rate)])
            context = audio[max(0, round((start - QUIET_CONTEXT_S) * rate)):round((end + QUIET_CONTEXT_S) * rate)]
            relative_db = span_db - rms_db(context)
            if relative_db < QUIET_SPAN_DB:
                report.add("warning", f"keyword span >{-QUIET_SPAN_DB:g} dB quieter than the "
                           f"{QUIET_CONTEXT_S:g} s around it", f"{where}: {relative_db:+.1f} dB")

    # Gain: large values mean a quiet source; very large ones near-silence boosted to speech level
    gains, limits = defaultdict(list), defaultdict(list)
    for r in manifest:
        gains[r["wav"].split("_")[0]].append(float(r["gain_db"]))
        limits[r["wav"].split("_")[0]].append(float(r.get("limit_db") or 0))
        if float(r["gain_db"]) > HIGH_GAIN_DB:
            report.add("warning", f"gain above {HIGH_GAIN_DB:g} dB", f"{r['wav']} {r['piece_id']}: {r['gain_db']} dB")
    # Each source's median speech level should be the target it was built with
    built_with = read_build_info(out_dir)["sources"]
    for source, lv in levels.items():
        target = built_with.get(source, {}).get("settings", {}).get("level_dbfs")
        if target is not None and abs(np.median(lv) - target) > 1.0:
            report.add("warning", "source median speech level more than 1 dB from target",
                       f"{SOURCE_NAMES.get(source, source)}: {np.median(lv):.1f} dBFS, built for {target:g}")

    print(f"  {'source':<12} {'files':>5} {'min':>6}  " + "  ".join(f"{k:>8}" for k in KEYWORDS + [IGNORE_LABEL])
          + "   gain dB: min / median / max")
    for source in sorted(per_source):
        c, g = per_source[source], np.array(gains[source])
        print(f"  {SOURCE_NAMES.get(source, source):<12} {c['files']:5d} {c['seconds'] / 60:6.1f}  "
              + "  ".join(f"{c[k]:8d}" for k in KEYWORDS + [IGNORE_LABEL])
              + (f"   {g.min():+.1f} / {np.median(g):+.1f} / {g.max():+.1f}" if g.size else ""))
    print(f"  {'source':<12} active speech dBFS 5% / 50% / 95% (spread)   pieces limited, largest reduction")
    for source in sorted(levels):
        p5, p50, p95 = np.percentile(levels[source], [5, 50, 95])
        lim = np.array(limits[source])
        print(f"  {SOURCE_NAMES.get(source, source):<12} {p5:7.1f} / {p50:6.1f} / {p95:6.1f}  ({p95 - p5:4.1f} dB)"
              f"{'':9s}{int(np.sum(lim > 0)):4d} of {len(lim)}, {lim.max() if lim.size else 0:.1f} dB")
    report.print()
    errors, warnings = report.count("error"), report.count("warning")
    print(f"  {errors} errors, {warnings} warnings" + ("" if errors else " (OK)"))
    return errors


# --- Main --------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build eval WAVs + truth CSVs from LibriSpeech, AMI and Spoken Wikipedia word alignments."
    )
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR,
                        help=f"public datasets dir (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("-o", "--out", type=pathlib.Path, default=DEFAULT_OUT_DIR,
                        help=f"output dir (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--stats", action="store_true", help="only print keyword counts per split / source")
    parser.add_argument("--check", action="store_true",
                        help="only check the existing build in --out (a build always checks itself too)")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=SOURCES,
                        help="sources to build (default: ls ami swc)")
    parser.add_argument("--max-per-keyword", type=int, default=150,
                        help="keyword occurrences per keyword per source (default: 150)")
    parser.add_argument("--max-per-speaker", type=int, default=2,
                        help="occurrences of one keyword per speaker (default: 2)")
    parser.add_argument("--ls-negative-min", type=float, default=120,
                        help="minutes of LibriSpeech utterances without keywords (default: 120)")
    parser.add_argument("--ami-channel", default=DEFAULT_AMI_CHANNEL,
                        help=f"AMI recording to use, e.g. Mix-Headset, Mix-Lapel, Array1-01 "
                             f"(default: {DEFAULT_AMI_CHANNEL})")
    parser.add_argument("--ami-negative-min", type=float, default=60,
                        help="minutes of AMI excerpts without keywords (default: 60)")
    parser.add_argument("--swc-negative-min", type=float, default=60,
                        help="minutes of SWC excerpts without keywords (default: 60)")
    parser.add_argument("--level-dbfs", type=float, default=DEFAULT_LEVEL_DBFS,
                        help=f"median active-speech level of each source (default: {DEFAULT_LEVEL_DBFS:g}, "
                             f"the board recordings' level)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    if args.stats:
        stats(args.data_dir, args.ami_channel)
        return
    if args.check:
        sys.exit(1 if check_build(args.out) else 0)

    args.out.mkdir(parents=True, exist_ok=True)
    builders = {"ls": build_librispeech, "ami": build_ami, "swc": build_swc}
    manifest = []
    built = set()
    for source in args.sources:
        print(f"Selecting {SOURCE_NAMES[source]}...")
        # One generator per source, so building some sources gives the same picks as building all
        rng = np.random.default_rng([args.seed, SOURCES.index(source)])
        pieces = builders[source](args.data_dir, args, rng)
        if not pieces:
            continue
        # Remove this source's files from an earlier build, so none are left over
        for old in args.out.iterdir():
            if OUTPUT_NAME.match(old.name) and old.name.startswith(f"{source}_"):
                old.unlink()
        rows, n_files = assemble(source, pieces, args.out, args, rng)
        manifest += rows
        built.add(SOURCE_NAMES[source])
        summarise(source, pieces, n_files)

    if not manifest:
        sys.exit("nothing built")
    write_manifest(args.out, manifest, built)
    update_build_info(args.out, args, [s for s in args.sources if SOURCE_NAMES[s] in built])
    if check_build(args.out):
        sys.exit(1)


def write_manifest(out_dir, rows, built_sources):
    """Writes manifest.csv (keeping an earlier build's rows for sources not rebuilt this
    time) and speakers.txt from it."""
    path = out_dir / "manifest.csv"
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["source"] not in built_sources] + rows
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[-1]))
        writer.writeheader()
        writer.writerows(rows)

    speakers = set()
    for r in rows:
        speakers |= {f"{r['source']} {s}" for s in r["speakers"].split() if s != "?"}
        if r["source"] == "ami":
            # The same people are in every meeting of a series (ES2002a-d): hold out the series
            group = r["group"]
            speakers.add(f"ami {group[:-1] if group[-1].isalpha() else group}")
    (out_dir / "speakers.txt").write_text(
        "# Speakers (and AMI meeting series) in this eval set: never train on them\n"
        + "\n".join(sorted(speakers)) + "\n")
    print(f"\nWrote {path} ({len(rows)} pieces) and {out_dir / 'speakers.txt'} ({len(speakers)} entries)")


if __name__ == "__main__":
    main()
