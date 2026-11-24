"""
Listen to the Speech Commands keyword clips that check_keyword_clips.py flagged, one at a time, and
keep (y) or remove (n) each. Run it in a terminal: it plays each clip as it comes up and shows its
transcript, flags and an energy bar (where the word sits, and whether it runs into the clip's edge).

Clips come in tiers, most likely to be cut first. Within a tier the order is shuffled (fixed, by file
name), so if you stop part-way the clips reviewed so far are a fair sample of the rest of the tier:
  priority  1  empty, or partial + cut_end (Whisper heard a fragment and the word runs into the end)
            2  cut_end, the end within EDGE_LOUD_DB of the peak
            3  short
            4  cut_start, the start within EDGE_LOUD_DB of the peak
  the rest  5  cut_end / cut_start, quieter at the edge
            6  partial without cut_end, or quiet
            7  near / other / extra
Unflagged clips aren't in the tiers. --unflagged reviews the likeliest cuts among them: forward / backward
clips just past cut_end's limit (UNFLAGGED_END_DB), where flagged clips were still being cut.
--unflagged-groups samples two groups of unflagged clips that --unflagged leaves out (UNFLAGGED_GROUPS):
files shorter than 1 s, and stop clips just past cut_end's limit.

--public reviews a sample of the public eval set's keyword clips (val only: test is held back), to check
that they were cut around the whole word. y = the whole keyword is there, n = it isn't. First a random
sample balanced by source, then the edge cases (PUBLIC_EDGES): the windows shifted to a source edge, the
longest, shortest and quietest words. Clips reviewed before stay in the sample.

A rule (rule_decision()), fitted to the hand labels, decides the clips not reviewed: keep, remove, or
check (kept only once heard, removed otherwise: groups where the sample was mixed). It leans towards
removing: a cut word left in teaches the model that a fragment is the keyword, while a good clip
removed costs little. So only the clips it would keep or check need hearing (--rule-keeps, checks first).

Keys:  y keep   n remove   u unsure   space / r replay   b back one clip   v volume (normalised /
       original)   p pause (or q, Ctrl-C)
Each decision is saved as you make it. Pausing prints a report (removed per tier, keyword and flag,
and cut_end clips by how loud the end is), and a rerun carries on from the first clip not reviewed.

Output: keyword_clip_review.csv in HostSW/ (committed: these are hand labels), one row per clip:
file, keyword, decision (keep / remove / unsure), tier (0: unflagged), flags, reviewed (time).
--public's go in public_clip_review.csv instead, so it can run alongside a review of the others.

Usage (from HostSW/):
    python EvalScripts/review_keyword_clips.py            # review, carrying on where you stopped
    python EvalScripts/review_keyword_clips.py --report   # just the report
    python EvalScripts/review_keyword_clips.py --unsure   # go back over the clips marked unsure
    python EvalScripts/review_keyword_clips.py --sample 30   # just the first 30 of each tier (a fair sample:
                                                           # the order is shuffled), counting ones already done
    python EvalScripts/review_keyword_clips.py --rule-keeps  # just the clips the rule would check or keep
    python EvalScripts/review_keyword_clips.py --unflagged [DB]  # unflagged clips ending up to DB below the peak
                                                           # (add --report or --unsure for just those clips)
    python EvalScripts/review_keyword_clips.py --unflagged-groups [N]  # N random clips of each of the two groups
                                                           # (a larger N carries on; --report / --unsure too)
    python EvalScripts/review_keyword_clips.py --public [N]  # up to N random public val keyword clips per source
                                                           # + the edge cases (--report / --unsure too)
"""

import argparse
import csv
import datetime
import hashlib
import sys
import termios
import time
import tty
from collections import Counter

import numpy as np
import sounddevice as sd
import soundfile as sf

import check_keyword_clips as ck

REVIEW_PATH = ck.HOSTSW_DIR / "keyword_clip_review.csv"
PUBLIC_REVIEW_PATH = ck.HOSTSW_DIR / "public_clip_review.csv"
REVIEW_FIELDS = ["file", "keyword", "decision", "tier", "flags", "reviewed"]
PUBLIC_REVIEW_FIELDS = ["file", "keyword", "decision", "source", "reviewed"]
DECISION_KEYS = {"y": "keep", "n": "remove", "u": "unsure"}

EDGE_LOUD_DB = 10.0
TIERS = [   # (name, test(row, flags)): a clip goes in the first tier it matches
    ("empty, or partial + cut_end",
     lambda r, f: r["whisper_class"] == "empty" or (r["whisper_class"] == "partial" and "cut_end" in f)),
    (f"cut_end, end within {EDGE_LOUD_DB:.0f} dB of the peak",
     lambda r, f: "cut_end" in f and float(r["end_rel_db"]) >= -EDGE_LOUD_DB),
    ("short", lambda r, f: "short" in f),
    (f"cut_start, start within {EDGE_LOUD_DB:.0f} dB of the peak",
     lambda r, f: "cut_start" in f and float(r["start_rel_db"]) >= -EDGE_LOUD_DB),
    ("cut_end / cut_start, quieter at the edge", lambda r, f: "cut_end" in f or "cut_start" in f),
    ("partial without cut_end, or quiet", lambda r, f: r["whisper_class"] == "partial" or "quiet" in f),
    ("near / other / extra", lambda r, f: bool(f)),
]
PRIORITY_TIERS = 4
UNFLAGGED_TIER = 0              # --unflagged clips, outside the tiers
PUBLIC_TIER = -1                # --public clips
PUBLIC_SPLIT_DIR = ck.HOSTSW_DIR / "eval_recordings_public" / "val"   # never test: it's held back
PUBLIC_SAMPLE = 32              # default per source: all of AMI's
PUBLIC_EDGE_CLIPS = 10          # per edge case, except shifted windows: all of them (22)
PUBLIC_EDGES = ["shifted window", "longest word", "shortest word", "quietest word"]
UNFLAGGED_KEYWORDS = ["forward", "backward"]
UNFLAGGED_GROUPS = ["file under 1 s", "stop, word reaches end"]   # --unflagged-groups
UNFLAGGED_GROUP_SAMPLE = 30     # default clips per group
UNFLAGGED_END_DB = 25.0         # cut_end flags ends up to 20 dB below the peak; look this far past it

LEAD_S, TAIL_S = 0.15, 0.3      # silence around each clip: lets the output start up, makes a cut audible
PLAY_PEAK_DBFS = -3.0           # peak level when the volume is normalised
BAR_S = 0.02                    # energy bar: one character per 20 ms
BAR_LEVELS = "▁▂▃▄▅▆▇█"
END_BINS_DB = [0, 3, 6, 10, 15, 20]   # report: cut_end clips by how far below the peak the end is
MAX_GAP_S = 20.0                # pace: longer gaps between decisions are breaks

BOLD, DIM, RESET, CLEAR = "\033[1m", "\033[2m", "\033[0m", "\033[2J\033[H"


# --- Clips and decisions -----------------------------------------------------

def load_clips():
    """Every checked clip, each with 'flag_list'."""
    path = ck.DEFAULT_DATA_DIR / ck.RESULTS_NAME
    rows = ck.read_results(path)
    if not rows:
        sys.exit(f"no results in {path}: run check_keyword_clips.py first")
    for r in rows:   # from the stored transcript, as check_keyword_clips.py's report does
        r["whisper_class"] = ck.whisper_class(r["whisper_text"], r["keyword"])
    ck.add_short_flags(rows)
    for r in rows:
        r["flag_list"] = ck.flags(r)
    return rows


def review_order(rows):
    # Shuffled within a tier by a hash of the file name: fixed, and independent of the results' row order
    return sorted(rows, key=lambda r: (r["tier"], hashlib.md5(r["file"].encode()).hexdigest()))


def load_flagged():
    """The flagged clips in review order, each with 'flag_list' and 'tier'."""
    flagged = [r for r in load_clips() if r["flag_list"]]
    for r in flagged:
        r["tier"] = next(i for i, (_, test) in enumerate(TIERS, 1) if test(r, r["flag_list"]))
    return review_order(flagged)


def load_unflagged(end_db):
    """The unflagged forward / backward clips whose word reaches the clip's end, with the end less than
    end_db below the peak, in review order (tier UNFLAGGED_TIER)."""
    clips = [r for r in load_clips() if not r["flag_list"] and r["keyword"] in UNFLAGGED_KEYWORDS
             and -float(r["end_rel_db"]) < end_db
             and float(r["word_end_s"]) >= float(r["duration_s"]) - WORD_BEFORE_END_S]
    for r in clips:
        r["tier"] = UNFLAGGED_TIER
    return review_order(clips)


def load_unflagged_groups(n):
    """A random sample of n unflagged clips from each of UNFLAGGED_GROUPS (the first n in hash order, so a
    larger n extends it), leaving out the clips --unflagged covers. Each has 'group' and 'group_size'."""
    covered = {r["file"] for r in load_unflagged(UNFLAGGED_END_DB)}
    full = lambda r: float(r["duration_s"]) >= 0.999   # noqa: E731
    members = {name: [] for name in UNFLAGGED_GROUPS}
    for r in load_clips():
        if r["flag_list"] or r["file"] in covered:
            continue
        if not full(r):
            members[UNFLAGGED_GROUPS[0]].append(r)
        elif (r["keyword"] == "stop" and -float(r["end_rel_db"]) < UNFLAGGED_END_DB
              and float(r["word_end_s"]) >= float(r["duration_s"]) - WORD_BEFORE_END_S):
            members[UNFLAGGED_GROUPS[1]].append(r)
    clips = []
    for name, rows in members.items():
        for r in rows:
            r.update(tier=UNFLAGGED_TIER, group=name, group_size=len(rows))
        clips += review_order(rows)[:n]
    return clips


def word_level_db(audio, word_start_s, word_end_s):
    """The word's loudest 10 ms frame relative to the clip's."""
    db = ck.frame_db(audio)
    word = db[int(word_start_s / ck.FRAME_S):max(int(word_start_s / ck.FRAME_S) + 1, round(word_end_s / ck.FRAME_S))]
    return float(word.max() - db.max()) if word.size else -100.0


def load_public(n, latest=()):
    """The sample of public val keyword clips: the first n per source in hash order (random; a larger n
    extends it), then the edge cases not already in it. Clips already reviewed (latest) count as in the
    random sample unless they're edge cases. The word is the alignment's span, not the energy measure's,
    since other words share the window."""
    path = PUBLIC_SPLIT_DIR / "clips.csv"
    if not path.exists():
        sys.exit(f"no {path}: build the public eval set with make_public_eval.py first")
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["label"] in ck.KEYWORDS]
    clips = []
    for r in rows:
        start = float(r["start_s"])
        span = [float(r["span_start_s"]) - start, float(r["span_end_s"]) - start]
        length = float(r["end_s"]) - start
        audio, _ = sf.read(str(PUBLIC_SPLIT_DIR / r["file"]), dtype="float32")
        clips.append({"file": ck.rel(PUBLIC_SPLIT_DIR / r["file"]), "keyword": r["label"], "source": r["source"],
                      "tier": PUBLIC_TIER, "flag_list": [], "duration_s": length,
                      "word_start_s": span[0], "word_end_s": span[1], "piece_id": r["piece_id"],
                      "shifted": abs(sum(span) / 2 - length / 2) > 0.001,
                      "level_db": word_level_db(audio, *span), "edges": []})
    clips = review_order(clips)
    length = lambda r: r["word_end_s"] - r["word_start_s"]   # noqa: E731
    edge_sets = {"shifted window": [r for r in clips if r["shifted"]],
                 "longest word": sorted(clips, key=length, reverse=True)[:PUBLIC_EDGE_CLIPS],
                 "shortest word": sorted(clips, key=length)[:PUBLIC_EDGE_CLIPS],
                 "quietest word": sorted(clips, key=lambda r: r["level_db"])[:PUBLIC_EDGE_CLIPS]}
    for name in PUBLIC_EDGES:
        for r in edge_sets[name]:
            r["edges"].append(name)
    seen = Counter()
    for r in clips:
        seen[r["source"]] += 1
        r["random"] = seen[r["source"]] <= n or (r["file"] in latest and not r["edges"])
    return [r for r in clips if r["random"]] + [r for r in clips if r["edges"] and not r["random"]]


RULE_TIERS = 7                  # the rule covers tiers 1..RULE_TIERS (the others haven't been sampled yet)
EARLY_WORD_S = 0.2              # stop: a word starting this early in a 1 s clip is complete, and sound after it...
WORD_BEFORE_END_S = 0.05        # ...or any word ending this long before the clip does: either way the loud end isn't it
LOUD_END_DB = 12.0              # forward / backward: an end this close to the peak is the word being cut


def rule_decision(r):
    """keep / remove / check (kept only once heard) for a flagged clip, from the hand labels so far (see the
    docstring); None where the rule doesn't reach yet."""
    if r["tier"] > RULE_TIERS:
        return None
    f, full = set(r["flag_list"]), float(r["duration_s"]) >= 0.999
    if r["tier"] >= 6:
        # No edge flags, but a word that reaches the clip's end (or a file cut short) may still be cut
        word_at_end = float(r["word_end_s"]) >= float(r["duration_s"]) - WORD_BEFORE_END_S
        if word_at_end or not full or "quiet" in f:
            return "remove"
        # Whisper's fragments and near-misses were complete words; 12 of 153 'other' were cut
        return "check" if r["whisper_class"] == "other" else "keep"
    if r["whisper_class"] != "ok":
        return "remove"
    # The early start holds for stop only (its keeps were all stop; a backward one was cut): the span can
    # start on noise before the word
    end_not_word = (full and "cut_end" in f and "cut_start" not in f
                    and ((r["keyword"] == "stop" and float(r["word_start_s"]) < EARLY_WORD_S)
                         or float(r["word_end_s"]) < float(r["duration_s"]) - WORD_BEFORE_END_S))
    if end_not_word:
        return "keep"
    if r["keyword"] == "stop":
        # 'short' alone is mostly a false flag for stop (its quiet s and p fall below the energy threshold),
        # but 1 of the 29 reviewed was cut and nothing measured tells it apart
        return "check" if f == {"short"} and full else "remove"
    if "cut_end" in f and (float(r["end_rel_db"]) >= -LOUD_END_DB or not full):
        return "remove"
    return "keep"


def read_review(path):
    """file -> its latest decision row (rows are appended while reviewing, so a changed mind comes later)."""
    latest = {}
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                latest[r["file"]] = r
    return latest


def save_decision(path, latest, r, decision):
    fields = PUBLIC_REVIEW_FIELDS if r["tier"] == PUBLIC_TIER else REVIEW_FIELDS
    row = {"file": r["file"], "keyword": r["keyword"], "decision": decision, "tier": r["tier"],
           "flags": " ".join(r["flag_list"]), "source": r.get("source"),
           "reviewed": datetime.datetime.now().isoformat(timespec="seconds")}
    row = {k: row[k] for k in fields}
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(row)
    latest[r["file"]] = row


def compact_review(path):
    """Rewrites the review file with one row per clip, sorted by file. From the file, not this session's
    copy, so decisions another session has saved meanwhile are kept."""
    latest = read_review(path)
    if not latest:
        return
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(latest.values()))))
        writer.writeheader()
        writer.writerows(sorted(latest.values(), key=lambda r: r["file"]))
    tmp.replace(path)


# --- Screen and sound --------------------------------------------------------

def play(audio, normalise):
    if normalise:
        audio = audio * (10 ** (PLAY_PEAK_DBFS / 20) / max(float(np.abs(audio).max()), 1e-6))
    silence = lambda s: np.zeros(round(s * ck.SAMPLE_RATE_HZ), np.float32)   # noqa: E731
    sd.play(np.concatenate([silence(LEAD_S), audio.astype(np.float32), silence(TAIL_S)]), ck.SAMPLE_RATE_HZ)


def energy_bar(audio, r):
    """One character per BAR_S over (at least) 1 s: each 20 ms's loudest frame between the clip's floor
    and peak, '·' past the file's end. And a line marking the word."""
    db, per = ck.frame_db(audio), round(BAR_S / ck.FRAME_S)
    floor, peak = float(np.percentile(db, 10)), float(db.max())
    width = max(round(1.0 / BAR_S), -(-len(db) // per))
    bar = ""
    for i in range(width):
        chunk = db[i * per:(i + 1) * per]
        level = np.clip((chunk.max() - floor) / max(peak - floor, 1e-6), 0, 1) if chunk.size else None
        bar += "·" if level is None else BAR_LEVELS[round(level * (len(BAR_LEVELS) - 1))]
    start = int(float(r["word_start_s"]) / BAR_S)
    end = max(start + 1, round(float(r["word_end_s"]) / BAR_S))
    return bar, " " * start + "━" * (end - start)


def clip_lines(r):
    return [f"  whisper    \"{r['whisper_text']}\"   ({r['whisper_class']})",
            f"  flags      {BOLD}{', '.join(r['flag_list']) or 'none'}{RESET}",
            f"  edges      start {abs(float(r['start_rel_db'])):.0f} dB below the peak · "
            f"end {abs(float(r['end_rel_db'])):.0f} dB below the peak"]


def public_lines(r):
    return [f"  piece      {r['piece_id']}",
            f"  alignment  word {r['word_start_s']:.2f}-{r['word_end_s']:.2f} s"
            + (f"   {BOLD}edge case: {', '.join(r['edges'])}{RESET}" if r["edges"] else ""),
            "  keep       if the whole keyword is in the clip (other words around it are normal)"]


def show(r, audio, progress, normalise, decision=None):
    tier, n_tiers = r["tier"], len(TIERS)
    if tier == PUBLIC_TIER:
        where = f"public val · {r['source']}"
    elif tier == UNFLAGGED_TIER:
        where = f"unflagged, {r['group']}" if "group" in r else "unflagged, word reaches the end"
    else:
        phase = "priority" if tier <= PRIORITY_TIERS else "the rest"
        where = f"{phase} · tier {tier}/{n_tiers}: {TIERS[tier - 1][0]}"
    bar, word = energy_bar(audio, r)
    half = round(0.5 / BAR_S)
    scale = "0 s".ljust(half) + "0.5 s".ljust(half) + "1 s"   # each label starts at its time
    lines = [
        f"{CLEAR}{DIM}Keyword clip review · {where}{RESET}",
        f"{DIM}{progress}{RESET}",
        "",
        f"  {BOLD}{r['keyword'].upper():10s}{RESET} {r['file']}   {float(r['duration_s']):.2f} s",
        *(public_lines(r) if tier == PUBLIC_TIER else clip_lines(r)),
        "",
        f"  energy     {bar}",
        f"  word       {word}",
        f"  {DIM}           {scale}{RESET}",
        "",
    ]
    if decision:
        lines.append(f"  {BOLD}(marked {decision} before){RESET}")
    lines.append(f"  {BOLD}y{RESET} keep   {BOLD}n{RESET} remove   {BOLD}u{RESET} unsure   "
                 f"{BOLD}space{RESET} replay   {BOLD}b{RESET} back   "
                 f"{BOLD}v{RESET} volume: {'normalised' if normalise else 'original'}   {BOLD}p{RESET} pause + report")
    print("\n".join(lines), flush=True)


def is_unflagged(clips):
    return bool(clips) and clips[0]["tier"] == UNFLAGGED_TIER


def is_unflagged_groups(clips):
    return is_unflagged(clips) and "group" in clips[0]


def is_public(clips):
    return bool(clips) and clips[0]["tier"] == PUBLIC_TIER


def progress_line(flagged, latest):
    done = [r for r in flagged if r["file"] in latest]
    removed = sum(latest[r["file"]]["decision"] == "remove" for r in done)
    if is_public(flagged):
        count = lambda test: f"{sum(test(r) for r in done)} / {sum(test(r) for r in flagged)}"   # noqa: E731
        sources = " · ".join(f"{src} {count(lambda r, src=src: r['random'] and r['source'] == src)}"
                             for src in sorted({r["source"] for r in flagged}))
        return (f"reviewed {len(done)} / {len(flagged)} public · {sources} · "
                f"edge cases {count(lambda r: not r['random'])} · {removed} marked remove")
    if is_unflagged_groups(flagged):
        count = lambda test: f"{sum(test(r) for r in done)} / {sum(test(r) for r in flagged)}"   # noqa: E731
        groups = " · ".join(f"{name} {count(lambda r, name=name: r['group'] == name)}" for name in UNFLAGGED_GROUPS)
        return f"reviewed {len(done)} / {len(flagged)} unflagged sample · {groups} · {removed} marked remove"
    if is_unflagged(flagged):
        per_keyword = " · ".join(f"{kw} {sum(r['keyword'] == kw for r in done)} / "
                                 f"{sum(r['keyword'] == kw for r in flagged)}" for kw in UNFLAGGED_KEYWORDS)
        return f"reviewed {len(done)} / {len(flagged)} unflagged · {per_keyword} · {removed} marked remove"
    prio = sum(r["tier"] <= PRIORITY_TIERS for r in flagged)
    prio_done = sum(r["tier"] <= PRIORITY_TIERS for r in done)
    return (f"reviewed {prio_done} / {prio} priority · {len(done)} / {len(flagged)} flagged · "
            f"{removed} marked remove")


# --- Report ------------------------------------------------------------------

def pace_s(times):
    """Seconds per clip this session, leaving out breaks."""
    gaps = [b - a for a, b in zip(times, times[1:]) if b - a < MAX_GAP_S]
    return float(np.mean(gaps)) if len(gaps) >= 5 else None


def report_groups(clips, latest, times, name, groups, path):
    done = [(r, latest[r["file"]]["decision"]) for r in clips if r["file"] in latest]
    print(f"\n{BOLD}Reviewed {len(done)} of {len(clips)} {name}{RESET}")
    pace = pace_s(times)
    if times:
        print(f"This session: {len(times)} decision{'s' * (len(times) != 1)}"
              + (f", {pace:.1f} s per clip, ~{(len(clips) - len(done)) * pace / 60:.0f} min left" if pace else ""))
    print(f"\n  {'':24s} {'clips':>6s} {'reviewed':>9s} {'remove':>7s} {'keep':>5s} {'unsure':>7s}  remove %")
    for label, test in groups:
        c = Counter(d for r, d in done if test(r))
        decided = c["keep"] + c["remove"]
        pct = f"{100 * c['remove'] / decided:7.0f}%" if decided else "       -"
        print(f"  {label:24s} {sum(test(r) for r in clips):6d} {sum(c.values()):9d} {c['remove']:7d} "
              f"{c['keep']:5d} {c['unsure']:7d} {pct}")
    print(f"Decisions are in {ck.rel(path)}")


def report_unflagged(clips, latest, times=()):
    end = lambda r: -float(r["end_rel_db"])   # noqa: E731
    groups = [("all", lambda r: True)]
    groups += [(kw, lambda r, kw=kw: r["keyword"] == kw) for kw in UNFLAGGED_KEYWORDS]
    groups += [(f"end {a}-{b} dB below peak", lambda r, a=a, b=b: a <= end(r) < b) for a, b in
               [(0, 20), (20, 22), (22, 25), (25, 30), (30, 200)] if any(a <= end(r) < b for r in clips)]
    report_groups(clips, latest, times, "unflagged clips", groups, REVIEW_PATH)


def report_unflagged_groups(clips, latest, times=()):
    groups = []
    for name in UNFLAGGED_GROUPS:
        size = next((r["group_size"] for r in clips if r["group"] == name), 0)
        print(f"{name}: {sum(r['group'] == name for r in clips)} sampled of {size}")
        groups.append((name, lambda r, name=name: r["group"] == name))
        groups += [(f"  {kw}", lambda r, name=name, kw=kw: r["group"] == name and r["keyword"] == kw)
                   for kw in ck.KEYWORDS if any(r["group"] == name and r["keyword"] == kw for r in clips)]
    report_groups(clips, latest, times, "sampled unflagged clips", groups, REVIEW_PATH)


def report_public(clips, latest, times=()):
    groups = [("all", lambda r: True), ("random sample", lambda r: r["random"])]
    groups += [(f"  {src}", lambda r, src=src: r["random"] and r["source"] == src)
               for src in sorted({r["source"] for r in clips})]
    groups += [(f"  {kw}", lambda r, kw=kw: r["random"] and r["keyword"] == kw) for kw in ck.KEYWORDS]
    # An edge case can also be in the random sample, so these overlap with it (and each other)
    groups += [(f"edge: {name}", lambda r, name=name: name in r["edges"]) for name in PUBLIC_EDGES]
    report_groups(clips, latest, times, "sampled public val keyword clips", groups, PUBLIC_REVIEW_PATH)
    bad = [r for r in clips if latest.get(r["file"], {}).get("decision") == "remove"]
    if bad:
        print("\nMarked remove:")
        for r in bad:
            print(f"  {r['file']}   {r['source']} {r['piece_id']}   word {r['word_start_s']:.2f}-{r['word_end_s']:.2f} s"
                  + (f"   {', '.join(r['edges'])}" if r["edges"] else ""))


def report(flagged, latest, times=()):
    if is_unflagged_groups(flagged):
        report_unflagged_groups(flagged, latest, times)
        return
    if is_unflagged(flagged):
        report_unflagged(flagged, latest, times)
        return
    if is_public(flagged):
        report_public(flagged, latest, times)
        return
    done = [(r, latest[r["file"]]["decision"]) for r in flagged if r["file"] in latest]
    prio = [r for r in flagged if r["tier"] <= PRIORITY_TIERS]
    prio_done = sum(r["tier"] <= PRIORITY_TIERS for r, _ in done)
    print(f"\n{BOLD}Reviewed {len(done)} of {len(flagged)} flagged clips ({prio_done} of {len(prio)} priority){RESET}")
    pace = pace_s(times)
    if times:
        print(f"This session: {len(times)} decision{'s' * (len(times) != 1)}"
              + (f", {pace:.1f} s per clip" if pace else ""))
    if pace:
        left = lambda rows: sum(r["file"] not in latest for r in rows) * pace / 60   # noqa: E731
        print(f"At that pace: priority ~{left(prio):.0f} min left, all flagged ~{left(flagged):.0f} min")

    def table(title, groups):
        print(f"\n{title}")
        print(f"  {'':46s} {'clips':>6s} {'reviewed':>9s} {'remove':>7s} {'keep':>5s} {'unsure':>7s}  remove %")
        for label, test in groups:
            n = sum(test(r) for r in flagged)
            c = Counter(d for r, d in done if test(r))
            decided = c["keep"] + c["remove"]
            pct = f"{100 * c['remove'] / decided:7.0f}%" if decided else "       -"
            print(f"  {label:46s} {n:6d} {sum(c.values()):9d} {c['remove']:7d} {c['keep']:5d} {c['unsure']:7d} {pct}")

    table("By tier (each clip in one tier)",
          [(f"{i} {name}" + (" *" if i <= PRIORITY_TIERS else ""), lambda r, i=i: r["tier"] == i)
           for i, (name, _) in enumerate(TIERS, 1)])
    print("  * priority")
    table("By keyword", [(kw, lambda r, kw=kw: r["keyword"] == kw) for kw in ck.KEYWORDS])
    table("By flag (a clip counts under each of its flags)",
          [(f, lambda r, f=f: f in r["flag_list"]) for f in ck.WHISPER_FLAGS + ck.ENERGY_FLAGS])
    end_bins = [(f"end {a}-{b} dB below the peak", lambda r, a=a, b=b: "cut_end" in r["flag_list"]
                 and a <= -float(r["end_rel_db"]) < b + (b == END_BINS_DB[-1]))
                for a, b in zip(END_BINS_DB, END_BINS_DB[1:])]
    length = [("file shorter than 1 s", lambda r: "cut_end" in r["flag_list"] and float(r["duration_s"]) < 0.999),
              ("file 1 s", lambda r: "cut_end" in r["flag_list"] and float(r["duration_s"]) >= 0.999)]
    table("cut_end clips, by how loud the end is, and by file length", end_bins + length)

    print(f"\nThe rule (rule_decision(), tiers 1-{RULE_TIERS}) on the clips you reviewed, and on the rest")
    print(f"  {'':10s} {'agrees':>7s} {'of':>4s}  {'removes a keep':>15s} {'keeps a remove':>15s} {'check':>6s}"
          f"   not reviewed: {'keep':>5s} {'check':>6s} {'remove':>7s} {'no rule':>8s}")
    for label, test in [(f"tier {i}", lambda r, i=i: r["tier"] == i) for i in range(1, len(TIERS) + 1)] + \
                       [("all", lambda r: True)]:
        rev = [(d, rule_decision(r)) for r, d in done if test(r) and d != "unsure" and rule_decision(r)]
        left = Counter(rule_decision(r) for r in flagged if test(r) and r["file"] not in latest)
        print(f"  {label:10s} {sum(a == b for a, b in rev):7d} {sum(b != 'check' for _, b in rev):4d}  "
              f"{sum(a == 'keep' and b == 'remove' for a, b in rev):15d} {sum(a == 'remove' and b == 'keep' for a, b in rev):15d}"
              f" {sum(b == 'check' for _, b in rev):6d}                 {left['keep']:5d} {left['check']:6d} {left['remove']:7d} {left[None]:8d}")
    print("  check: kept only once heard. 'keeps a remove' must stay 0. Hear the rest of the keeps and checks with\n"
          "  --rule-keeps (checks first): any not heard count as remove")

    removed = Counter(r["keyword"] for r, d in done if d == "remove")
    unsure = sum(d == "unsure" for _, d in done)
    print(f"\nMarked remove: {sum(removed.values())} ("
          + ", ".join(f"{kw} {removed[kw]}" for kw in ck.KEYWORDS) + ")"
          + (f". Unsure: {unsure} (go back over them with --unsure)" if unsure else ""))
    print(f"Decisions are in {ck.rel(REVIEW_PATH)}")


# --- Review loop -------------------------------------------------------------

def read_key():
    return sys.stdin.read(1).lower()


def review(flagged, only_unsure, sample=None, rule_keeps=False):
    path = PUBLIC_REVIEW_PATH if is_public(flagged) else REVIEW_PATH
    latest = read_review(path)
    if only_unsure:
        queue = [r for r in flagged if latest.get(r["file"], {}).get("decision") == "unsure"]
        pending = lambda r: latest[r["file"]]["decision"] == "unsure"   # noqa: E731
    else:
        queue = flagged
        if sample:   # each tier's first clips in review order: a random sample, since a tier is shuffled
            queue = [r for t in range(1, len(TIERS) + 1) for r in [q for q in flagged if q["tier"] == t][:sample]]
        elif rule_keeps:   # the checks first (the less certain), then the keeps
            queue = ([r for r in flagged if rule_decision(r) == "check"]
                     + [r for r in flagged if rule_decision(r) == "keep"])
        pending = lambda r: r["file"] not in latest   # noqa: E731
    next_pending = lambda i: next((j for j in range(i, len(queue)) if pending(queue[j])), len(queue))   # noqa: E731
    pos = next_pending(0)
    if pos == len(queue):
        print("No clips marked unsure." if only_unsure else "Sample done." if sample
              else "Rule keeps and checks all reviewed." if rule_keeps
              else "Unflagged clips all reviewed." if is_unflagged(flagged)
              else "Public sample all reviewed." if is_public(flagged) else "Nothing left to review.")
        report(flagged, latest)
        return

    fd = sys.stdin.fileno()
    old_mode = termios.tcgetattr(fd)
    times, normalise, announced = [], True, (only_unsure or bool(sample) or rule_keeps   # the priority-done stop
                                           or is_unflagged(flagged) or is_public(flagged))
    try:
        tty.setcbreak(fd)
        while pos < len(queue):
            r = queue[pos]
            audio, _ = sf.read(str(ck.HOSTSW_DIR / r["file"]), dtype="float32")
            decision = latest.get(r["file"], {}).get("decision")
            show(r, audio, progress_line(flagged, latest), normalise, decision)
            play(audio, normalise)
            while (key := read_key()) not in {*DECISION_KEYS, "b", "p", "q"}:
                if key in (" ", "r"):
                    play(audio, normalise)
                elif key == "v":
                    normalise = not normalise
                    show(r, audio, progress_line(flagged, latest), normalise, decision)
                    play(audio, normalise)
            sd.stop()
            if key in ("p", "q"):
                break
            if key == "b":
                pos = max(pos - 1, 0)
                continue
            save_decision(path, latest, r, DECISION_KEYS[key])
            times.append(time.time())
            new_pos = next_pending(pos + 1)
            # Once, when the priority clips run out: a report, and the choice to stop there
            if (not announced and new_pos < len(queue) and r["tier"] <= PRIORITY_TIERS
                    and queue[new_pos]["tier"] > PRIORITY_TIERS
                    and not any(pending(q) for q in queue if q["tier"] <= PRIORITY_TIERS)):
                announced = True
                print(CLEAR + f"{BOLD}Priority clips done.{RESET}")
                report(flagged, latest, times)
                print(f"\n{BOLD}Press any key to go on with the rest of the flagged clips, or p to pause.{RESET}",
                      flush=True)
                if read_key() in ("p", "q"):
                    break
            pos = new_pos
    except KeyboardInterrupt:
        pass
    finally:
        sd.stop()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_mode)
    compact_review(path)
    print(CLEAR + (f"{BOLD}{'Sample' if sample else 'Rule keeps' if rule_keeps else 'All'} done.{RESET}"
                   if pos >= len(queue) else f"{BOLD}Paused.{RESET} "
                   "Run the same command to carry on."))
    report(flagged, latest, times)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="store_true", help="just print the report")
    parser.add_argument("--unsure", action="store_true", help="go back over the clips marked unsure")
    parser.add_argument("--sample", type=int, metavar="N",
                        help="review only each tier's first N clips (a fair sample), counting ones already done")
    parser.add_argument("--rule-keeps", action="store_true",
                        help="review only the clips the rule would check or keep (checks first)")
    parser.add_argument("--unflagged", type=float, nargs="?", const=UNFLAGGED_END_DB, metavar="DB",
                        help="review the unflagged forward / backward clips whose word reaches the clip's end, "
                             f"less than DB below the peak (default {UNFLAGGED_END_DB:.0f}); --report / --unsure "
                             "then cover just those")
    parser.add_argument("--unflagged-groups", type=int, nargs="?", const=UNFLAGGED_GROUP_SAMPLE, metavar="N",
                        help="review N random unflagged clips from each of: files under 1 s, stop clips whose word "
                             f"reaches the end (default {UNFLAGGED_GROUP_SAMPLE}); --report / --unsure then cover "
                             "just those")
    parser.add_argument("--public", type=int, nargs="?", const=PUBLIC_SAMPLE, metavar="N",
                        help=f"review up to N random public val keyword clips per source (default {PUBLIC_SAMPLE}) and "
                             "the edge cases; --report / --unsure then cover just those")
    args = parser.parse_args()
    if args.public is not None and (args.sample or args.rule_keeps or args.unflagged is not None or args.public < 1):
        parser.error("--public needs N >= 1 and can't be combined with --sample, --rule-keeps or --unflagged")
    if args.unflagged is not None and (args.sample or args.rule_keeps):
        parser.error("--unflagged can't be combined with --sample or --rule-keeps")
    if args.unflagged_groups is not None and (args.sample or args.rule_keeps or args.unflagged is not None
                                              or args.public is not None or args.unflagged_groups < 1):
        parser.error("--unflagged-groups needs N >= 1 and can't be combined with the other clip sets")
    flagged = (load_unflagged_groups(args.unflagged_groups) if args.unflagged_groups is not None
               else load_public(args.public, read_review(PUBLIC_REVIEW_PATH)) if args.public is not None
               else load_flagged() if args.unflagged is None else load_unflagged(args.unflagged))
    if args.report:
        report(flagged, read_review(PUBLIC_REVIEW_PATH if args.public is not None else REVIEW_PATH))
        return
    if not sys.stdin.isatty():
        sys.exit("run this in a terminal: it reads single key presses")
    review(flagged, args.unsure, args.sample, args.rule_keeps)


if __name__ == "__main__":
    main()
