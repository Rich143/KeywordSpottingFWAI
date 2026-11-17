"""
Download background-noise audio without intelligible speech, for the _background_noise_ class
(training) and for non-speech eval negatives: music, household and other environmental noise.
Resumable with retries and progress bars (download_ami.py's downloader).

Sources (all Creative Commons):
  musan   MUSAN (openslr.org/17, CC BY 4.0): ~6 h of assorted noise (technical: DTMF, dial
          tones, fax; ambient: car idling, rain, wind, thunder, footsteps, paper, animals; no
          intelligible speech) and ~42.5 h of music, annotated for vocals. It's one 11 GB
          .tar.gz that can't be downloaded in parts; only noise/ and music/ are extracted, not
          speech/ (20 h of it is LibriVox, the source of LibriSpeech, so its readers may be in
          the public eval set). Music with vocals is extracted too but marked vocals = yes in
          the manifest: lyrics can contain the keywords, so it isn't background.
  demand  DEMAND (zenodo.org/records/1227121, CC BY 4.0 on Zenodo, CC BY-SA 3.0 in its PDF):
          ~5 min per environment, 16-channel array. The 16 kHz zip of each environment
          without conversation is downloaded and one channel kept (the mics are 5-22 cm
          apart, so the others add little). Left out: OMEETING (a discussion), PCAFETER and
          PRESTO (busy cafeteria / restaurant), SPSQUARE (square with many tourists), SCAFE
          (no 16 kHz version).
  soh     Sounds of Home (zenodo.org/records/12737915 and three more parts, CC BY 4.0): one-
          hour 48 kHz recordings from 14 AudioMoth mics, two per home (living room and
          kitchen) in 7 homes, 07:00-22:00 over a week, with speech removed by the dataset
          authors. 1,344 hours (~200 GB) in four big zips: only --soh-hours-per-mic hours per
          mic are downloaded, each by HTTP range straight out of its zip, spread over days
          and times of day (a home's two mics get different hours). Each hour's automatic
          sound-event labels (<stamp>_light.json) come with it. Part 1 is compressed with
          Deflate64, which Python can't decompress: members are unpacked with the system
          unzip (macOS's supports it), then checked against the zip's CRC.

Output (--out, default HostSW/dataset_noise/, gitignored):
  musan/noise/<sub>/*.wav, musan/music/<sub>/*.wav (+ ANNOTATIONS, README, LICENSE)
  demand/<ENV>/ch01.wav
  sounds_of_home/<mic>/<YYYYMMDD_HHMMSS>.wav + <YYYYMMDD_HHMMSS>_light.json
  manifest.csv   one row per audio file: source, file, category, group, vocals, duration_s,
                 sample_rate, licence. Split train / val / test by group, never by clip
                 (make_public_eval.py does, into eval_recordings_public/splits.csv). Groups:
                 MUSAN music by artist (a performer's tracks sound alike), Sound Bible noise by
                 recorder, Free Sound noise by file (no names), a DEMAND environment, a Sounds
                 of Home home: its two mics record at the same time, so can hear the same sound
Sounds of Home homes are mic folders (01, 02), (03, 04), ... (the dataset's README on GitHub,
gbibbo/voice_anonymization). Its speech was muted by the dataset authors; noise_speech.py finds
where, and any speech left in all three datasets.

Speed: the servers limit each connection (measured Nov 2026: Zenodo ~0.7 MB/s, openslr.org
~0.3 MB/s, its EU mirror 0.5-3 MB/s depending on the time), so --workers downloads run at once
(default 8), from all sources together. MUSAN is one 10.3 GB file, so it's split into
--musan-segments byte ranges (default 4), downloaded in parallel and joined once all are in. They
come from the EU mirror: the ranges are equal, so one from openslr.org would finish hours after
the rest (--musan-mirror elda openslr spreads them over both; the two serve the identical file).
A partial musan.tar.gz.part from a single-stream download becomes the start of the first range.

Resuming: re-run the same command. Finished files are skipped, partial downloads continue,
and an extraction that was interrupted is redone. Archives (musan.tar.gz, DEMAND zips) are
kept unless --delete-archives. Only one copy of the script can run on an output dir at a time
(two writing the same partial file would corrupt it).

Usage (from HostSW/):
    python EvalScripts/download_noise.py --dry-run
    python EvalScripts/download_noise.py
    python EvalScripts/download_noise.py --sources demand soh --soh-hours-per-mic 10
    python EvalScripts/download_noise.py --workers 6
    python EvalScripts/download_noise.py --manifest-only
"""

import argparse
import csv
import fcntl
import io
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
import zlib

import soundfile as sf
from tqdm import tqdm

import download_ami as dl

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
HOSTSW_DIR = SCRIPT_DIR.parent
DEFAULT_OUT_DIR = HOSTSW_DIR / "dataset_noise"
SOURCES = ["musan", "demand", "soh"]
MANIFEST_NAME = "manifest.csv"

MUSAN_MIRRORS = {
    "elda": "https://openslr.elda.org/resources/17/musan.tar.gz",   # EU mirror, ~3 MB/s
    "openslr": "https://www.openslr.org/resources/17/musan.tar.gz",  # ~0.3 MB/s per connection
}
DEFAULT_WORKERS = 8
MUSAN_SEGMENTS = 4
LOCK_NAME = ".download.lock"
MUSAN_SIZE = 11086114085
MUSAN_KEEP = ("musan/noise/", "musan/music/")
MUSAN_TOP_FILES = ("musan/README", "musan/LICENSE")

DEMAND_URL = "https://zenodo.org/api/records/1227121/files/{env}_16k.zip/content"
DEMAND_ENVS = {   # from DEMAND.pdf
    "DKITCHEN": "kitchen during the preparation of food",
    "DLIVING": "living room",
    "DWASHING": "washroom with a front-loading washer running",
    "OOFFICE": "small office, three people using computers",
    "OHALLWAY": "office hallway, people passing by occasionally",
    "PSTATION": "main transfer area of a busy subway station",
    "NFIELD": "sports field with activity nearby",
    "NPARK": "well-visited city park",
    "NRIVER": "creek of running water",
    "STRAFFIC": "busy traffic intersection",
    "TBUS": "inside a public transit bus",
    "TCAR": "inside a car",
    "TMETRO": "inside a subway train",
}
DEMAND_CHANNELS = ["ch01"]

SOH_PARTS = [   # (Zenodo record, archive name): mic folders 01-07, 08-10, 11-13, 14
    ("12737915", "VITALISE data light 1.0.zip"),
    ("12800789", "VITALISE data light 2.1.zip"),
    ("12802632", "VITALISE data light 2.2.zip"),
    ("12802716", "VITALISE data light 2.3.zip"),
]
SOH_URL = "https://zenodo.org/api/records/{record}/files/{name}/content"
SOH_HOURS_PER_MIC = 8

LICENCES = {"musan": "CC BY 4.0 (per file: CC or US public domain)", "demand": "CC BY 4.0 / CC BY-SA 3.0",
            "soh": "CC BY 4.0"}


# --- Remote zips (Sounds of Home) --------------------------------------------

class HttpRangeFile(io.RawIOBase):
    """A read-only file over HTTP Range requests, so zipfile can read a remote zip's
    central directory without downloading the archive."""

    def __init__(self, url, timeout):
        self.url, self.timeout, self.pos = url, timeout, 0
        with dl.request(url, timeout, method="HEAD") as resp:
            self.size = int(resp.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer):
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        with dl.request(self.url, self.timeout, offset=self.pos, end=self.pos + n) as resp:
            data = resp.read()
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def remote_members(url, timeout):
    """[(ZipInfo, end offset of its local record)] of a remote zip, from its central directory.
    A member's record (local header, data, any data descriptor) runs to the next one's header."""
    with zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url, timeout), buffer_size=1 << 20)) as zf:
        infos = sorted((i for i in zf.infolist() if not i.is_dir()), key=lambda i: i.header_offset)
        ends = [i.header_offset for i in infos[1:]] + [zf.start_dir]
    return list(zip(infos, ends))


def central_record(info):
    """A central directory entry for info, as the only member at offset 0 of a new zip."""
    year, month, day, hour, minute, second = info.date_time
    dos_date = (year - 1980) << 9 | month << 5 | day
    dos_time = hour << 11 | minute << 5 | second // 2
    name = info.filename.encode("utf-8" if info.flag_bits & 0x800 else "cp437")
    return struct.pack("<4s4B4HL2L5H2L", b"PK\x01\x02", info.create_version, info.create_system,
                       info.extract_version, info.reserved, info.flag_bits, info.compress_type, dos_time,
                       dos_date, info.CRC, info.compress_size, info.file_size, len(name), 0, 0, 0, 0,
                       info.external_attr, 0) + name


def unpack_member(record_path, record_size, info, dest):
    """Turns a downloaded local record (record_size bytes) into dest: appends a central directory
    to make it a one-member zip, unpacks it with the system unzip (Deflate64 included), checks
    size and CRC. Safe to repeat: the file is cut back to the record first. A record that unpacks
    to the wrong size or CRC is deleted, so the next run downloads it again."""
    with open(record_path, "rb") as f:
        if f.read(4) != b"PK\x03\x04":
            record_path.unlink()
            raise RuntimeError(f"{record_path.name}: not a zip local header, deleted to download again")
    os.truncate(record_path, record_size)
    cd_offset = record_size
    central = central_record(info)
    end = struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, 1, 1, len(central), cd_offset, 0)
    with open(record_path, "ab") as f:
        f.write(central + end)
    part = dest.with_name(dest.name + ".part")
    try:
        with open(part, "wb") as out:
            proc = subprocess.run(["unzip", "-p", str(record_path)], stdout=out, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"unzip failed on {record_path.name}: {proc.stderr.decode(errors='replace').strip()}")
        crc, size = 0, 0
        with open(part, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                crc, size = zlib.crc32(block, crc), size + len(block)
        if size != info.file_size or crc != info.CRC:
            record_path.unlink()
            raise RuntimeError(f"{info.filename}: unpacked {size} bytes with CRC {crc:08x}, the zip says "
                               f"{info.file_size} bytes, CRC {info.CRC:08x}. Deleted to download again")
        os.replace(part, dest)
    finally:
        if part.exists():
            part.unlink()
    record_path.unlink()


def soh_plan(out_dir, hours_per_mic, mics, timeout):
    """[(Item for the member's record, ZipInfo, final path)] for the selected hours, audio and labels."""
    plan = []
    for record, name in SOH_PARTS:
        url = SOH_URL.format(record=record, name=urllib.request.quote(name))
        members = remote_members(url, timeout)
        by_name = {info.filename: (info, end) for info, end in members}
        audio = {}
        for info, end in members:
            parts = info.filename.split("/")
            if len(parts) == 4 and parts[1] == "Audio" and parts[3].lower().endswith("_modified.wav"):
                audio.setdefault(parts[2], []).append(info)
        for mic, infos in sorted(audio.items()):
            if mics and mic not in mics:
                continue
            infos.sort(key=lambda i: i.filename)   # by date, then hour
            for info in pick_spread(infos, hours_per_mic, second_of_pair=int(mic) % 2 == 0):
                stamp = info.filename.rsplit("/", 1)[1][:len("YYYYMMDD_HHMMSS")]
                labels_name = f"{info.filename.split('/Audio/')[0]}/Annotations/{mic}/{stamp}_light.json"
                for member, dest in ((info, out_dir / "sounds_of_home" / mic / f"{stamp}.wav"),
                                     (by_name.get(labels_name, (None,))[0],
                                      out_dir / "sounds_of_home" / mic / f"{stamp}_light.json")):
                    if member is None:
                        continue
                    end = by_name[member.filename][1]
                    item = dl.Item(url, dest.with_name(dest.name + ".zipmember"), optional=dest.suffix == ".json",
                                   size=end - member.header_offset, start=member.header_offset, ranged=True)
                    plan.append((item, member, dest))
    return plan


def pick_spread(infos, n, second_of_pair):
    """n of a mic's hours, evenly spaced through its week, which is sorted by day then hour, so
    the picks move through the day from one day to the next. The second mic of a home is offset
    by half a step, so the two rooms of a home don't get the same hours."""
    if n >= len(infos):
        return infos
    step = len(infos) / n
    offset = step / 2 if second_of_pair else 0
    return [infos[int(i * step + offset)] for i in range(n)]


# --- MUSAN and DEMAND --------------------------------------------------------

def extract_musan(archive, out_dir):
    """Extracts noise/ and music/ (and the top-level README / LICENSE) from the MUSAN archive,
    skipping speech/. Files already extracted with the right size are left alone."""
    kept = skipped = 0
    with tarfile.open(archive, "r:gz") as tar, tqdm(desc="Extracting MUSAN", unit=" files") as bar:
        for member in tar:
            name = member.name.lstrip("./")
            if not member.isfile() or not (name.startswith(MUSAN_KEEP) or name in MUSAN_TOP_FILES):
                skipped += member.isfile()
                continue
            dest = out_dir / name
            if dest.exists() and dest.stat().st_size == member.size:
                kept += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            with tar.extractfile(member) as src, open(part, "wb") as f:
                shutil.copyfileobj(src, f, 1 << 20)
            os.replace(part, dest)
            kept += 1
            bar.update()
    if not kept:
        raise RuntimeError(f"{archive.name}: no files under {' or '.join(MUSAN_KEEP)}: is the layout different?")
    print(f"MUSAN: {kept} files in noise/ and music/, {skipped} others (speech) skipped")


def extract_demand(archive, env, channels, out_dir):
    with zipfile.ZipFile(archive) as zf:
        for channel in channels:
            matches = [n for n in zf.namelist() if n.lower().endswith(f"/{channel}.wav") or n.lower() == f"{channel}.wav"]
            if not matches:
                raise RuntimeError(f"{archive.name}: no {channel}.wav in {zf.namelist()[:5]}...")
            dest = out_dir / "demand" / env / f"{channel}.wav"
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            with zf.open(matches[0]) as src, open(part, "wb") as f:
                shutil.copyfileobj(src, f, 1 << 20)
            os.replace(part, dest)


def segmented_jobs(dest, urls, size, segments, finish_whole, migrate=True):
    """[(Item, finish)] downloading dest in byte ranges (one Item each, spread over the mirror
    urls) that are joined into dest once all are in, then finish_whole() runs. If dest is
    already whole, a single job that only finishes. A .part from a single-stream download of
    dest is the start of the first range, so it isn't lost (renamed only if migrate)."""
    bounds = [round(k * size / segments) for k in range(segments + 1)]
    items = [dl.Item(urls[k % len(urls)], dest.with_name(f"{dest.name}.seg{k}"), optional=False,
                     size=bounds[k + 1] - bounds[k], start=bounds[k], ranged=True) for k in range(segments)]
    if dest.exists():
        # Joined already (maybe stopped before tidying up): only what comes after the download
        for leftover in dest.parent.glob(f"{dest.name}.seg*"):
            if migrate:
                leftover.unlink()
        return [(dl.Item(urls[0], dest, optional=False, size=size), finish_whole)]
    old = dest.with_name(dest.name + ".part")
    first = items[0]
    if migrate and old.exists() and not first.part.exists() and not first.dest.exists():
        if old.stat().st_size > first.size:
            os.truncate(old, first.size)
        os.replace(old, first.part)
    lock = threading.Lock()

    def finish():
        # Each range calls this when it's in; the last one joins them (the lock stops two doing it)
        with lock:
            if dest.exists() or not all(i.dest.exists() for i in items):
                return
            part = dest.with_name(dest.name + ".joining")
            with open(part, "wb") as out:
                for item in items:
                    with open(item.dest, "rb") as f:
                        shutil.copyfileobj(f, out, 1 << 24)
            if part.stat().st_size != size:
                part.unlink()
                raise RuntimeError(f"{dest.name}: joined ranges are {part.stat().st_size} bytes, expected {size}")
            os.replace(part, dest)
            for item in items:
                item.dest.unlink()
            finish_whole()

    return [(item, finish) for item in items]


# --- Manifest ----------------------------------------------------------------

# The same performer spelt two ways in MUSAN's annotations
MUSAN_ARTIST_ALIASES = {"kevin_macload": "kevin_macleod"}


def musan_music_annotations(music_dir):
    """{file stem: (vocals 'yes' / 'no', artist)} from the ANNOTATIONS files of MUSAN's music
    folders, whose rows are: <file> <genres> <vocals Y/N> <artist (performer)> [<composer>]"""
    music = {}
    for annotations in music_dir.glob("*/ANNOTATIONS"):
        for line in annotations.read_text(errors="replace").splitlines():
            fields = line.split()
            if len(fields) >= 4 and fields[2] in ("Y", "N"):
                artist = fields[3].lower()
                music[fields[0]] = ("yes" if fields[2] == "Y" else "no", MUSAN_ARTIST_ALIASES.get(artist, artist))
    return music


def musan_recorders(license_path):
    """{file stem: recorder} from a MUSAN noise LICENSE whose entries (separated by ==== lines)
    name the file and say 'Recorded by <name>' (Sound Bible). Free Sound's has no names."""
    recorders = {}
    if not license_path.exists():
        return recorders
    for entry in re.split(r"\n=+\n", license_path.read_text(errors="replace")):
        name = re.search(r"^Recorded by (.+)$", entry, re.M)
        if name:
            recorder = re.sub(r"\W+", "_", name.group(1).strip().lower())
            for stem in re.findall(r"^(noise-[\w-]+)\s*$", entry, re.M):
                recorders[stem] = recorder
    return recorders


def musan_group(kind, sub, stem, music, recorders):
    """Split group of a MUSAN file: music by artist (a performer's tracks sound alike), Sound
    Bible noise by recorder, other noise by file. Files without that information: by file."""
    if kind == "music" and stem in music:
        return f"musan/music/{music[stem][1]}"
    if kind == "noise" and stem in recorders:
        return f"musan/noise/{sub}/{recorders[stem]}"
    return f"musan/{stem}"


def write_manifest(out_dir):
    rows = []
    music = musan_music_annotations(out_dir / "musan" / "music")
    recorders = {}
    for license_path in (out_dir / "musan" / "noise").glob("*/LICENSE"):
        recorders.update(musan_recorders(license_path))
    for path in sorted(out_dir.rglob("*.wav")):
        rel = path.relative_to(out_dir)
        parts = rel.parts
        if parts[0] == "musan" and len(parts) == 4:
            kind = parts[1]
            row = {"source": "musan", "category": f"musan/{kind}/{parts[2]}",
                   "group": musan_group(kind, parts[2], path.stem, music, recorders),
                   "vocals": music.get(path.stem, ("unknown",))[0] if kind == "music" else "no"}
        elif parts[0] == "demand" and len(parts) == 3:
            row = {"source": "demand", "category": f"demand/{parts[1]}", "group": f"demand/{parts[1]}", "vocals": "no"}
        elif parts[0] == "sounds_of_home" and len(parts) == 3:
            home = (int(parts[1]) + 1) // 2
            row = {"source": "soh", "category": f"soh/mic{parts[1]}", "group": f"soh/home{home}", "vocals": "no"}
        else:
            continue
        info = sf.info(str(path))
        rows.append({"source": row["source"], "file": str(rel), "category": row["category"], "group": row["group"],
                     "vocals": row["vocals"], "duration_s": f"{info.duration:.2f}", "sample_rate": info.samplerate,
                     "licence": LICENCES[row["source"]]})
    with open(out_dir / MANIFEST_NAME, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "file", "category", "group", "vocals", "duration_s",
                                               "sample_rate", "licence"])
        writer.writeheader()
        writer.writerows(rows)

    totals = {}
    for r in rows:
        key = r["source"] if r["source"] != "musan" else f"musan {r['category'].split('/')[1]}" + (
            f" (vocals {r['vocals']})" if "music" in r["category"] else "")
        files, hours = totals.get(key, (0, 0.0))
        totals[key] = (files + 1, hours + float(r["duration_s"]) / 3600)
    print(f"\nWrote {out_dir / MANIFEST_NAME}: {len(rows)} files")
    for key, (files, hours) in sorted(totals.items()):
        print(f"  {key:28s} {files:5d} files  {hours:6.1f} h")


# --- Main --------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Download non-speech background noise (MUSAN, DEMAND, Sounds of Home).")
    parser.add_argument("-o", "--out", default=str(DEFAULT_OUT_DIR), help=f"output dir (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=SOURCES, help="datasets to fetch (default: all)")
    parser.add_argument("--musan-mirror", nargs="+", choices=list(MUSAN_MIRRORS), default=["elda"],
                        help="mirrors MUSAN's ranges are spread over (default: elda, the EU mirror)")
    parser.add_argument("--musan-segments", type=int, default=MUSAN_SEGMENTS,
                        help=f"byte ranges MUSAN is downloaded in, in parallel (default: {MUSAN_SEGMENTS})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"files downloaded at once, from all sources together (default: {DEFAULT_WORKERS})")
    parser.add_argument("--demand-envs", nargs="+", choices=sorted(DEMAND_ENVS), default=sorted(DEMAND_ENVS),
                        help="DEMAND environments (default: all without conversation)")
    parser.add_argument("--demand-channels", nargs="+", default=DEMAND_CHANNELS,
                        help="DEMAND channels to keep, e.g. ch01 ch09 (default: ch01)")
    parser.add_argument("--soh-hours-per-mic", type=int, default=SOH_HOURS_PER_MIC,
                        help=f"one-hour Sounds of Home files per mic, 14 mics (default: {SOH_HOURS_PER_MIC})")
    parser.add_argument("--soh-mics", nargs="+", help="only these Sounds of Home mic folders, e.g. 01 02 (default: all)")
    parser.add_argument("--delete-archives", action="store_true",
                        help="delete musan.tar.gz and the DEMAND zips once extracted")
    parser.add_argument("--retries", type=int, default=10, help="retries per file without progress (default: 10)")
    parser.add_argument("--timeout", type=float, default=60, help="network timeout per read, s (default: 60)")
    parser.add_argument("--dry-run", action="store_true", help="only print what would be downloaded and its size")
    parser.add_argument("--manifest-only", action="store_true", help="only rewrite manifest.csv from what's on disk")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = pathlib.Path(args.out)
    if args.manifest_only:
        write_manifest(out_dir)
        return
    if "soh" in args.sources and shutil.which("unzip") is None:
        sys.exit("Sounds of Home needs the unzip command (with Deflate64 support, as macOS's has)")
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = open(out_dir / LOCK_NAME, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"Another download_noise.py is already running on {out_dir}: stop it first (Ctrl-C), "
                 f"then run this again, and it carries on where that one stopped.")

    # What to fetch: (Item, what to do once it's downloaded)
    jobs = []
    if "musan" in args.sources:
        archive = out_dir / "musan.tar.gz"
        marker = out_dir / "musan" / ".extracted"
        if not marker.exists():
            def finish_musan(a=archive, m=marker):
                extract_musan(a, out_dir)
                m.touch()
                if args.delete_archives:
                    a.unlink()
            urls = [MUSAN_MIRRORS[m] for m in args.musan_mirror]
            jobs += [("musan", item, finish) for item, finish in
                     segmented_jobs(archive, urls, MUSAN_SIZE, args.musan_segments, finish_musan,
                                    migrate=not args.dry_run)]
    if "demand" in args.sources:
        for env in args.demand_envs:
            if all((out_dir / "demand" / env / f"{c}.wav").exists() for c in args.demand_channels):
                continue
            archive = out_dir / "demand_zips" / f"{env}_16k.zip"
            jobs.append(("demand", dl.Item(DEMAND_URL.format(env=env), archive, optional=False),
                         lambda a=archive, e=env: (extract_demand(a, e, args.demand_channels, out_dir),
                                                   args.delete_archives and a.unlink())))
    if "soh" in args.sources:
        print("Reading the Sounds of Home zip indexes...")
        for item, info, dest in soh_plan(out_dir, args.soh_hours_per_mic, args.soh_mics, args.timeout):
            if not dest.exists():
                jobs.append(("soh labels" if item.optional else "soh audio", item,
                             lambda r=item.dest, n=item.size, i=info, d=dest: unpack_member(r, n, i, d)))

    # Sizes for the progress bar (Sounds of Home items already have theirs)
    for _, item, _ in jobs:
        if item.size is None:
            try:
                item.size = dl.head_size(item, args.timeout, dl.HEAD_RETRIES)
            except Exception as exc:
                tqdm.write(f"warning: no size for {item.dest.name} ({exc}), downloading it anyway")
    on_disk = sum(i.dest.stat().st_size for _, i, _ in jobs if i.dest.exists())
    partial = sum(i.part.stat().st_size for _, i, _ in jobs if i.part.exists())
    total = sum(i.size or 0 for _, i, _ in jobs)
    counts = {}
    for key, item, _ in jobs:
        n, size = counts.get(key, (0, 0))
        counts[key] = (n + 1, size + (item.size or 0))
    print(f"{len(jobs)} files to fetch, {dl.format_bytes(total - on_disk - partial)} still to download"
          + (f" ({dl.format_bytes(on_disk + partial)} already on disk)" if on_disk + partial else ""))
    for key, (n, size) in sorted(counts.items()):
        print(f"  {key:12s} {n:4d} files  {dl.format_bytes(size)}")
    if args.dry_run or not jobs:
        if not jobs:
            write_manifest(out_dir)
        return

    failed = run_parallel(jobs, args, SharedBar(tqdm(total=total, initial=on_disk + partial, unit="B", unit_scale=True,
                                                     unit_divisor=1024, desc="Total", dynamic_ncols=True)))

    write_manifest(out_dir)
    print(f"\nFetched {len(jobs) - len(failed)} of {len(jobs)}")
    if failed:
        print("Failed (re-run the same command to retry):")
        for item in failed:
            print(f"  {item.dest.name}")
        sys.exit(1)


class SharedBar:
    """The total progress bar, updated from several download threads."""

    def __init__(self, bar):
        self.bar, self.lock = bar, threading.Lock()

    def update(self, n):
        with self.lock:
            self.bar.update(n)

    def close(self):
        self.bar.close()


def run_parallel(jobs, args, total_bar):
    """Downloads and finishes the jobs with args.workers threads, taking them in order (MUSAN's
    big file first). Returns the failed Items. Ctrl-C stops at once: the threads are daemons,
    and everything they write is resumable (.part files, extraction that's redone)."""
    queue, failed, queue_lock = list(jobs), [], threading.Lock()

    def worker():
        while True:
            with queue_lock:
                if not queue:
                    return
                _, item, finish = queue.pop(0)
            ok = item.dest.exists() or dl.download(item, args.timeout, args.retries, total_bar)
            if ok:
                try:
                    finish()
                    continue
                except Exception as exc:
                    tqdm.write(f"FAILED {item.dest.name}: {exc}")
            failed.append(item)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, min(args.workers, len(jobs))))]
    for thread in threads:
        thread.start()
    try:
        while any(thread.is_alive() for thread in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        total_bar.close()
        sys.exit("\nInterrupted. Partial files are kept: re-run the same command to resume.")
    total_bar.close()
    return failed


if __name__ == "__main__":
    main()
