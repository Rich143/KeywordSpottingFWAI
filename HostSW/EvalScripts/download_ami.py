"""
Download AMI meeting corpus audio from the wget script that the AMI download
page generates (https://groups.inf.ed.ac.uk/ami/download), with resume,
retries and a progress bar.

The generated script lists every selected channel of every meeting (all 16
array mics come to ~163 GB). The eval set only needs one far-field mic per
meeting, so by default only Array1-01 is downloaded (169 files, ~11 GB).

Resuming:
  - each file downloads to <name>.part and is renamed once it has the size
    the server reported, so a finished file is never partial
  - a finished file is skipped; a .part is continued with an HTTP Range
    request, so Ctrl-C (or a dropped connection) loses nothing: re-run the
    same command to carry on
  - timeouts, connection errors and 5xx / 429 responses are retried with
    exponential backoff; the retry count resets whenever data was received

Output layout matches the wget script: <out>/amicorpus/<meeting>/audio/<file>.wav,
plus the build manifest and licence in <out>.

Usage (from HostSW/):
    python EvalScripts/download_ami.py -s eval_recordings_public/amiBuild-152242-Thu-Oct-22-2026.wget.sh --dry-run
    python EvalScripts/download_ami.py -s eval_recordings_public/amiBuild-152242-Thu-Oct-22-2026.wget.sh
    python EvalScripts/download_ami.py -s <script> --channels Array1-01 Array2-01 --meetings 'ES2002*'
"""

import argparse
import concurrent.futures
import fnmatch
import http.client
import os
import pathlib
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from tqdm import tqdm

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
HOSTSW_DIR = SCRIPT_DIR.parent
DEFAULT_OUT_DIR = HOSTSW_DIR / "dataset_public" / "ami"
DEFAULT_CHANNELS = ["Array1-01"]

CHUNK_BYTES = 1 << 20
MAX_BACKOFF_S = 60
HEAD_WORKERS = 4
HEAD_RETRIES = 3
USER_AGENT = "download_ami.py"
# The AMI mirror answers some requests with a transient 403 under load (the same URL
# works a moment later), so 403 is retried like 429 and 5xx
RETRYABLE_HTTP_CODES = {403, 408, 429}

# "wget    -P amicorpus/ES2002a/audio https://.../ES2002a.Array1-01.wav" or "wget https://.../x.txt"
WGET_LINE = re.compile(r"^\s*wget\s+(?:-P\s+(\S+)\s+)?(\S+)\s*$")
# ES2002a.Array1-01.wav -> meeting ES2002a, channel Array1-01
AUDIO_NAME = re.compile(r"^(?P<meeting>[^.]+)\.(?P<channel>.+)\.wav$")


class RetryableError(Exception):
    """A download attempt failed in a way that's worth retrying."""


@dataclass
class Item:
    url: str
    dest: pathlib.Path
    optional: bool        # extras (manifest, licence): a failure is only a warning
    size: int | None = None

    @property
    def part(self):
        return self.dest.with_name(self.dest.name + ".part")


def parse_wget_script(path, out_dir):
    items = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = WGET_LINE.match(line)
        if not m:
            sys.exit(f"{path}:{line_no}: can't parse '{line}'")
        subdir, url = m.groups()
        name = url.rsplit("/", 1)[-1]
        dest = out_dir / subdir / name if subdir else out_dir / name
        items.append(Item(url, dest, optional=not name.endswith(".wav")))
    return items


def select(items, channels, meetings):
    """Keeps the audio files matching the channel and meeting filters, plus all extras."""
    kept = []
    for item in items:
        m = AUDIO_NAME.match(item.dest.name)
        if item.optional or not m:
            kept.append(item)
            continue
        if channels != ["all"] and m["channel"] not in channels:
            continue
        if meetings and not any(fnmatch.fnmatch(m["meeting"], p) for p in meetings):
            continue
        kept.append(item)
    return kept


def request(url, timeout, method="GET", offset=0):
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers, method=method), timeout=timeout)


def is_retryable(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500 or exc.code in RETRYABLE_HTTP_CODES
    return isinstance(exc, (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError,
                            http.client.HTTPException))


def head_size(item, timeout, retries):
    """Content-Length from a HEAD request (None if the server doesn't say)."""
    for attempt in range(retries + 1):
        try:
            with request(item.url, timeout, method="HEAD") as resp:
                length = resp.headers.get("Content-Length")
                return int(length) if length is not None else None
        except Exception as exc:
            if not is_retryable(exc) or attempt == retries:
                raise
            time.sleep(min(2 * 2 ** attempt, MAX_BACKOFF_S))


def download_attempt(item, timeout, file_bar, total_bar):
    """One attempt: continues item.part from its current size. Returns the bytes received.
    Raises RetryableError for network trouble, anything else for permanent failures."""
    offset = item.part.stat().st_size if item.part.exists() else 0
    if item.size is not None and offset == item.size:
        return 0  # .part already complete, e.g. interrupted just before the rename

    received = 0
    try:
        with request(item.url, timeout, offset=offset) as resp:
            if offset and resp.status != 206:
                # Server ignored the Range header and sent the whole file: start over
                total_bar.update(-offset)
                file_bar.reset(total=item.size)
                offset = 0
            with open(item.part, "ab" if offset else "wb") as f:
                while True:
                    chunk = resp.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    f.write(chunk)
                    received += len(chunk)
                    file_bar.update(len(chunk))
                    total_bar.update(len(chunk))
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and item.size is not None and offset == item.size:
            return received  # range starts at the end: the .part is already complete
        if exc.code == 416:
            # .part is bigger than the file (or the file changed): start over
            total_bar.update(-offset)
            file_bar.reset(total=item.size)
            item.part.unlink()
            raise RetryableError(f"HTTP 416 with {offset} bytes on disk, restarting") from exc
        if is_retryable(exc):
            raise RetryableError(f"HTTP {exc.code}") from exc
        raise
    except Exception as exc:
        if is_retryable(exc):
            raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
        raise
    return received


def download(item, timeout, retries, total_bar):
    """Downloads item.url to item.dest, resuming item.part. Returns True on success."""
    item.dest.parent.mkdir(parents=True, exist_ok=True)
    offset = item.part.stat().st_size if item.part.exists() else 0
    file_bar = tqdm(total=item.size, initial=offset, unit="B", unit_scale=True, unit_divisor=1024,
                    desc=item.dest.name, leave=False, dynamic_ncols=True)
    attempt = 0
    try:
        while True:
            try:
                received = download_attempt(item, timeout, file_bar, total_bar)
            except RetryableError as exc:
                # Progress since the last failure resets the count, so a long file with the
                # odd timeout doesn't run out of retries
                attempt = 0 if file_bar.n > offset else attempt + 1
                offset = file_bar.n
                if attempt > retries:
                    tqdm.write(f"FAILED {item.dest.name}: {exc} (gave up after {retries} retries)")
                    return False
                delay = min(2 * 2 ** max(attempt - 1, 0), MAX_BACKOFF_S)
                tqdm.write(f"  {item.dest.name}: {exc}, retrying in {delay}s")
                time.sleep(delay)
                continue

            size_on_disk = item.part.stat().st_size if item.part.exists() else 0
            if item.size is not None and size_on_disk != item.size:
                # Connection closed early without an error
                attempt += 1
                offset = size_on_disk
                if attempt > retries:
                    tqdm.write(f"FAILED {item.dest.name}: {size_on_disk} of {item.size} bytes")
                    return False
                tqdm.write(f"  {item.dest.name}: stopped at {size_on_disk} of {item.size} bytes, resuming")
                continue
            if received == 0 and not item.part.exists():
                tqdm.write(f"FAILED {item.dest.name}: no data received")
                return False
            os.replace(item.part, item.dest)
            return True
    except urllib.error.HTTPError as exc:
        tqdm.write(f"FAILED {item.dest.name}: HTTP {exc.code} {exc.reason}")
        return False
    finally:
        file_bar.close()


def format_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download AMI audio from the download page's wget script, resumably, with a progress bar."
    )
    parser.add_argument("-s", "--script", required=True, help="amiBuild-*.wget.sh from the AMI download page")
    parser.add_argument("-o", "--out", default=str(DEFAULT_OUT_DIR), help=f"output dir (default: {DEFAULT_OUT_DIR})")
    parser.add_argument(
        "--channels", nargs="+", default=DEFAULT_CHANNELS,
        help="audio channels to keep, e.g. Array1-01 Mix-Headset, or 'all' (default: Array1-01)",
    )
    parser.add_argument("--meetings", nargs="+", help="meeting ids or globs to keep, e.g. ES2002a 'IS1*' (default: all)")
    parser.add_argument("--retries", type=int, default=10, help="retries per file without progress (default: 10)")
    parser.add_argument("--timeout", type=float, default=60, help="network timeout per read, s (default: 60)")
    parser.add_argument("--dry-run", action="store_true", help="only print what would be downloaded and its size")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = pathlib.Path(args.out)
    items = select(parse_wget_script(pathlib.Path(args.script), out_dir), args.channels, args.meetings)
    audio = [i for i in items if not i.optional]
    if not audio:
        sys.exit("no audio files left after the --channels / --meetings filters")

    done = [i for i in items if i.dest.exists()]
    todo = [i for i in items if not i.dest.exists()]
    print(f"{len(audio)} audio files selected ({len(items) - len(audio)} extras), "
          f"{len(done)} already downloaded, {len(todo)} to go")

    # Sizes for the overall progress bar (and to know when a file is complete). A file whose
    # size can't be found is still downloaded, just without a size check.
    with concurrent.futures.ThreadPoolExecutor(HEAD_WORKERS) as pool:
        futures = {pool.submit(head_size, i, args.timeout, HEAD_RETRIES): i for i in todo}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                        desc="Checking sizes", leave=False):
            item = futures[fut]
            try:
                item.size = fut.result()
            except Exception as exc:
                tqdm.write(f"warning: no size for {item.dest.name} ({exc}), downloading it anyway")

    done_bytes = sum(i.dest.stat().st_size for i in done)
    partial_bytes = sum(i.part.stat().st_size for i in todo if i.part.exists())
    todo_bytes = sum(i.size or 0 for i in todo)
    unknown = sum(i.size is None for i in todo)
    print(f"Already on disk: {format_bytes(done_bytes)} done + {format_bytes(partial_bytes)} partial. "
          f"Still to download: {format_bytes(todo_bytes - partial_bytes)}"
          + (f" (+{unknown} files of unknown size)" if unknown else ""))
    if args.dry_run:
        return

    failed = []
    total_bar = tqdm(total=done_bytes + todo_bytes, initial=done_bytes + partial_bytes, unit="B",
                     unit_scale=True, unit_divisor=1024, desc="Total", dynamic_ncols=True)
    try:
        for item in sorted(todo, key=lambda i: str(i.dest)):
            if not download(item, args.timeout, args.retries, total_bar) and not item.optional:
                failed.append(item)
    except KeyboardInterrupt:
        total_bar.close()
        sys.exit("\nInterrupted. Partial files are kept as .part: re-run the same command to resume.")
    total_bar.close()

    n_ok = len(todo) - len(failed)
    print(f"\nDownloaded {n_ok}, already present {len(done)}, failed {len(failed)}")
    if failed:
        print("Failed (re-run the same command to retry):")
        for item in failed:
            print(f"  {item.url}")
        sys.exit(1)


if __name__ == "__main__":
    main()
