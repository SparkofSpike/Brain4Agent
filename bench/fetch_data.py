"""Download benchmark data.

Kept as a script rather than a runtime fetch so a scored run is reproducible:
the bytes on disk are the bytes that were scored.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from bench.loader import DATA_DIR, ORACLE, S_CLEANED

BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

FILES = {
    ORACLE: f"{BASE}/{ORACLE}",
    S_CLEANED: f"{BASE}/{S_CLEANED}",
}


def fetch(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / name
    if dest.exists():
        print(f"{name}: already present ({dest.stat().st_size / 1_048_576:.1f} MB)")
        return dest

    url = FILES[name]
    print(f"{name}: downloading from {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as resp, tmp.open("wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1_048_576:.1f}/{total / 1_048_576:.1f} MB", end="")
    print()
    tmp.replace(dest)
    print(f"{name}: ok ({dest.stat().st_size / 1_048_576:.1f} MB)")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "names",
        nargs="*",
        default=[ORACLE],
        help=f"which files to fetch; choose from {sorted(FILES)}",
    )
    args = ap.parse_args()
    for name in args.names:
        if name not in FILES:
            raise SystemExit(f"unknown dataset {name!r}; choose from {sorted(FILES)}")
        fetch(name)


if __name__ == "__main__":
    main()
