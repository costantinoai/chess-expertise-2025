"""
One-time data migration helper: copy legacy eyetracking derivatives to
the canonical location BIDS/derivatives/eye-tracking.

Source defaults to BIDS/derivatives/bidsmreye if exists, or can be provided
via --src argument. Target is CONFIG['BIDS_EYETRACK'].

This script does not overwrite existing files by default; pass --overwrite to
allow overwriting.
"""

import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from common import CONFIG


def copy_tree(src: Path, dst: Path, overwrite: bool = False):
    src = Path(src); dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        outdir = dst / rel
        outdir.mkdir(parents=True, exist_ok=True)
        for f in files:
            sp = Path(root) / f
            dp = outdir / f
            if dp.exists() and not overwrite:
                continue
            shutil.copy2(sp, dp)
# Configuration (edit these when running from IDE)
SRC = Path(CONFIG['BIDS_ROOT']) / 'derivatives' / 'bidsmreye'
OVERWRITE = False

src = SRC
dst = Path(CONFIG['BIDS_EYETRACK'])
print(f"Copying eyetracking derivatives\n  from: {src}\n    to: {dst}\n  overwrite: {OVERWRITE}")
copy_tree(src, dst, overwrite=OVERWRITE)
print("Done.")
