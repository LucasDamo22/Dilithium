#!/usr/bin/env python3
"""Build a standalone keccakviz executable for the machine it runs on.

    python build.py              # one folder in dist/keccakviz (starts fastest)
    python build.py --onefile    # a single executable in dist/

There is no cross-compiling: run this on Linux for the Linux build and on
Windows for the Windows build (the GitHub workflow in .github/workflows does
both).  The result needs no Python on the target machine.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onefile", action="store_true", help="single executable instead of a folder")
    ap.add_argument("--clean", action="store_true", help="remove build/ and dist/ first")
    args = ap.parse_args()

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is missing:  pip install pyinstaller", file=sys.stderr)
        return 2

    if args.clean:
        for d in ("build", "dist"):
            shutil.rmtree(ROOT / d, ignore_errors=True)

    env = dict(os.environ, KECCAKVIZ_ONEFILE="1" if args.onefile else "0")
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", str(ROOT / "keccakviz.spec")]
    print(" ".join(cmd))
    rc = subprocess.call(cmd, cwd=ROOT, env=env)
    if rc != 0:
        return rc

    exe = ROOT / "dist" / ("keccakviz" if args.onefile else "keccakviz/keccakviz")
    if sys.platform == "win32":
        exe = exe.with_suffix(".exe")
    size = sum(f.stat().st_size for f in (ROOT / "dist").rglob("*") if f.is_file())
    print(f"\nbuilt {exe}  ({size / 2**20:.0f} MB in dist/)")
    print("smoke test:  " + str(exe) + " -m abc --screenshot shot.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
