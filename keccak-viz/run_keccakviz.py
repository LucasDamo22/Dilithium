#!/usr/bin/env python3
"""Entry point for the frozen builds (PyInstaller runs this as a top-level script)."""

import sys

from keccakviz.ui.app import main

if __name__ == "__main__":
    sys.exit(main())
