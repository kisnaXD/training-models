#!/usr/bin/env python3
"""One-file launcher for full CMD Polymarket interface."""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CLI_DIR = os.path.join(ROOT, "PolymarketCLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from polymarket_cmd_app import main


if __name__ == "__main__":
    main()

