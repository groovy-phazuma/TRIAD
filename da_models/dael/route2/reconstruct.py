#!/usr/bin/env python3
"""Compatibility entrypoint for reconstruction training."""

import sys
from pathlib import Path

current_file = Path(__file__).resolve()
triad_root = current_file.parents[3]
if str(triad_root) not in sys.path:
    sys.path.append(str(triad_root))

from da_models.dael.route2.reconstruction import main


if __name__ == "__main__":
    main()
