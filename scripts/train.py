"""Entry point.

  RTX 3060 (Windows/Linux, single GPU):  python scripts/train.py --config configs/rtx3060.yaml --arch kda_dsa
  1 x H100:                              python scripts/train.py --config configs/h100x1.yaml --arch dense
  8 x H100:   torchrun --standalone --nproc_per_node 8 scripts/train.py --config configs/h100x8.yaml --arch dsa
Wrap any of them with scripts/supervise.py for automatic crash recovery.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmarch.train.trainer import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
