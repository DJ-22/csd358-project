"""Make `import hinglishkey` work when running `pytest` from the repo root (src/ layout)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
