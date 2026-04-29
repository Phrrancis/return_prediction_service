# conftest.py — makes the repo root available on sys.path so that
# `from backend.xxx import ...` works without setting PYTHONPATH manually.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
