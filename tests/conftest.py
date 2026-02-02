"""Test config: ensure repository root is on sys.path so `import src` works.

This is a small helper to avoid requiring an editable install of the package
when running tests from the repository root.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
