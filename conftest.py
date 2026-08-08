"""Root conftest: ensure the repo root is importable so tests can `import model`."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
