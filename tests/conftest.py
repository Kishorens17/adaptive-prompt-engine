"""
conftest.py

Ensures the project root is on sys.path so tests can `import` the
package modules (classifier, strategies, handlers, factory, evaluator,
llm) using the same absolute-import style as main.py, regardless of
which directory pytest is invoked from.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
