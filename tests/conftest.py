"""
tests/conftest.py

Додає корінь проекту та src/ у sys.path, щоб `import config.settings`
і `import bot.risk.risk_manager` працювали незалежно від того, звідки
запущено pytest. Тимчасове рішення до появи editable-інсталяції пакета
через pyproject.toml (`pip install -e .`), про яку йдеться в
docs/tech_stack.md.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

for _path in (PROJECT_ROOT, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))