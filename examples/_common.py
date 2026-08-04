# -*- coding:utf-8 -*-
"""Lets the examples run straight from a clone, without ``pip install``.

Importing this module puts the repo root on ``sys.path`` so ``import carmen``
resolves. If you installed the package, you do not need it.
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
