from __future__ import annotations

from docnote import DocnoteConfig
from docnote import MarkupLang

from cleancopy._cst.abstractifier import Abstractifier
from cleancopy._cst.treesitter_ import parse

__all__ = [
    'Abstractifier',
    'parse'
]

DOCNOTE_CONFIG = DocnoteConfig(markup_lang=MarkupLang.CLEANCOPY)
