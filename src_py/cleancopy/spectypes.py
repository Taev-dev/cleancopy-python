"""This module contains additional types, constants, and literals
associated with the language spec itself.
"""
from __future__ import annotations

from enum import Enum
from enum import StrEnum


class InlineFormatting(StrEnum):
    PRE = '__pre__'
    UNDERLINE = '__underline__'
    STRONG = '__strong__'
    EMPHASIS = '__emphasis__'
    STRIKE = '__strike__'
    QUOTE = '__quote__'


class BlockFormatting(StrEnum):
    QUOTE = '__quote__'


# These string values are not part of the spec -- they're purely an
# implementation detail -- hence Enum
class ListType(Enum):
    ORDERED = 'ORDERED'
    UNORDERED = 'UNORDERED'


# These string values are literals that are part of the spec, hence StrEnum
class EmbedFallbackBehavior(StrEnum):
    VISIBLE_PREFORMATTED = 'VISIBLE_PREFORMATTED'
    HIDDEN = 'HIDDEN'


class BlockMetadataMagic(Enum):
    # Note: the field **names** here need to match the field names in ast, and
    # the field **values** need to match the spec.
    crossref = '__crossref__'
    embed = '__embed__'
    target = '__target__'
    formatting = '__formatting__'
    citation = '__citation__'
    is_doc_metadata = '__doc_meta__'
    fallback = '__fallback__'
    source = '__source__'
    style_modifiers = '__style_modifiers__'
    semantic_modifiers = '__semantic_modifiers__'
    layout_modifiers = '__layout_modifiers__'


class InlineMetadataMagic(Enum):
    # Note: the field **names** here need to match the field names in ast, and
    # the field **values** need to match the spec.
    crossref = '__crossref__'
    target = '__target__'
    icu_1 = '__icu-1__'
    formatting = '__formatting__'
    citation = '__citation__'
    sugared = '__sugared__'
    style_modifiers = '__style_modifiers__'
    semantic_modifiers = '__semantic_modifiers__'
    layout_modifiers = '__layout_modifiers__'
