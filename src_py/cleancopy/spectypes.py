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


# These string values are not part of the spec -- they're purely an
# implementation detail -- hence Enum
class ListType(Enum):
    ORDERED = 'ORDERED'
    UNORDERED = 'UNORDERED'


# These string values are literals that are part of the spec, hence StrEnum
class EmbedFallbackBehavior(StrEnum):
    VISIBLE_PREFORMATTED = 'VISIBLE_PREFORMATTED'
    HIDDEN = 'HIDDEN'


class MetadataMagics(Enum):
    # Note: the field names here need to match the field names in ast
    id_ = '__id__'
    embed = '__embed__'
    target = '__target__'
    icu_1 = '__icu-1__'
    fmt = '__fmt__'
    sugared = '__sugared__'
    is_doc_metadata = '__doc_meta__'
    fallback = '__fallback__'
