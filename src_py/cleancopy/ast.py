# Note: this shadows the stdlib ast module, so... nothing here is going to be
# able to import it. Which should be fine.
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from dataclasses import fields
from decimal import Decimal
from typing import Any
from typing import ClassVar
from typing import Optional

from cleancopy.spectypes import EmbedFallbackBehavior
from cleancopy.spectypes import InlineFormatting
from cleancopy.spectypes import ListType
from cleancopy.spectypes import MetadataMagics

# Note: URIs are automatically converted to strings; they're only separate in
# the CST because sugared strings are a strict subset of strings and need to
# be differentiated from the other target types in the grammar itself
type MetadataTarget = (
    MetadataStr | MetadataMention | MetadataTag | MetadataVariable
    | MetadataReference)
# This is used to omit reserved but unused metadata keys from the results. We
# do this so people don't accidentally try to use them, and then later on
# we have to worry about compatibility issues if we need to introduce new
# fields.
METADATA_MAGIC_PATTERN = re.compile(r'^__.+__$')
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ASTNode:
    """Currently not really used for anything except for annotations,
    but at any rate: this is the base class for all AST nodes.
    """


@dataclass(kw_only=True)
class Document(ASTNode):
    # Note: this comes from the __doc_meta__ node
    title: Richtext | None
    metadata: BlockMetadata | None
    root: RichtextDocNode
    # TODO: add other helper methods, like searching by ID


@dataclass(kw_only=True)
class DocNode(ASTNode):
    """The base class for both richtext and embedded nodes."""
    title: Optional[Richtext] = None
    metadata: Optional[BlockMetadata] = None


@dataclass(kw_only=True)
class RichtextDocNode(DocNode):
    content: list[Paragraph | DocNode]

    def __getitem__(self, index: int) -> Paragraph | DocNode:
        return self.content[index]

    def __len__(self):
        return len(self.content)

    def __iter__(self):
        return iter(self.content)


@dataclass(kw_only=True)
class EmbeddingDocNode(DocNode):
    # Can be an explicit None if it's an empty node
    content: str | None


@dataclass(kw_only=True)
class Paragraph(ASTNode):
    """Paragraphs can contain multiple lines and/or multiple line types,
    but they **cannot** contain an empty line.
    """
    # Note: these are separate because lists have their own separate
    # RichtextContext for items
    content: list[Richtext | List_ | Annotation]

    def __bool__(self) -> bool:
        """Returns True if the paragraph has any content at all, whether
        displayed or not displayed.
        """
        return bool(self.content)


@dataclass(kw_only=True)
class List_(ASTNode):  # noqa: N801
    type_: ListType
    content: list[ListItem]


@dataclass(kw_only=True)
class ListItem(ASTNode):
    index: int | None
    content: list[Paragraph]


@dataclass(kw_only=True)
class Richtext[C: InlineMetadata | None](ASTNode):
    context: C
    content: list[str | Richtext]

    @property
    def has_display_content(self) -> bool:
        """Returns True if the paragraph contains any non-annotation
        lines. If all lines are annotations, or there are no lines,
        returns False.
        """
        if not self.content:
            return False
        else:
            return not all(
                isinstance(segment, Annotation) for segment in self.content)


@dataclass(kw_only=True)
class Annotation(ASTNode):
    content: str


# Note: can't be protocol due to missing intersection type
class _MemoizedFieldNames:
    _field_names: ClassVar[frozenset[str]]

    @staticmethod
    def memoize[C: type](for_cls: C) -> C:
        for_cls._field_names = frozenset(
            {field.name for field in fields(for_cls)})
        return for_cls


@_MemoizedFieldNames.memoize
@dataclass(kw_only=True)
class Metadata(ASTNode, _MemoizedFieldNames):

    # TODO: uri vs ref vs mention vs...
    target: Optional[MetadataTarget] = None

    def __post_init__(self):
        self._lookup: dict[str, MetadataValue | None] = {}
        self._payload: list

    # TODO: we should add items, .get, etc
    def __getitem__(self, key: str) -> MetadataValue:
        """Note: THIS is where we normalize things re: missings, None,
        etc.
        """
        maybe_result = self._lookup[key]

        if maybe_result is None:
            raise KeyError(
                'Metadata key was explicitly set to __missing__', key)

        else:
            return maybe_result

    def _add(self, line: MetadataAssignment | Annotation):
        """Call this when building the AST. Intended for use within the
        CST -> AST transition. So... semi-public. Public in the sense
        that it's used outside of this module, but not public in the
        sense that it's documented or intended for outside use.
        """
        self._payload.append(line)
        if isinstance(line, MetadataAssignment):
            key = line.key

            try:
                metadata_magic = MetadataMagics(key)

            except ValueError:
                if METADATA_MAGIC_PATTERN.match(key) is None:
                    self._lookup[key] = line.value
                else:
                    logger.warning('Ignoring reserved metadata key: %s', key)

            else:
                maybe_field_name = metadata_magic.name
                if maybe_field_name in self._field_names:
                    maybe_value = line.value
                    # Note that we want to extract the actual metadata value;
                    # the magics are all strongly-typed, so we don't need to
                    # worry about the MetadataValue container around them.
                    # (even though it's preserved within .as_declared)
                    if maybe_value is None:
                        value_to_use = None
                    else:
                        value_to_use = maybe_value.value

                    setattr(self, maybe_field_name, value_to_use)
                else:
                    logger.warning(
                        'Wrong metadata type for reserved key %s; ignoring',
                        key)


@_MemoizedFieldNames.memoize
@dataclass(kw_only=True)
class InlineMetadata(Metadata):
    """InlineMetadata is used only for, yknow, inline metadata.
    Note that all of the various formatting tags get sugared into
    inline metadatas.
    """
    icu_1: Optional[MetadataReference] = None
    fmt: Optional[InlineFormatting] = None
    sugared: Optional[bool] = None

    def __post_init__(self):
        super().__post_init__()
        self._payload: list[MetadataAssignment] = []

    @property
    def as_declared(self) -> tuple[MetadataAssignment, ...]:
        """This can be used to access the raw assignments, as declared,
        in their exact order. For inline metadata, this is only relevant
        if there are multiple metadata assignments using the same key
        in the same InlineMetadata instance.
        """
        # The only reason we do a tuple here is to make sure that the outside
        # world doesn't try to modify this!
        return tuple(self._payload)


@_MemoizedFieldNames.memoize
@dataclass(kw_only=True)
class BlockMetadata(Metadata):
    """BlockMetadata is used both for node metadata and for document
    metadata (which is itself just an empty node at the toplevel with
    a special magic key set).
    """
    is_doc_metadata: bool = False
    id_: Optional[MetadataStr] = None
    embed: Optional[MetadataStr] = None
    fallback: Optional[EmbedFallbackBehavior] = None

    def __post_init__(self):
        super().__post_init__()
        self._payload: list[MetadataAssignment | Annotation] = []

    @property
    def as_declared(self) -> tuple[MetadataAssignment | Annotation, ...]:
        """This can be used to access the raw assignments, as declared,
        in their exact order, **including any annotations**. This is
        useful both if you want access to the annotations, ^^or^^ if
        there are multiple metadata assignments using the same key in
        the same BlockMetadata instance.
        """
        # The only reason we do a tuple here is to make sure that the outside
        # world doesn't try to modify this!
        return tuple(self._payload)


@dataclass
class MetadataAssignment(ASTNode):
    key: str
    value: MetadataValue | None


@dataclass
class MetadataValue(ASTNode):
    # Note: needs to be overridden by subclasses
    value: Any


@dataclass
class MetadataStr(MetadataValue):
    value: str


@dataclass
class MetadataInt(MetadataValue):
    value: int


@dataclass
class MetadataDecimal(MetadataValue):
    value: Decimal


@dataclass
class MetadataBool(MetadataValue):
    value: bool


@dataclass
class MetadataNull(MetadataValue):
    value: None


@dataclass
class MetadataMention(MetadataValue):
    value: str


@dataclass
class MetadataTag(MetadataValue):
    value: str


@dataclass
class MetadataVariable(MetadataValue):
    value: str


@dataclass
class MetadataReference(MetadataValue):
    value: str
