# Note: this shadows the stdlib ast module, so... nothing here is going to be
# able to import it. Which should be fine.
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from decimal import Decimal
from typing import Annotated
from typing import Any
from typing import ClassVar

from docnote import Note

from cleancopy.spectypes import EmbedFallbackBehavior
from cleancopy.spectypes import InlineFormatting
from cleancopy.spectypes import ListType
from cleancopy.spectypes import MetadataMagics

# Note: URIs are automatically converted to strings; they're only separate in
# the CST because sugared strings are a strict subset of strings and need to
# be differentiated from the other target types in the grammar itself
type LinkTarget = (
    StrDataType | MentionDataType | TagDataType | VariableDataType
    | ReferenceDataType)
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
    title: RichtextInlineNode | None
    info: BlockNodeInfo | None
    root: RichtextBlockNode
    # TODO: add other helper methods, like searching by ID


@dataclass(kw_only=True)
class BlockNode(ASTNode):
    """The base class for both richtext and embedded nodes."""
    title: RichtextInlineNode | None = None
    info: BlockNodeInfo | None = None
    depth: int


@dataclass(kw_only=True)
class RichtextBlockNode(BlockNode):
    content: list[Paragraph | BlockNode]

    def __getitem__(self, index: int) -> Paragraph | BlockNode:
        return self.content[index]

    def __len__(self):
        return len(self.content)

    def __iter__(self):
        return iter(self.content)


@dataclass(kw_only=True)
class EmbeddingBlockNode(BlockNode):
    # Can be an explicit None if it's an empty node
    content: str | None


@dataclass(kw_only=True)
class Paragraph(ASTNode):
    """Paragraphs can contain multiple lines and/or multiple line types,
    but they **cannot** contain an empty line.
    """
    # Note: these are separate because lists have their own separate
    # info contexts for items
    content: list[RichtextInlineNode | List_ | Annotation]

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
class RichtextInlineNode(ASTNode):
    info: InlineNodeInfo | None
    content: list[str | RichtextInlineNode]

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
    """Annotations / comments: full lines beginning with ``##``.
    """
    content: str


# Note: can't be protocol due to missing intersection type
class _MemoizedFieldNames:
    _field_names: ClassVar[frozenset[str]]

    @staticmethod
    def memoize[C: type](for_cls: C) -> C:
        for_cls._field_names = frozenset(
            {dc_field.name for dc_field in fields(for_cls)})
        return for_cls


@_MemoizedFieldNames.memoize
@dataclass(kw_only=True)
class NodeInfo[T: MetadataAssignment | Annotation](
        ASTNode, _MemoizedFieldNames):

    target: LinkTarget | None = None
    metadata: Annotated[
            dict[str, DataType],
            Note('''Any normalized metadata values (ie, ``__missing__``
                removed, etc) that do not have special meaning within the
                cleancopy spec.''')
        ] = field(default_factory=dict)

    _payload: list[T] = field(
        default_factory=list, init=False, repr=False, compare=False)

    def _add(self, line: T):
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
                    # Note: this removes any explicit __missing__ value
                    if line.value is not None:
                        self.metadata[key] = line.value
                else:
                    logger.warning('Ignoring reserved metadata key: %s', key)

            else:
                maybe_field_name = metadata_magic.name
                if maybe_field_name in self._field_names:
                    maybe_value = line.value
                    # Note that we want to extract the actual metadata value;
                    # the magics are all strongly-typed, so we don't need to
                    # worry about the DataType container around them.
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

    @property
    def as_declared(self) -> tuple[T, ...]:
        """This can be used to access the raw assignments, as declared,
        in their exact order. For inline metadata, this is only relevant
        if there are multiple metadata assignments using the same key
        in the same InlineNodeInfo instance.
        """
        # The only reason we do a tuple here is to make sure that the outside
        # world doesn't try to modify this!
        return tuple(self._payload)


@dataclass
class MetadataAssignment(ASTNode):
    key: str
    value: DataType | None


@_MemoizedFieldNames.memoize
@dataclass(kw_only=True)
class InlineNodeInfo(NodeInfo[MetadataAssignment]):
    """InlineNodeInfo is used only for, yknow, inline metadata.
    Note that all of the various formatting tags get sugared into
    inline metadatas.
    """
    icu_1: ReferenceDataType | None = None
    fmt: InlineFormatting | None = None
    sugared: bool | None = None


@_MemoizedFieldNames.memoize
@dataclass(kw_only=True)
class BlockNodeInfo(NodeInfo[MetadataAssignment | Annotation]):
    """BlockNodeInfo is used both for node info and for document
    info (which is itself just an empty node at the toplevel with
    a special magic key set).
    """
    is_doc_metadata: bool = False
    id_: StrDataType | None = None
    embed: StrDataType | None = None
    fallback: EmbedFallbackBehavior | None = None


@dataclass
class DataType(ASTNode):
    # Note: needs to be overridden by subclasses
    value: Any


@dataclass
class StrDataType(DataType):
    value: str


@dataclass
class IntDataType(DataType):
    value: int


@dataclass
class DecimalDataType(DataType):
    value: Decimal


@dataclass
class BoolDataType(DataType):
    value: bool


@dataclass
class NullDataType(DataType):
    value: None


@dataclass
class MentionDataType(DataType):
    value: str


@dataclass
class TagDataType(DataType):
    value: str


@dataclass
class VariableDataType(DataType):
    value: str


@dataclass
class ReferenceDataType(DataType):
    value: str
