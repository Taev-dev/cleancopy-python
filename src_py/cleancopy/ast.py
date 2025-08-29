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
from typing import Protocol

from docnote import DocnoteConfig
from docnote import Note
from docnote import docnote

from cleancopy.spectypes import BlockFormatting
from cleancopy.spectypes import BlockMetadataMagic
from cleancopy.spectypes import EmbedFallbackBehavior
from cleancopy.spectypes import InlineFormatting
from cleancopy.spectypes import InlineMetadataMagic
from cleancopy.spectypes import ListType

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


@dataclass(kw_only=True, slots=True)
class ASTNode:
    """Currently not really used for anything except for annotations,
    but at any rate: this is the base class for all AST nodes.
    """


@dataclass(kw_only=True, slots=True)
class Document(ASTNode):
    # Note: this comes from the __doc_meta__ node
    title: RichtextInlineNode | None
    info: BlockNodeInfo | None
    root: RichtextBlockNode
    # TODO: add other helper methods, like searching by ID


@dataclass(kw_only=True, slots=True)
class BlockNode(ASTNode):
    """The base class for both richtext and embedded nodes."""
    title: RichtextInlineNode | None = None
    info: BlockNodeInfo | None = None
    depth: int


@dataclass(kw_only=True, slots=True)
class RichtextBlockNode(BlockNode):
    content: list[Paragraph | BlockNode]

    def __getitem__(self, index: int) -> Paragraph | BlockNode:
        return self.content[index]

    def __len__(self):
        return len(self.content)

    def __iter__(self):
        return iter(self.content)


@dataclass(kw_only=True, slots=True)
class EmbeddingBlockNode(BlockNode):
    # Can be an explicit None if it's an empty node
    content: str | None


@dataclass(kw_only=True, slots=True)
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


@dataclass(kw_only=True, slots=True)
class List_(ASTNode):  # noqa: N801
    type_: ListType
    content: list[ListItem]


@dataclass(kw_only=True, slots=True)
class ListItem(ASTNode):
    index: int | None
    content: list[Paragraph]


type _RecStrList = list[str | _RecStrList]


@dataclass
class _RecInfoList(list['None | _RecInfoList']):
    info: InlineNodeInfo | None

    def flatten(self, indices: list[int]) -> list[InlineNodeInfo | None]:
        """Traverses the tree and extracts the values based on the
        provided indices, returning them as a list, with the index of
        the list item corresponding to the depth in the tree.
        """
        retval: list[InlineNodeInfo | None] = []
        current_node: _RecInfoList | None = self
        # The zero here is just so that we do an extra iteration at the tail
        # to get the last element; the result never gets used.
        for index in [*indices, 0]:
            if current_node is None:
                raise TypeError('Cannot index into a null nodeinfo!')
            retval.append(current_node.info)
            current_node = current_node[index]

        return retval


@dataclass(kw_only=True, slots=True)
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

    @docnote(config=DocnoteConfig(include_in_docs=False))
    def recursive_strip(self) -> tuple[_RecStrList, _RecInfoList]:
        """This converts the node into a recursive list of strings (and
        list of ...) that matches the structure of the node and its
        children. It also does the same with the infos, coercing any
        strings to None.

        Should not be considered part of the public API for the library.
        Meant for debugging and tests.
        """
        contents: _RecStrList = []
        infos: _RecInfoList = _RecInfoList(info=self.info)

        for child in self.content:
            if isinstance(child, str):
                contents.append(child)
                infos.append(None)
            else:
                rec_contents, rec_infos = child.recursive_strip()
                contents.append(rec_contents)
                infos.append(rec_infos)

        return contents, infos


@dataclass(kw_only=True, slots=True)
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


class _NodeInfoProtocol(Protocol):
    METADATA_MAGICS: ClassVar[
        type[BlockMetadataMagic] | type[InlineMetadataMagic]]
    FORMATTINGS: ClassVar[type[InlineFormatting] | type[BlockFormatting]]


@_MemoizedFieldNames.memoize
@dataclass(kw_only=True, slots=True)
class NodeInfo[T: MetadataAssignment | Annotation](
        ASTNode, _MemoizedFieldNames, _NodeInfoProtocol):

    target: LinkTarget | None = None
    crossref: StrDataType | None = None
    style_modifiers: StrDataType | None = None
    semantic_modifiers: StrDataType | None = None
    layout_modifiers: StrDataType | None = None

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
                metadata_magic = self.METADATA_MAGICS(key)

            except ValueError:
                if METADATA_MAGIC_PATTERN.match(key) is None:
                    # Note: this removes any explicit __missing__ value
                    if line.data is not None:
                        self.metadata[key] = line.data
                else:
                    logger.warning(
                        'Ignoring unknown reserved metadata key: %s', key)

            else:
                # None here means __missing__, which we normalize out
                if line.data is None:
                    return

                maybe_field_name = metadata_magic.name
                # The values here are themselves enums, so we have special
                # handling to make sure they're correct
                if maybe_field_name == 'formatting':
                    try:
                        value_to_use = self.FORMATTINGS(line.data.value)
                    except ValueError:
                        logger.warning(
                            'Ignoring invalid formatting: %s', line.data)
                        return

                else:
                    value_to_use = line.data

                # Note: if you get attribute errors here, it almost certainly
                # means that the spectypes have drifted out of sync with the
                # AST fields on the dataclasses
                setattr(self, maybe_field_name, value_to_use)

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


@dataclass(slots=True)
class MetadataAssignment(ASTNode):
    key: str
    data: DataType | None


@_MemoizedFieldNames.memoize
@dataclass(kw_only=True, slots=True)
class InlineNodeInfo(NodeInfo[MetadataAssignment]):
    """InlineNodeInfo is used only for, yknow, inline metadata.
    Note that all of the various formatting tags get sugared into
    inline metadatas.
    """
    METADATA_MAGICS = InlineMetadataMagic
    FORMATTINGS = InlineFormatting

    icu_1: ReferenceDataType | None = None
    formatting: InlineFormatting | None = None
    citation: StrDataType | None = None
    sugared: BoolDataType | None = None


@_MemoizedFieldNames.memoize
@dataclass(kw_only=True, slots=True)
class BlockNodeInfo(NodeInfo[MetadataAssignment | Annotation]):
    """BlockNodeInfo is used both for node info and for document
    info (which is itself just an empty node at the toplevel with
    a special magic key set).
    """
    METADATA_MAGICS = BlockMetadataMagic
    FORMATTINGS = BlockFormatting

    is_doc_metadata: bool = False
    embed: StrDataType | None = None
    fallback: EmbedFallbackBehavior | None = None
    formatting: BlockFormatting | None = None
    citation: StrDataType | None = None
    source: LinkTarget | None = None


@dataclass(slots=True, frozen=True)
class DataType:
    # Note: needs to be overridden by subclasses
    value: Any


@dataclass(slots=True, frozen=True)
class StrDataType(DataType):
    value: str


@dataclass(slots=True, frozen=True)
class IntDataType(DataType):
    value: int


@dataclass(slots=True, frozen=True)
class DecimalDataType(DataType):
    value: Decimal


@dataclass(slots=True, frozen=True)
class BoolDataType(DataType):
    value: bool


@dataclass(slots=True, frozen=True)
class NullDataType(DataType):
    value: None


@dataclass(slots=True, frozen=True)
class MentionDataType(DataType):
    value: str


@dataclass(slots=True, frozen=True)
class TagDataType(DataType):
    value: str


@dataclass(slots=True, frozen=True)
class VariableDataType(DataType):
    value: str


@dataclass(slots=True, frozen=True)
class ReferenceDataType(DataType):
    value: str
