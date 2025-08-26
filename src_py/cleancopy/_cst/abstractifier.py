"""This contains special-purpose adapters to transform from the CST to
the AST.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field
from functools import singledispatchmethod
from typing import cast
from typing import overload

from cleancopy.ast import Annotation
from cleancopy.ast import ASTNode
from cleancopy.ast import BlockNode
from cleancopy.ast import BlockNodeInfo
from cleancopy.ast import BoolDataType
from cleancopy.ast import DataType
from cleancopy.ast import DecimalDataType
from cleancopy.ast import Document
from cleancopy.ast import EmbeddingBlockNode
from cleancopy.ast import InlineFormatting
from cleancopy.ast import InlineNodeInfo
from cleancopy.ast import IntDataType
from cleancopy.ast import List_
from cleancopy.ast import ListItem
from cleancopy.ast import MentionDataType
from cleancopy.ast import MetadataAssignment
from cleancopy.ast import NullDataType
from cleancopy.ast import Paragraph
from cleancopy.ast import ReferenceDataType
from cleancopy.ast import RichtextBlockNode
from cleancopy.ast import RichtextInlineNode
from cleancopy.ast import StrDataType
from cleancopy.ast import TagDataType
from cleancopy.ast import VariableDataType
from cleancopy.cst import CSTAnnotation
from cleancopy.cst import CSTDocument
from cleancopy.cst import CSTDocumentNode
from cleancopy.cst import CSTDocumentNodeContentEmbedding
from cleancopy.cst import CSTDocumentNodeContentRichtext
from cleancopy.cst import CSTEmptyLine
from cleancopy.cst import CSTFmtBracketLink
from cleancopy.cst import CSTFmtBracketMetadata
from cleancopy.cst import CSTFormattingMarker
from cleancopy.cst import CSTLineBreak
from cleancopy.cst import CSTList
from cleancopy.cst import CSTListItem
from cleancopy.cst import CSTMetadataAssignment
from cleancopy.cst import CSTMetadataBool
from cleancopy.cst import CSTMetadataDecimal
from cleancopy.cst import CSTMetadataInt
from cleancopy.cst import CSTMetadataMention
from cleancopy.cst import CSTMetadataMissing
from cleancopy.cst import CSTMetadataNull
from cleancopy.cst import CSTMetadataReference
from cleancopy.cst import CSTMetadataStr
from cleancopy.cst import CSTMetadataTag
from cleancopy.cst import CSTMetadataVariable
from cleancopy.cst import CSTNode
from cleancopy.cst import CSTRichtext
from cleancopy.exceptions import MultipleDocumentMetadatas
from cleancopy.spectypes import MetadataMagics

logger = logging.getLogger(__name__)
_depth_tracker: ContextVar[int] = ContextVar('_depth_tracker', default=0)


@dataclass(kw_only=True, slots=True)
class Abstractifier:
    replace_linebreak_with: str = ' '

    def convert(self, cst_document: CSTDocument) -> Document:
        cst_root_container = cst_document.root.content

        if cst_root_container is None:
            return Document(
                title=None,
                info=None,
                root=RichtextBlockNode(
                    content=[],
                    depth=0))

        content_helper = _RootNodeContentHelper()
        root_node = RichtextBlockNode(content=content_helper, depth=0)
        self._process_richtext_block_content(
            cst_root_container,
            into_node=root_node)
        document = Document(title=None, info=None, root=root_node)

        if content_helper.document_metadata_node is not None:
            document.title = content_helper.document_metadata_node.title
            document.info = content_helper.document_metadata_node.info

        return document

    @singledispatchmethod
    def _convert(self, cst_node: CSTNode) -> ASTNode:
        raise TypeError('No transform method available for node!', cst_node)

    @_convert.register
    def _(
            self,
            cst_node: CSTDocumentNode
            ) -> RichtextBlockNode | EmbeddingBlockNode:
        # I'd prefer to do this within the CST, but unfortunately there's not a
        # good way of doing it. I mean, ideally I'd like to track it within
        # treesitter, but that ship has long since sailed. But the CST would be
        # my second choice; however, the way we process the treesitter nodes
        # doesn't lend itself well to conditionally updating a tracker, since
        # it's not cleanly recursive (there's an intermediary). So this is a
        # compromise, which should be sufficient until we migrate away from
        # treesitter.
        parent_depth = _depth_tracker.get()
        depth = parent_depth + 1

        ctx_token = _depth_tracker.set(depth)
        try:
            # Awkwardly, we can have empty lines interleaved with richtexts
            merged_title_richtext_content = []
            for cst_title_node in cst_node.title:
                if isinstance(cst_title_node, CSTEmptyLine):
                    # Let the dedicated function handle conversion instead of
                    # repeating it here. The important thing is that, in a
                    # title line, we're ignoring empty lines
                    merged_title_richtext_content.append(
                        self.replace_linebreak_with)
                else:
                    merged_title_richtext_content.extend(cst_title_node.content)

            # A little weird, but we can just create a new virtual richtext
            # item and use it instead
            if (
                merged_title_richtext_content
                and any(merged_title_richtext_content)
            ):
                title = cast(
                    RichtextInlineNode,
                    self._convert(
                        CSTRichtext(content=merged_title_richtext_content)))
            else:
                title = None

            nonempty_metadata = cst_node.nonempty_metadata
            if nonempty_metadata:
                metadata = BlockNodeInfo()
                for cst_metadata_node in nonempty_metadata:
                    ast_metadata_node = cast(
                        MetadataAssignment | Annotation,
                        self._convert(cst_metadata_node))
                    metadata._add(ast_metadata_node)
                is_embed = (metadata.embed is not None)

            else:
                is_embed = False
                metadata = None

            cst_node_content = cst_node.content
            if cst_node_content is None:
                if is_embed:
                    return EmbeddingBlockNode(
                        title=title, info=metadata, content=None, depth=depth)
                else:
                    return RichtextBlockNode(
                        title=title, info=metadata, content=[], depth=depth)

            elif isinstance(cst_node_content, CSTDocumentNodeContentRichtext):
                result = RichtextBlockNode(
                    title=title,
                    info=metadata,
                    content=[],
                    depth=depth)
                self._process_richtext_block_content(
                    cst_node_content, into_node=result)
                return result

            elif isinstance(cst_node_content, CSTDocumentNodeContentEmbedding):
                return EmbeddingBlockNode(
                    title=title,
                    info=metadata,
                    content=cst_node_content.content,
                    depth=depth)

            else:
                raise TypeError(
                    'Impossible branch: unknown CST doc node type!')

        finally:
            _depth_tracker.reset(ctx_token)

    @_convert.register
    def _(self, cst_node: CSTRichtext) -> RichtextInlineNode:
        outermost_span = RichtextInlineNode(
            info=None,
            content=[])
        fmt_stack: list[_FmtStackState] = [
            _FmtStackState(outermost_span, None)]

        for child_cst_node in cst_node.content:
            if isinstance(child_cst_node, str):
                fmt_stack[-1].to_join.append(child_cst_node)
            elif isinstance(child_cst_node, CSTLineBreak):
                fmt_stack[-1].to_join.append(self.replace_linebreak_with)

            # Status check:
            # child_cst_node:
            #   CSTFormattingMarker | CSTFmtBracketLink | CSTFmtBracketMetadata
            else:
                stale_stack_state = fmt_stack[-1]

                if stale_stack_state.to_join:
                    span_text = ''.join(stale_stack_state.to_join)
                    stale_stack_state.to_join.clear()
                    stale_stack_state.current_span.content.append(span_text)

                if isinstance(child_cst_node, CSTFormattingMarker):
                    this_fmt_marker = child_cst_node.marker
                    if this_fmt_marker == stale_stack_state.fmt_marker:
                        span_to_pop = fmt_stack.pop()
                        if span_to_pop.to_join:
                            span_to_pop.current_span.content.append(
                                ''.join(span_to_pop.to_join))
                    else:
                        nested_richtext = RichtextInlineNode(
                            info=InlineNodeInfo(),
                            content=[])
                        nested_richtext.info._add(MetadataAssignment(
                            key=MetadataMagics.sugared.value,
                            value=BoolDataType(value=True)))
                        nested_richtext.info._add(MetadataAssignment(
                            key=MetadataMagics.fmt.value,
                            value=StrDataType(value=this_fmt_marker)))
                        fmt_stack.append(_FmtStackState(
                            current_span=nested_richtext,
                            fmt_marker=this_fmt_marker))
                        stale_stack_state.current_span.content.append(
                            nested_richtext)

                else:
                    # Note: this doesn't get added to the formatting stack,
                    # because it's a whole self-contained thing
                    nested_richtext = cast(
                        RichtextInlineNode, self._convert(child_cst_node))
                    stale_stack_state.current_span.content.append(
                        nested_richtext)
                    # fmt_stack.append(_FmtStackState(
                    #     current_span=nested_richtext,
                    #     fmt_marker=None))

        if (
            len(fmt_stack) != 1
            or fmt_stack[-1].current_span is not outermost_span
        ):
            raise RuntimeError('Failed to fully exhaust richtext fmt stack!')

        outermost_stack_state, = fmt_stack
        outermost_span.content.append(''.join(outermost_stack_state.to_join))

        # Note: because we've had BOTH the stack AND been constructing the
        # inline node as we go along, we don't need to manipulate the tree at
        # all; we already have it fully constructed and can simply return the
        # outermost span.
        return outermost_span

    @_convert.register
    def _(self, cst_node: CSTList) -> List_:
        result = List_(type_=cst_node.type_, content=[])

        for cst_list_item in cst_node.content:
            result.content.append(
                cast(ListItem, self._convert(cst_list_item)))

        return result

    @_convert.register
    def _(self, cst_node: CSTListItem) -> ListItem:
        result = ListItem(index=cst_node.index, content=[])

        # Reminder: we've got an iterator of
        # list[CSTList | CSTAnnotation | CSTRichtext], corresponding to a
        # single paragraph. Note that CSTAnnotations and CSTRichtexts have
        # already consolidated multiple lines; you don't need to do that again!
        for cst_nodegroup in _group_by_paragraph(cst_node.content):
            this_paragraph = Paragraph(content=[])

            for child_cst_node in cst_nodegroup:
                child_cst_node = cast(
                    CSTList | CSTAnnotation | CSTRichtext, child_cst_node)
                this_paragraph.content.append(
                    cast(
                        RichtextInlineNode | List_ | Annotation,
                        self._convert(child_cst_node)))

            result.content.append(this_paragraph)

        return result

    @_convert.register
    def _(self, cst_node: CSTFmtBracketMetadata) -> RichtextInlineNode:
        info = InlineNodeInfo()
        for child_metadata in cst_node.metadata:
            info._add(
                cast(MetadataAssignment, self._convert(child_metadata)))

        span_content = cst_node.content
        if span_content is None:
            return RichtextInlineNode(info=info, content=[])
        else:
            return RichtextInlineNode(
                info=info,
                content=[
                    cast(RichtextInlineNode, self._convert(span_content))])

    @_convert.register
    def _(self, cst_node: CSTFmtBracketLink) -> RichtextInlineNode:
        target = self._convert_metadata(cst_node.target)
        info = InlineNodeInfo()
        info._add(MetadataAssignment(
            key=MetadataMagics.sugared.value, value=BoolDataType(value=True)))
        info._add(MetadataAssignment(
            key=MetadataMagics.target.value, value=target))

        span_content = cst_node.content
        if span_content is None:
            return RichtextInlineNode(info=info, content=[])
        else:
            return RichtextInlineNode(
                info=info,
                content=[
                    cast(RichtextInlineNode, self._convert(span_content))])

    @_convert.register
    def _(self, cst_node: CSTMetadataAssignment) -> MetadataAssignment:
        return MetadataAssignment(
            key=cst_node.key.value,
            value=self._convert_metadata(cst_node.value))

    @_convert.register
    def _(self, cst_node: CSTAnnotation) -> Annotation:
        to_join: list[str] = []
        for maybe_line_break in cst_node.content:
            if isinstance(maybe_line_break, CSTLineBreak):
                # Note: collapsing multiple consecutive line breaks into one
                # is part of the actual grammar, so we don't -- or at least
                # shouldn't -- need to worry about it here
                to_join.append(self.replace_linebreak_with)
            else:
                to_join.append(maybe_line_break)

        return Annotation(content=''.join(to_join))

    def _process_richtext_block_content(
            self,
            cst_node: CSTDocumentNodeContentRichtext,
            *,
            into_node: RichtextBlockNode
            ) -> None:
        """Given a partially-processed richtext block,
        transforms the CST's content into AST form and appends it to the
        into_node.
        """
        for cst_nodegroup_or_nested_node in _group_by_paragraph(
            cst_node.content
        ):
            if isinstance(cst_nodegroup_or_nested_node, CSTDocumentNode):
                into_node.content.append(
                    cast(
                        BlockNode,
                        self._convert(cst_nodegroup_or_nested_node)))

            # Status check: now we have
            # list[CSTList | CSTAnnotation | CSTRichtext], corresponding to a
            # single paragraph. Note that CSTAnnotations and CSTRichtexts have
            # already consolidated multiple lines; you don't need to do that
            # again!
            else:
                this_paragraph = Paragraph(content=[])

                for child_cst_node in cst_nodegroup_or_nested_node:
                    child_cst_node = cast(
                        CSTList | CSTAnnotation | CSTRichtext, child_cst_node)
                    this_paragraph.content.append(
                        cast(
                            RichtextInlineNode | List_ | Annotation,
                            self._convert(child_cst_node)))

                into_node.content.append(this_paragraph)

    @singledispatchmethod
    def _convert_metadata(self, cst_node: CSTNode) -> DataType | None:
        raise TypeError('Unknown CST metadata type!', cst_node)

    @_convert_metadata.register
    def _(self, cst_node: CSTMetadataStr) -> StrDataType:
        return StrDataType(value=cst_node.value)

    @_convert_metadata.register
    def _(self, cst_node: CSTMetadataInt) -> IntDataType:
        return IntDataType(value=cst_node.value)

    @_convert_metadata.register
    def _(self, cst_node: CSTMetadataDecimal) -> DecimalDataType:
        return DecimalDataType(value=cst_node.value)

    @_convert_metadata.register
    def _(self, cst_node: CSTMetadataBool) -> BoolDataType:
        return BoolDataType(value=cst_node.value)

    @_convert_metadata.register
    def _(self, cst_node: CSTMetadataNull) -> NullDataType:
        return NullDataType(value=cst_node.value)

    @_convert_metadata.register
    def _(self, cst_node: CSTMetadataMissing) -> None:
        return None

    @_convert_metadata.register
    def _(self, cst_node: CSTMetadataMention) -> MentionDataType:
        return MentionDataType(value=cst_node.value.value)

    @_convert_metadata.register
    def _(self, cst_node: CSTMetadataTag) -> TagDataType:
        return TagDataType(value=cst_node.value.value)

    @_convert_metadata.register
    def _(self, cst_node: CSTMetadataVariable) -> VariableDataType:
        return VariableDataType(value=cst_node.value.value)

    @_convert_metadata.register
    def _(self, cst_node: CSTMetadataReference) -> ReferenceDataType:
        return ReferenceDataType(value=cst_node.value.value)


@dataclass(kw_only=True, slots=True, repr=False)
class _RootNodeContentHelper[T](list[T]):
    """This helper is used to do an inline extraction of the document
    metadata node (thereby avoiding a list mutation) while generating
    the root node.
    """
    document_metadata_node: RichtextBlockNode | None = field(
        default=None, init=False)

    def append(self, item: T) -> None:
        if (
            isinstance(item, RichtextBlockNode)
            and item.info is not None
            and item.info.is_doc_metadata
        ):
            if self.document_metadata_node is not None:
                raise MultipleDocumentMetadatas()
            else:
                self.document_metadata_node = item

        else:
            # Explicit super() because of dataclass + slots
            super(_RootNodeContentHelper, self).append(item)


@dataclass
class _FmtStackState:
    current_span: RichtextInlineNode
    fmt_marker: InlineFormatting | None
    to_join: list[str] = field(default_factory=list)


@overload
def _group_by_paragraph[
            T: CSTEmptyLine | CSTList | CSTAnnotation | CSTRichtext](
        lines: Sequence[T]
        ) -> Iterator[list[T]]: ...
@overload
def _group_by_paragraph[
            T: CSTEmptyLine | CSTList | CSTAnnotation | CSTRichtext](
        lines: Sequence[T | CSTDocumentNode]
        ) -> Iterator[list[T] | CSTDocumentNode]: ...
def _group_by_paragraph[
            T: CSTEmptyLine | CSTList | CSTAnnotation | CSTRichtext](
        lines: Sequence[T | CSTDocumentNode]
        ) -> Iterator[list[T] | CSTDocumentNode]:
    this_paragraph = []
    for line in lines:
        if isinstance(line, CSTEmptyLine):
            if this_paragraph:
                yield this_paragraph
                this_paragraph = []

        elif isinstance(line, CSTDocumentNode):
            if this_paragraph:
                yield this_paragraph
                this_paragraph = []
            yield line

        else:
            this_paragraph.append(line)

    if this_paragraph:
        yield this_paragraph
