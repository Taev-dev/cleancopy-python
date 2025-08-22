"""This contains special-purpose adapters to transform from the CST to
the AST.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from functools import singledispatch
from typing import cast
from typing import overload

from cleancopy.ast import Annotation
from cleancopy.ast import ASTNode
from cleancopy.ast import BlockMetadata
from cleancopy.ast import DocNode
from cleancopy.ast import Document
from cleancopy.ast import EmbeddingDocNode
from cleancopy.ast import InlineFormatting
from cleancopy.ast import InlineMetadata
from cleancopy.ast import List_
from cleancopy.ast import ListItem
from cleancopy.ast import MetadataAssignment
from cleancopy.ast import MetadataBool
from cleancopy.ast import MetadataDecimal
from cleancopy.ast import MetadataInt
from cleancopy.ast import MetadataMention
from cleancopy.ast import MetadataNull
from cleancopy.ast import MetadataReference
from cleancopy.ast import MetadataStr
from cleancopy.ast import MetadataTag
from cleancopy.ast import MetadataTarget
from cleancopy.ast import MetadataValue
from cleancopy.ast import MetadataVariable
from cleancopy.ast import Paragraph
from cleancopy.ast import Richtext
from cleancopy.ast import RichtextDocNode
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


@singledispatch
def transform(cst_node: CSTNode, **kwargs) -> ASTNode | str | None:
    raise TypeError('No transformer available for node!', cst_node)


@transform.register
def _(cst_node: CSTDocument, **kwargs) -> Document:
    cst_root_container = cst_node.root.content

    if cst_root_container is None:
        return Document(
            title=None, metadata=None, root=RichtextDocNode(content=[]))

    root_content = []
    root_node = RichtextDocNode(content=root_content)
    document = Document(title=None, metadata=None, root=root_node)
    for child_cst_node in cst_root_container.content:
        child_ast_node = transform(child_cst_node, **kwargs)

        if (
            isinstance(child_ast_node, RichtextDocNode)
            and child_ast_node.metadata is not None
            and child_ast_node.metadata.is_doc_metadata
        ):
            if document.metadata is not None:
                raise MultipleDocumentMetadatas()
            else:
                document.metadata = child_ast_node.metadata
                document.title = child_ast_node.title

        else:
            root_content.append(child_ast_node)

    return document


@transform.register
def _(
        cst_node: CSTDocumentNode, **kwargs
        ) -> RichtextDocNode | EmbeddingDocNode:
    # Awkwardly, we can have empty lines interleaved with richtexts
    merged_title_richtext_content = []
    for cst_title_node in cst_node.title:
        if isinstance(cst_title_node, CSTEmptyLine):
            # Let the dedicated function handle conversion instead of repeating
            # it here. The important thing is that, in a title line, we're
            # ignoring empty lines
            merged_title_richtext_content.append(
                transform(CSTLineBreak(content=None), **kwargs))
        else:
            merged_title_richtext_content.extend(cst_title_node.content)

    # A little weird, but we can just create a new virtual richtext item and
    # use it instead
    title = cast(
        Richtext,
        transform(
            CSTRichtext(content=merged_title_richtext_content), **kwargs))

    metadata = BlockMetadata()
    for cst_metadata_node in cst_node.metadata:
        if not isinstance(cst_metadata_node, CSTEmptyLine):
            ast_metadata_node = cast(
                MetadataAssignment | Annotation,
                transform(cst_metadata_node, **kwargs))
            metadata._add(ast_metadata_node)

    cst_node_content = cst_node.content
    if cst_node_content is None:
        if metadata.embed is None:
            return RichtextDocNode(title=title, metadata=metadata, content=[])
        else:
            return EmbeddingDocNode(
                title=title, metadata=metadata, content=None)

    elif isinstance(cst_node_content, CSTDocumentNodeContentRichtext):
        result = RichtextDocNode(
            title=title,
            metadata=metadata,
            content=[])
        transform(cst_node_content, into_node=result, **kwargs)
        return result

    elif isinstance(cst_node_content, CSTDocumentNodeContentEmbedding):
        return EmbeddingDocNode(
            title=title,
            metadata=metadata,
            content=cst_node_content.content)

    else:
        raise TypeError('Impossible branch: unknown CST doc node type!')


@transform.register
def _(
        cst_node: CSTDocumentNodeContentRichtext,
        *, into_node: RichtextDocNode, **kwargs) -> RichtextDocNode:
    """Given a partially-processed RichtextNode with an empty content,
    transforms the CST's content into AST form and appends it to the
    into_node, and then returns the node back.
    """
    for cst_nodegroup_or_nested_node in _group_by_paragraph(cst_node.content):
        if isinstance(cst_nodegroup_or_nested_node, CSTDocumentNode):
            into_node.content.append(
                cast(
                    DocNode,
                    transform(cst_nodegroup_or_nested_node, **kwargs)))

        # Status check: now we have
        # list[CSTList | CSTAnnotation | CSTRichtext], corresponding to a
        # single paragraph. Note that CSTAnnotations and CSTRichtexts have
        # already consolidated multiple lines; you don't need to do that again!
        else:
            this_paragraph = Paragraph(content=[])

            for child_cst_node in cst_nodegroup_or_nested_node:
                child_cst_node = cast(
                    CSTList | CSTAnnotation | CSTRichtext, child_cst_node)
                this_paragraph.content.append(
                    cast(
                        Richtext | List_ | Annotation,
                        transform(child_cst_node, **kwargs)))

            into_node.content.append(this_paragraph)

    return into_node


@dataclass
class _FmtStackState:
    current_span: Richtext
    fmt_marker: InlineFormatting | None
    to_join: list[str] = field(default_factory=list)


@transform.register
def _(cst_node: CSTRichtext, **kwargs) -> Richtext:
    outermost_span = Richtext(
        context=None,
        content=[])
    fmt_stack: list[_FmtStackState] = [_FmtStackState(outermost_span, None)]

    for child_cst_node in cst_node.content:
        if isinstance(child_cst_node, str):
            fmt_stack[-1].to_join.append(child_cst_node)
        elif isinstance(child_cst_node, CSTLineBreak):
            fmt_stack[-1].to_join.append(
                cast(str, transform(child_cst_node, **kwargs)))

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
                    nested_richtext = Richtext(
                        context=InlineMetadata(),
                        content=[])
                    nested_richtext.context._add(MetadataAssignment(
                        key=MetadataMagics.sugared.value,
                        value=MetadataBool(value=True)))
                    nested_richtext.context._add(MetadataAssignment(
                        key=MetadataMagics.fmt.value,
                        value=MetadataStr(value=this_fmt_marker)))
                    fmt_stack.append(_FmtStackState(
                        current_span=nested_richtext,
                        fmt_marker=this_fmt_marker))
                    stale_stack_state.current_span.content.append(
                        nested_richtext)

            else:
                # Note: this doesn't get added to the formatting stack, because
                # it's a whole self-contained thing
                nested_richtext = cast(
                    Richtext, transform(child_cst_node, **kwargs))
                stale_stack_state.current_span.content.append(nested_richtext)
                # fmt_stack.append(_FmtStackState(
                #     current_span=nested_richtext,
                #     fmt_marker=None))

    if len(fmt_stack) != 1 or fmt_stack[-1].current_span is not outermost_span:
        print(fmt_stack)
        raise RuntimeError('Failed to fully exhaust richtext fmt stack!')
    outermost_stack_state, = fmt_stack
    outermost_span.content.append(''.join(outermost_stack_state.to_join))

    # Note: because we've had BOTH the stack AND been constructing the Richtext
    # tree as we go along, we don't need to manipulate the tree at all; we
    # already have it fully constructed and can simply return the outermost
    # span.
    return outermost_span


@transform.register
def _(cst_node: CSTList, **kwargs) -> List_:
    result = List_(type_=cst_node.type_, content=[])

    for cst_list_item in cst_node.content:
        result.content.append(
            cast(ListItem, transform(cst_list_item, **kwargs)))

    return result


@transform.register
def _(cst_node: CSTListItem, **kwargs) -> ListItem:
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
                    Richtext | List_ | Annotation,
                    transform(child_cst_node, **kwargs)))

        result.content.append(this_paragraph)

    return result


@transform.register
def _(cst_node: CSTFmtBracketMetadata, **kwargs) -> Richtext:
    context = InlineMetadata()
    for child_metadata in cst_node.metadata:
        context._add(
            cast(MetadataAssignment, transform(child_metadata, **kwargs)))

    span_content = cst_node.content
    if span_content is None:
        return Richtext(context=context, content=[])
    else:
        return Richtext(
            context=context,
            content=[cast(Richtext, transform(span_content, **kwargs))])


@transform.register
def _(cst_node: CSTFmtBracketLink, **kwargs) -> Richtext:
    target = cast(MetadataTarget, transform(cst_node.target, **kwargs))
    context = InlineMetadata()
    context._add(MetadataAssignment(
        key=MetadataMagics.sugared.value, value=MetadataBool(value=True)))
    context._add(MetadataAssignment(
        key=MetadataMagics.target.value, value=target))

    span_content = cst_node.content
    if span_content is None:
        return Richtext(context=context, content=[])
    else:
        return Richtext(
            context=context,
            content=[cast(Richtext, transform(span_content, **kwargs))])


@transform.register
def _(cst_node: CSTMetadataAssignment, **kwargs) -> MetadataAssignment:
    return MetadataAssignment(
        key=cst_node.key.value,
        value=cast(MetadataValue, transform(cst_node.value)))


@transform.register
def _(cst_node: CSTMetadataStr, **kwargs) -> MetadataStr:
    return MetadataStr(value=cst_node.value)


@transform.register
def _(cst_node: CSTMetadataInt, **kwargs) -> MetadataInt:
    return MetadataInt(value=cst_node.value)


@transform.register
def _(cst_node: CSTMetadataDecimal, **kwargs) -> MetadataDecimal:
    return MetadataDecimal(value=cst_node.value)


@transform.register
def _(cst_node: CSTMetadataBool, **kwargs) -> MetadataBool:
    return MetadataBool(value=cst_node.value)


@transform.register
def _(cst_node: CSTMetadataNull, **kwargs) -> MetadataNull:
    return MetadataNull(value=cst_node.value)


@transform.register
def _(cst_node: CSTMetadataMissing, **kwargs) -> None:
    return None


@transform.register
def _(cst_node: CSTMetadataMention, **kwargs) -> MetadataMention:
    return MetadataMention(value=cst_node.value.value)


@transform.register
def _(cst_node: CSTMetadataTag, **kwargs) -> MetadataTag:
    return MetadataTag(value=cst_node.value.value)


@transform.register
def _(cst_node: CSTMetadataVariable, **kwargs) -> MetadataVariable:
    return MetadataVariable(value=cst_node.value.value)


@transform.register
def _(cst_node: CSTMetadataReference, **kwargs) -> MetadataReference:
    return MetadataReference(value=cst_node.value.value)


@transform.register
def _(cst_node: CSTAnnotation, **kwargs) -> Annotation:
    to_join: list[str] = []
    for maybe_line_break in cst_node.content:
        if isinstance(maybe_line_break, CSTLineBreak):
            stringified = cast(str, transform(maybe_line_break, **kwargs))
            to_join.append(stringified)
        else:
            to_join.append(maybe_line_break)

    return Annotation(content=''.join(to_join))


@transform.register
def _(
        cst_node: CSTLineBreak, *,
        replace_linebreak_with: str = ' ',
        **kwargs) -> str:
    # Note: collapsing multiple line breaks is part of the actual grammar,
    # so we don't -- or at least shouldn't -- need to worry about it here
    return replace_linebreak_with


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

    yield this_paragraph
