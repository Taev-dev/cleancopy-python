"""This module has all of the magic incantations we use to get a pseudo-CST
from the cleancopy tree-sitter binding.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from collections.abc import Iterator
from decimal import Decimal
from enum import StrEnum
from typing import Literal
from typing import cast

import tree_sitter_cleancopy
from tree_sitter import Language as TSLanguage
from tree_sitter import Node as TSNode
from tree_sitter import Parser as TSParser
from tree_sitter import Tree as TSTree

from cleancopy.cst import CSTAnnotation
from cleancopy.cst import CSTDocument
from cleancopy.cst import CSTDocumentNode
from cleancopy.cst import CSTDocumentNodeContentEmbedding
from cleancopy.cst import CSTDocumentNodeContentRichtext
from cleancopy.cst import CSTDocumentNodeMetadata
from cleancopy.cst import CSTDocumentNodeTitle
from cleancopy.cst import CSTEmptyLine
from cleancopy.cst import CSTFmtBracketLink
from cleancopy.cst import CSTFmtBracketMetadata
from cleancopy.cst import CSTFmtBracketMetadataContainer
from cleancopy.cst import CSTFormattingMarker
from cleancopy.cst import CSTLineBreak
from cleancopy.cst import CSTList
from cleancopy.cst import CSTListItem
from cleancopy.cst import CSTListItemOLIndexContainer
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
from cleancopy.cst import CSTMetadataTarget
from cleancopy.cst import CSTMetadataValue
from cleancopy.cst import CSTMetadataVariable
from cleancopy.cst import CSTNode
from cleancopy.cst import CSTRichtext
from cleancopy.spectypes import InlineFormatting
from cleancopy.spectypes import ListType

CLEANCOPY_LANGUAGE = TSLanguage(tree_sitter_cleancopy.language())
CLEANCOPY_PARSER = TSParser(CLEANCOPY_LANGUAGE)

logger = logging.getLogger(__name__)
_TS_NODE_REGISTRY = {}
_AUTOCLOSE_GRAMMAR_TYPE = 'fmt_autoclose'
_EMPTY_LINE_GRAMMAR_TYPE = 'empty_line'


class _TSNamedField(StrEnum):
    DOCUMENT = 'document'
    ROOT_NODE_CONTENT = 'root_node_content'

    # Note: also includes fmt_autoclose
    RICHTEXT_LINE = 'richtext_line'
    RICHTEXT_FMT_BRACKET = 'fmt_bracket'
    RICHTEXT_FMT_TARGET = 'target'

    NODE = 'node'
    NODE_TITLE = 'title'
    NODE_METADATA = 'metadata'
    NODE_CONTENT = 'node_content'
    EMBEDDING_CONTENT = 'embedding_content'

    INLINE_METADATA = 'inline_metadata'
    METADATA_ASSIGNMENT = 'declaration'
    METADATA_KEY = 'key'
    METADATA_VALUE = 'value'

    LIST_OL = 'list_ol'
    LIST_UNOL = 'list_unol'
    LIST_ITEM = 'list_item'
    LIST_OL_INDEX = 'index'
    # Frustrating artifact of the treesitter grammar
    ANNOTATION_LINE = 'annotation'
    ANNOTATION_LINE_CONTENT = 'annotation_content'

    EMPTY_LINE = 'empty_line'


class _TSValueTypes(StrEnum):
    STRING1 = 'STRING1'
    STRING2 = 'STRING2'
    NULL = 'NULL'
    MISSING = 'MISSING'
    TRUE = 'TRUE'
    FALSE = 'FALSE'
    INT = 'integer'
    DECIMAL = 'decimal'
    MENTION = 'mention'
    TAG = 'tag'
    VARIABLE = 'variable'
    REF = 'ref'
    URI = 'uri'


class _TSRichtextTypes(StrEnum):
    PLAINTEXT = 'plaintext'
    FMT_PRE = 'ext_fmt_pre'
    FMT_UNDERLINE = 'ext_fmt_underline'
    FMT_STRONG = 'ext_fmt_strong'
    FMT_EMPHASIS = 'ext_fmt_emphasis'
    FMT_STRIKE = 'ext_fmt_strike'
    FMT_BRACKET_LINK_ANON = 'fmt_bracket_anon_link'
    FMT_BRACKET_LINK = 'fmt_bracket_named_link'
    FMT_BRACKET_METADATA = 'fmt_bracket_metadata'


def _get_string_quote_style(
        string_type:
            Literal[_TSValueTypes.STRING1] | Literal[_TSValueTypes.STRING2]
        ) -> Literal[1] | Literal[2]:
    if string_type == _TSValueTypes.STRING1:
        return 1
    else:
        return 2


def _register[T, **P](
        field_name: _TSNamedField
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Use this as a value dispatch for tree sitter node types."""
    if field_name in _TS_NODE_REGISTRY:
        raise ValueError('Already registered!', field_name)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        _TS_NODE_REGISTRY[field_name] = func
        return func

    return decorator


def handle_node(field_name: _TSNamedField, node: TSNode) -> CSTNode | None:
    handler = _TS_NODE_REGISTRY.get(field_name)

    if handler is None:
        logger.warning(
            'Node type %s not yet implemented: grammar_name %s, field_name %s',
            node.type, node.grammar_name, field_name)
        return None

    return handler(node)


def parse(document: bytes) -> CSTDocument:
    ts_tree = CLEANCOPY_PARSER.parse(document)
    return to_cst(ts_tree)


def to_cst(tree: TSTree) -> CSTDocument:
    """Converts a CST parse tree into a Document."""
    root_node = tree.root_node

    grammar_type = root_node.type
    if grammar_type != _TSNamedField.DOCUMENT:
        raise ValueError('AST tree must have document as outermost node!')

    maybe_document = handle_node(_TSNamedField.DOCUMENT, root_node)
    if maybe_document is None:
        raise RuntimeError('Failed to process document!')
    elif not isinstance(maybe_document, CSTDocument):
        raise TypeError('Parse tree had no document at root!', maybe_document)
    return maybe_document


@_register(_TSNamedField.DOCUMENT)
def process_document_root(node: TSNode) -> CSTDocument:
    raw_root_node_body, *invalid_extras = _process_named_children(node)

    if invalid_extras:
        raise TypeError('Document root had invalid extras!', invalid_extras)
    if not isinstance(raw_root_node_body, CSTDocumentNodeContentRichtext):
        raise TypeError('Invalid child of document root!', raw_root_node_body)

    # Note: as per spec, we skip any leading emptylines
    first_nonempty_encountered = False
    filtered_content: list[
        CSTEmptyLine | CSTList | CSTAnnotation | CSTRichtext | CSTDocumentNode
    ] = []
    for child_cst_node in raw_root_node_body.content:
        if isinstance(child_cst_node, CSTEmptyLine):
            if first_nonempty_encountered:
                filtered_content.append(child_cst_node)
        else:
            first_nonempty_encountered = True
            filtered_content.append(child_cst_node)

    # Note that the AST document might have a title and/or metadata, but ONLY
    # the AST -- that doesn't get added in the CST!
    root_node = CSTDocumentNode(
        title=[],
        metadata=[],
        content=CSTDocumentNodeContentRichtext(content=filtered_content))
    return CSTDocument(root=root_node)


@_register(_TSNamedField.ROOT_NODE_CONTENT)
@_register(_TSNamedField.NODE_CONTENT)
def process_node_richtext_content(
        node: TSNode) -> CSTDocumentNodeContentRichtext:
    result = CSTDocumentNodeContentRichtext(content=[])

    for child_cst_node in _process_named_children(node):
        _add_or_merge_content(child_cst_node, result.content)

    return result


@_register(_TSNamedField.EMBEDDING_CONTENT)
def process_node_embedding_content(
        node: TSNode) -> CSTDocumentNodeContentEmbedding:
    lines = []
    for child in node.children:
        if child.type == _EMPTY_LINE_GRAMMAR_TYPE:
            lines.append('')
        elif child.text is not None:
            lines.append(child.text.decode())
        else:
            lines.append('')

    return CSTDocumentNodeContentEmbedding(
        content='\n'.join(lines))


@_register(_TSNamedField.LIST_OL)
def process_list_ordered(node: TSNode) -> CSTList:
    items: list[CSTListItem] = []
    for child_cst_node in _process_named_children(node):
        if isinstance(child_cst_node, CSTListItem):
            items.append(child_cst_node)
        else:
            raise TypeError('Invalid child of ordered list!', child_cst_node)

    return CSTList(type_=ListType.ORDERED, content=items)


@_register(_TSNamedField.LIST_UNOL)
def process_list_unordered(node: TSNode) -> CSTList:
    items: list[CSTListItem] = []
    for child_cst_node in _process_named_children(node):
        if isinstance(child_cst_node, CSTListItem):
            items.append(child_cst_node)
        else:
            raise TypeError('Invalid child of unordered list!', child_cst_node)

    return CSTList(type_=ListType.UNORDERED, content=items)


@_register(_TSNamedField.LIST_ITEM)
def process_list_item(node: TSNode) -> CSTListItem:
    index: int | None = None
    content: list[CSTEmptyLine | CSTList | CSTAnnotation | CSTRichtext] = []

    for child_cst_node in _process_named_children(node):
        if isinstance(
            child_cst_node,
            CSTEmptyLine | CSTList | CSTAnnotation | CSTRichtext
        ):
            _add_or_merge_content(child_cst_node, content)

        elif isinstance(child_cst_node, CSTListItemOLIndexContainer):
            if index is not None:
                raise TypeError(
                    'Multiple indices for list item', child_cst_node)

            index = child_cst_node.value
        else:
            raise TypeError('Invalid child of list item!', child_cst_node)

    return CSTListItem(index=index, content=content)


@_register(_TSNamedField.LIST_OL_INDEX)
def process_list_ol_index(node: TSNode) -> CSTListItemOLIndexContainer:
    if node.text is None:
        raise ValueError('Empty list index!', node)
    else:
        return CSTListItemOLIndexContainer(value=int(node.text.decode()))


@_register(_TSNamedField.EMPTY_LINE)
def process_empty_line(node: TSNode) -> CSTEmptyLine:
    if node.text is None:
        return CSTEmptyLine(content='')
    else:
        return CSTEmptyLine(content=node.text.decode())


@_register(_TSNamedField.NODE)
def process_document_node(node: TSNode) -> CSTDocumentNode:
    doc_node = CSTDocumentNode(
        title=[],
        metadata=[],
        content=None)

    for child_cst_node in _process_named_children(node):
        if isinstance(child_cst_node, CSTEmptyLine):
            if doc_node.metadata:
                doc_node.metadata.append(child_cst_node)
            else:
                doc_node.title.append(child_cst_node)

        elif isinstance(child_cst_node, CSTDocumentNodeMetadata):
            doc_node.metadata.extend(child_cst_node.content)

        elif isinstance(child_cst_node, CSTDocumentNodeTitle):
            doc_node.title.extend(child_cst_node.content)

        elif isinstance(
            child_cst_node,
            CSTDocumentNodeContentRichtext | CSTDocumentNodeContentEmbedding
        ):
            if doc_node.content is not None:
                raise ValueError('Document node with multiple contents!')
            doc_node.content = child_cst_node

        else:
            raise TypeError('Invalid child of document node!', child_cst_node)

    return doc_node


@_register(_TSNamedField.RICHTEXT_LINE)
def process_richtext_line(node: TSNode) -> CSTRichtext:
    """Note: this also handles everything with formatting autoclosing.
    """
    result = CSTRichtext(content=[])
    was_autoclose = (node.type == _AUTOCLOSE_GRAMMAR_TYPE)

    for child in node.children:
        if child.type == _TSRichtextTypes.PLAINTEXT:
            if child.text is None:
                result.content.append('')
            else:
                result.content.append(child.text.decode())

        elif child.type == _TSRichtextTypes.FMT_PRE:
            result.content.append(CSTFormattingMarker(
                marker=InlineFormatting.PRE,
                was_autoclose=was_autoclose))
        elif child.type == _TSRichtextTypes.FMT_UNDERLINE:
            result.content.append(CSTFormattingMarker(
                marker=InlineFormatting.UNDERLINE,
                was_autoclose=was_autoclose))
        elif child.type == _TSRichtextTypes.FMT_STRONG:
            result.content.append(CSTFormattingMarker(
                marker=InlineFormatting.STRONG,
                was_autoclose=was_autoclose))
        elif child.type == _TSRichtextTypes.FMT_EMPHASIS:
            result.content.append(CSTFormattingMarker(
                marker=InlineFormatting.EMPHASIS,
                was_autoclose=was_autoclose))
        elif child.type == _TSRichtextTypes.FMT_STRIKE:
            result.content.append(CSTFormattingMarker(
                marker=InlineFormatting.STRIKE,
                was_autoclose=was_autoclose))

        elif child.type in (
            _TSRichtextTypes.FMT_BRACKET_LINK_ANON,
            _TSRichtextTypes.FMT_BRACKET_LINK,
            _TSRichtextTypes.FMT_BRACKET_METADATA
        ):
            result.content.append(process_fmt_bracket(child))
        else:
            raise TypeError(
                'Invalid richtext line child type!',
                child, child.type, child.grammar_name)

    return result


@_register(_TSNamedField.RICHTEXT_FMT_BRACKET)
def process_fmt_bracket(  # noqa: C901, PLR0912, PLR0915
        node: TSNode
        ) -> CSTFmtBracketMetadata | CSTFmtBracketLink:
    if node.type == _TSRichtextTypes.FMT_BRACKET_LINK_ANON:
        target, *invalid_extras = _process_named_children(node)
        if invalid_extras:
            raise TypeError('Fmt bracket has invalid extras!', invalid_extras)
        if not isinstance(target, CSTMetadataTarget.__value__):
            raise TypeError('Fmt bracket has invalid target!', target)
        target = cast(CSTMetadataTarget, target)

        return CSTFmtBracketLink(
            content=None,
            target=target)

    elif node.type == _TSRichtextTypes.FMT_BRACKET_LINK:
        content: list[CSTRichtext] = []
        invalid_extras = []
        target = None
        for child in _process_named_children(node):
            if isinstance(child, CSTMetadataTarget.__value__):
                if target is not None:
                    raise TypeError(
                        'Fmt bracket has multiple targets!', child)
                # Note: only needed because of the weird isinstance hack above
                target = cast(CSTMetadataTarget, child)

            elif isinstance(child, CSTRichtext):
                _add_or_merge_content(child, content)

            else:
                invalid_extras.append(child)

        if invalid_extras:
            raise TypeError('Fmt bracket has invalid extras!', invalid_extras)
        if target is None:
            raise TypeError('Fmt bracket is missing target!', node)
        if len(content) > 1:
            raise TypeError('Fmt bracket has multiple richtexts!', content)

        if content:
            normalized_content, = content
        else:
            normalized_content = CSTRichtext(content=[])

        return CSTFmtBracketLink(
            content=normalized_content,
            target=target)

    elif node.type == _TSRichtextTypes.FMT_BRACKET_METADATA:
        content: list[CSTRichtext] = []
        invalid_extras = []
        metadata = None
        for child in _process_named_children(node):
            if isinstance(child, CSTFmtBracketMetadataContainer):
                if metadata is not None:
                    raise TypeError(
                        'Fmt bracket has multiple metadatas!', child)
                metadata = child

            elif isinstance(child, CSTRichtext):
                _add_or_merge_content(child, content)

            else:
                invalid_extras.append(child)

        if invalid_extras:
            raise TypeError('Fmt bracket has invalid extras!', invalid_extras)
        if metadata is None:
            raise TypeError('Fmt bracket is missing metadata!', node)
        if len(content) > 1:
            raise TypeError('Fmt bracket has multiple richtexts!', content)

        if content:
            normalized_content, = content
        else:
            normalized_content = CSTRichtext(content=[])

        return CSTFmtBracketMetadata(
            content=normalized_content,
            metadata=metadata.content)

    else:
        raise TypeError('Invalid fmt bracket node type!', node)


@_register(_TSNamedField.METADATA_ASSIGNMENT)
def process_metadata_assignment(node: TSNode) -> CSTMetadataAssignment:
    key, value, *invalid_extras = _process_named_children(node)
    if invalid_extras:
        raise TypeError(
            'Invalid child of metadata assignment!', invalid_extras)

    if not isinstance(key, CSTMetadataStr):
        raise TypeError('Metadata key is not metadata string!', key)
    if not isinstance(value, CSTMetadataValue.__value__):
        raise TypeError('Metadata value is not a metadata value type!', value)
    value = cast(CSTMetadataValue, value)

    return CSTMetadataAssignment(key=key, value=value)


@_register(_TSNamedField.METADATA_KEY)
def process_metadata_key(node: TSNode) -> str | CSTMetadataStr:
    return _normalize_sugarable_string(node)


@_register(_TSNamedField.METADATA_VALUE)
def process_metadata_value(  # noqa: C901, PLR0911, PLR0912
        node: TSNode
        ) -> CSTMetadataValue:
    first_child, *other_children = node.children

    if first_child.text is None:
        raise ValueError('Metadata values must have content!')

    all_child_types = {child.type for child in node.children}
    if len(all_child_types) != 1:
        raise TypeError('Metadata values must have exactly one type of value!')

    if first_child.type in (
        _TSValueTypes.STRING1, _TSValueTypes.STRING2
    ):
        value_type = cast(
            Literal[_TSValueTypes.STRING1] | Literal[_TSValueTypes.STRING2],
            _TSValueTypes(first_child.type))
        # Used to detect multi-line strings... which currently aren't working,
        # but shhhhh
        string_collector = []
        string_collector.append(first_child.text.decode())
        for other_child in other_children:
            if other_child.text is None:
                raise ValueError(
                    'Metadata values must have content!', other_child)

            string_collector.append(other_child.text.decode())

        return CSTMetadataStr(
            '\n'.join(string_collector),
            quote_style=_get_string_quote_style(value_type))

    elif other_children:
        raise ValueError(
            'Only multiline strings can have multiple metadata values!',
            other_children)

    elif first_child.type == _TSValueTypes.NULL:
        return CSTMetadataNull(value=None)
    elif first_child.type == _TSValueTypes.MISSING:
        return CSTMetadataMissing(value=None)
    elif first_child.type == _TSValueTypes.TRUE:
        return CSTMetadataBool(value=True)
    elif first_child.type == _TSValueTypes.FALSE:
        return CSTMetadataBool(value=False)
    elif first_child.type == _TSValueTypes.INT:
        return CSTMetadataInt(value=int(first_child.text.decode()))
    elif first_child.type == _TSValueTypes.DECIMAL:
        return CSTMetadataDecimal(value=Decimal(first_child.text.decode()))
    elif first_child.type == _TSValueTypes.MENTION:
        return CSTMetadataMention(
            value=_normalize_sugarable_string(
                first_child,
                strip_prefix='@'),)
    elif first_child.type == _TSValueTypes.TAG:
        return CSTMetadataTag(
            value=_normalize_sugarable_string(
                first_child,
                strip_prefix='#'),)
    elif first_child.type == _TSValueTypes.VARIABLE:
        return CSTMetadataVariable(
            value=_normalize_sugarable_string(
                first_child,
                strip_prefix='%'),)
    elif first_child.type == _TSValueTypes.REF:
        return CSTMetadataReference(
            value=_normalize_sugarable_string(
                first_child,
                strip_prefix='&'),)
    else:
        raise TypeError('Invalid metadata value type!', first_child)


@_register(_TSNamedField.RICHTEXT_FMT_TARGET)
def process_fmt_bracket_link_target(node: TSNode) -> CSTMetadataTarget:
    if node.type == _TSValueTypes.MENTION:
        return CSTMetadataMention(
            value=_normalize_sugarable_string(
                node,
                strip_prefix='@'))
    elif node.type == _TSValueTypes.TAG:
        return CSTMetadataTag(
            value=_normalize_sugarable_string(
                node,
                strip_prefix='#'))
    elif node.type == _TSValueTypes.VARIABLE:
        return CSTMetadataVariable(
            value=_normalize_sugarable_string(
                node,
                strip_prefix='%'))
    elif node.type == _TSValueTypes.REF:
        return CSTMetadataReference(
            value=_normalize_sugarable_string(
                node,
                strip_prefix='&'))
    elif node.type == _TSValueTypes.URI:
        return _normalize_sugarable_string(node)
    else:
        raise TypeError('Invalid format bracket link target!', node)


@_register(_TSNamedField.NODE_TITLE)
def process_node_title_container(node: TSNode) -> CSTDocumentNodeTitle:
    container = CSTDocumentNodeTitle(content=[])
    for child_cst_node in _process_named_children(node):
        if isinstance(child_cst_node, CSTRichtext | CSTEmptyLine):
            container.add_line(child_cst_node)
        else:
            raise TypeError(
                'Invalid child of node title container!', child_cst_node)

    return container


@_register(_TSNamedField.NODE_METADATA)
def process_node_metadata_container(node: TSNode) -> CSTDocumentNodeMetadata:
    container = CSTDocumentNodeMetadata(content=[])
    for child_cst_node in _process_named_children(node):
        if isinstance(
            child_cst_node,
            CSTMetadataAssignment | CSTAnnotation | CSTEmptyLine
        ):
            container.add_line(child_cst_node)
        else:
            raise TypeError(
                'Invalid child of node metadata container!', child_cst_node)

    return container


@_register(_TSNamedField.INLINE_METADATA)
def process_inline_metadata_container(
        node: TSNode) -> CSTFmtBracketMetadataContainer:
    container = CSTFmtBracketMetadataContainer(content=[])
    for child_cst_node in _process_named_children(node):
        if isinstance(child_cst_node, CSTMetadataAssignment):
            container.content.append(child_cst_node)
        else:
            raise TypeError(
                'Invalid child of inline metadata container!', child_cst_node)

    return container


@_register(_TSNamedField.ANNOTATION_LINE)
def process_annotation_line(node: TSNode) -> CSTAnnotation:
    content = []
    for child_name, child in _extract_named_children(node):
        if (
            child_name == _TSNamedField.ANNOTATION_LINE_CONTENT
            and child.text is not None
        ):
            content.append(child.text.decode())
    return CSTAnnotation(content=[''.join(content)])


def _process_named_children(ts_node: TSNode) -> Iterator[CSTNode]:
    for child_name, child in _extract_named_children(ts_node):
        maybe_cst_node = handle_node(child_name, child)
        if maybe_cst_node is not None:
            yield maybe_cst_node


def _extract_named_children(
        ts_node: TSNode) -> Iterator[tuple[_TSNamedField, TSNode]]:
    """This simply extracts the named children from the given TSNode,
    yielding them back in the same order as they appear in
    ts_node.children.
    """
    for index, child in enumerate(ts_node.children):
        field_name = ts_node.field_name_for_child(index)
        if field_name is not None:
            try:
                yield _TSNamedField(field_name), child
            except ValueError as exc:
                exc.add_note('Unknown field() annotation in grammar.js')
                exc.add_note(f'Grammar name: {ts_node.grammar_name}')
                exc.add_note(f'Grammar type: {ts_node.type}')
                exc.add_note(f'Child index: {index}')
                raise


def _add_or_merge_content(
        line: CSTNode | CSTRichtext | CSTAnnotation,
        # I'd like to be more specific than just plain list here, but mutable
        # collections are invariant, so that would defeat the purpose of
        # having a general-purpose method for merging things.
        # See: https://mypy.readthedocs.io/en/stable/generics.html#variance-of-generics
        content: list
        ) -> None:
    if content:
        previous_line = content[-1]
        if (
            isinstance(line, CSTRichtext)
            and isinstance(previous_line, CSTRichtext)
        ):
            previous_line.content.append(CSTLineBreak(content=None))
            previous_line.content.extend(line.content)
        # Note: the elif here (instead of an OR above) is purely so that the
        # type checker succeeds
        elif (
            isinstance(line, CSTAnnotation)
            and isinstance(previous_line, CSTAnnotation)
        ):
            previous_line.content.append(CSTLineBreak(content=None))
            previous_line.content.extend(line.content)

        else:
            content.append(line)

    else:
        content.append(line)


def _normalize_sugarable_string(
        node: TSNode,
        *,
        strip_prefix: str | None = None
        ) -> CSTMetadataStr:
    """Sugarable strings can be either a sugared string, which is a
    node with no children and only text, or a nested STRING1|STRING2
    within the children of the node.

    This normalizes and flattens both of them into the types expected by
    the metadata values.
    """
    if node.children:
        should_be_string = process_metadata_value(node)

        if not isinstance(should_be_string, CSTMetadataStr):
            raise TypeError(
                'Sugarable string was not a string?', should_be_string)
        return should_be_string
    elif node.text is None:
        raise ValueError('Sugarable string must have content!', node)
    else:
        # This is either a metadata key, which is just a plain string, or
        # a mention, tags, etc. In the case of those special sugars, note that
        # this will include the leading ``@|#|%|&``.
        raw_sugared_string = node.text.decode()

        if strip_prefix is None:
            normalized_value = raw_sugared_string
        else:
            normalized_value = raw_sugared_string.lstrip(strip_prefix)

        return CSTMetadataStr(
            value=normalized_value,
            quote_style=None)
