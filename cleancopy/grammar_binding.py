"""The grammar binding produces a CST from the grammar, using whatever
third-party parser generator library we decide to use.
"""
import itertools
from copy import copy
from dataclasses import dataclass
from decimal import Decimal
from math import inf
from pathlib import Path
from typing import Any
from typing import Collection
from typing import Optional

from lark import Lark
from lark import Transformer
from lark.exceptions import UnexpectedCharacters
from lark.exceptions import UnexpectedToken
from lark.lark import PostLex
from lark.lexer import BasicLexer
from lark.lexer import Lexer
from lark.lexer import Token
from lark.load_grammar import GrammarBuilder

from cleancopy.cst_nodes import CleancopyDocument
from cleancopy.cst_nodes import CommentLine
from cleancopy.cst_nodes import ContentLine
from cleancopy.cst_nodes import DocumentNode
from cleancopy.cst_nodes import EmbedLine
from cleancopy.cst_nodes import EmptyLine
from cleancopy.cst_nodes import PendingNodeEmbedTypeAssignmentLine
from cleancopy.cst_nodes import PendingNodeMetadataLine
from cleancopy.cst_nodes import PendingNodeTitleLine
from cleancopy.cst_nodes import VersionComment
from cleancopy.exceptions import IndentationError
from cleancopy.exceptions import InvalidMetadataValue

_grammar_path = Path(__file__).parent / 'grammar.lark'
_grammar_builder = GrammarBuilder(global_keep_all_tokens=True)
_grammar_builder.load_grammar(
    grammar_text=_grammar_path.read_text(),
    grammar_name='cleancopy')
_GRAMMAR = _grammar_builder.build()

INDENT_DEPTH = 4
TERMINAL_EOL = '_EOL'
TERMINAL_EMPTY_LINE = '_EMPTY_LINE'
TERMINAL_NL_THEN_INDENT = 'NEWLINE_THEN_INDENTATION'
TERMINAL_NL_THEN_EMPTY = 'NEWLINE_THEN_EMPTY'
TERMINAL_BLOCK_BEGIN = '_BLOCK_BEGIN'
TERMINAL_BLOCK_END = '_BLOCK_END'
TERMINAL_COMMENT_BEGIN = 'SYMBOL_COMMENT_BEGIN'
TERMINAL_COMMENT_END = 'SYMBOL_COMMENT_END'
TERMINAL_COMMENT_LINE = 'TEXT_COMMENT_LINE'
TERMINAL_EMBED_LINE = 'TEXT_EMBED_LINE'
TERMINAL_RECOVERED_INDENT = '_RECOVERED_INDENTATION'
TERMINAL_PENDING_EMBED = 'METADATA_KEY_EMBED'
TERMINAL_METADATA_STR = 'METADATA_VALUE_STR'
TERMINAL_METADATA_NULL = 'METADATA_VALUE_NULL'
TERMINAL_METADATA_TRUE = 'METADATA_VALUE_TRUE'
TERMINAL_METADATA_FALSE = 'METADATA_VALUE_FALSE'
TERMINAL_METADATA_NUMERIC = 'METADATA_VALUE_NUMERIC'

_PENDING_NODE_IS_EMPTY = object()


def parse(text):
    parser_container = Lark(
        _GRAMMAR,
        parser='lalr',
        postlex=_ShimShamPostlex(),
        start='document',
        # This is where we insert our custom contextual lexer. It's completely
        # undocumented within lark, but after tons of reading source code, this
        # is the best thing I came up with.
        _plugins={'ContextualLexer': CleancopyLexer})
    raw_tree = parser_container.parse(text)
    return CstTransformer().transform(raw_tree)


def _token_dispatch(terminal_name):
    """This is a super quick and dirty decorator to make it easier to
    declare token type handlers. DO NOT REUSE THIS! It depends on
    global module state to work (I know that's gross!)
    """
    def decorator(func):
        _token_dispatch.registry[terminal_name] = func
        return func
    return decorator


_token_dispatch.registry = {}


def _metadata_value_parser(terminal_name):
    """Use this decorator to mark something as a parser for that
    particular terminal.
    """
    def decorator(func):
        _metadata_value_parser.registry[terminal_name] = func
        return func
    return decorator


_metadata_value_parser.registry = {}


def _get_parent_terminals(terminal_name):
    """This returns a set for any terminals that get added during
    postlex, consisting of the terminals that they are derived from.
    """
    if terminal_name in {TERMINAL_EMPTY_LINE, TERMINAL_EOL}:
        return {TERMINAL_NL_THEN_INDENT, TERMINAL_NL_THEN_EMPTY}
    else:
        return set()


class CleancopyLexer(Lexer):

    def __init__(
            self,
            lexer_conf,
            states: dict[str, Collection[str]],
            always_accept: Collection[str] = None,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

        if always_accept is None:
            always_accept = frozenset()

        terminals_by_name = lexer_conf.terminals_by_name

        # Note that the lark contextual lexer does some regex-based validity
        # checking here if an optional upstream dep is installed

        lexers = self._lexers = {}
        # This lets us reuse the lexer if another parser state accepts the same
        # set of valid next tokens
        lexer_reuse_lookup: dict[frozenset[str], BasicLexer] = {}

        for parser_state_token, valid_next_tokens in states.items():
            lexer_reuse_key = valid_next_tokens = frozenset(valid_next_tokens)

            if lexer_reuse_key in lexer_reuse_lookup:
                child_lexer = lexer_reuse_lookup[lexer_reuse_key]
            else:
                # This can be slightly different than valid_next_tokens
                # depending on our settings
                child_lexer_tokens = (
                    valid_next_tokens
                    | frozenset(lexer_conf.ignore)
                    | frozenset(always_accept)
                    | _get_parent_terminals(parser_state_token))
                child_lexer_conf = copy(lexer_conf)
                child_lexer_conf.terminals = [
                    terminals_by_name[token] for token in child_lexer_tokens
                    if token in terminals_by_name]
                child_lexer = BasicLexer(child_lexer_conf)
                lexer_reuse_lookup[lexer_reuse_key] = child_lexer

            lexers[parser_state_token] = child_lexer

        fallback_lexer_conf = copy(lexer_conf)
        fallback_lexer_conf.terminals = list(lexer_conf.terminals)
        fallback_lexer_conf.skip_validation = True
        self._fallback_lexer = BasicLexer(fallback_lexer_conf)

    def _do_lex(self, clc_lex_state, lexer_state, parser_state):
        """This is called by lex() to handle the actual lexing."""
        token_handlers = _token_dispatch.registry

        while True:
            try:
                current_token = parser_state.position
                contextual_lexer = self._lexers[current_token]
                next_token = contextual_lexer.next_token(
                    lexer_state, parser_state)

                token_type = next_token.type
                if token_type in token_handlers:
                    yield from token_handlers[token_type](
                        clc_lex_state, next_token)
                else:
                    yield next_token

            except EOFError:
                break

        # Okay, so... parsing the end of the file is a little bit tricky.
        # Because of the shenanigans we're doing to work around the lack of
        # zero-width tokens, we can't actually detect an empty line at the end
        # of the file -- we're looking for a line with only whitespace before
        # the **next newline**, but the next newline will never come. But this
        # also means we can't detect the EOL if the file ends on a non-blank
        # line. The solution is to check the last token we found; if it was
        # an EOL, then we KNOW it has to be an empty line, and if not, then
        # we KNOW it wasn't. Either way, we know what to do.
        if clc_lex_state.last_token.type == TERMINAL_EOL:
            yield Token.new_borrow_pos(
                TERMINAL_EMPTY_LINE, next_token, next_token)
        yield Token.new_borrow_pos(TERMINAL_EOL, next_token, next_token)

        while clc_lex_state.indent_level > 0:
            clc_lex_state.indent_level -= 1
            yield Token(TERMINAL_BLOCK_END, '')

    def lex(self, lexer_state, parser_state):
        """This method delegates the actual lexing to self._do_lex(),
        and mostly is just responsible for error handling and logging.
        """
        clc_lex_state = _CleancopyLexState(
            indent_level=0, within_comment=False, within_embed=False,
            pending_embed=False)

        try:
            for next_token in self._do_lex(
                clc_lex_state, lexer_state, parser_state
            ):
                indent_level = clc_lex_state.indent_level
                dbg_token_val = next_token.strip()
                if len(dbg_token_val) > 45:
                    dbg_token_val = f'{dbg_token_val[:42]}...'
                else:
                    dbg_token_val = dbg_token_val.ljust(45)
                dbg_token_desc = f'<L{indent_level} {next_token.type}>'.ljust(
                    40)
                dbg_lex_state = clc_lex_state.debug_repr()
                print(f'{dbg_token_desc}{dbg_token_val}    {dbg_lex_state}')
                # I hate that we're manipulating the state both within and
                # outside of the iterator, but it makes the code much simpler.
                # Hopefully nothing breaks!
                clc_lex_state.last_token = next_token
                next_token.value = _TokenValueWrapper(
                    value=next_token.value, indent_level=indent_level)
                yield next_token

        # Contextual lexers work by restricting the possible set of terminals
        # based on the current parse context. That means that we might have a
        # token that is valid in a different parse context, but invalid here.
        # In that case, we can attempt to do a fallback parse, just to present
        # a nicer parse error in case the terminal is valid elsewhere.
        except UnexpectedCharacters as invalid_parse_exc:
            # The fallback parsing will screw up the parse state, and we can't
            # rewind, so preserve it here.
            last_valid_token = lexer_state.last_token

            try:
                fallback_token = self._fallback_lexer.next_token(
                    lexer_state, parser_state)
            except UnexpectedCharacters:
                # We don't have any valid terminal, in any valid context, for
                # this character, so we can't give a fallback.
                # Raising from None allows us to drop the nested exception and
                # return to the original one, resulting in a nicer traceback.
                raise invalid_parse_exc from None
            else:
                raise UnexpectedToken(
                    fallback_token,
                    invalid_parse_exc.allowed,
                    state=parser_state,
                    token_history=[last_valid_token],
                    terminals_by_name=self._fallback_lexer.terminals_by_name
                ) from None


@dataclass(frozen=True)
class _TokenValueWrapper:
    value: Any
    indent_level: int


@dataclass
class _CleancopyLexState:
    indent_level: int
    pending_embed: bool
    within_comment: bool
    within_embed: bool
    # Set after the first token! Note that this is slightly different than the
    # lark last token, because we modify the tokens being produced during
    # lexing. So these are the tokens we **return** to the parser.
    last_token: Token = None

    def debug_repr(self):
        return (
            f'<Lexstate: pend_emb {self.pending_embed}, ' +
            f'in_emb {self.within_embed}, in_comm {self.within_comment}>')


@_token_dispatch(TERMINAL_NL_THEN_INDENT)
def _process_nl_then_indentation(clc_lex_state, token):
    """Checks for indentation on a newline, potentially yielding
    indent/dedent tokens in addition to the passed token.
    """
    yield Token.new_borrow_pos(TERMINAL_EOL, token, token)

    # Note that both embeds and comments DO care about **shallower**
    # indentation, but ignore deeper indentation.
    recover_indent = clc_lex_state.within_comment or clc_lex_state.within_embed
    detect_deeper_indentation = not recover_indent
    pending_embed = clc_lex_state.pending_embed

    # Note that we might have a carriage return in addition to the
    # newline, so we discard both like this
    # TODO: I think this is actually incorrect, because of the way that we
    # do ordering... can lark tokens define capture groups in regex? That
    # would be a trivial way to get the indent out
    indent_str = token.rsplit('\n', 1)[1]
    if detect_deeper_indentation:
        level_count, indent_levels = _split_by_indent_level(indent_str)
    elif pending_embed:
        level_count, indent_levels = _split_by_indent_level(
            indent_str, max_level=clc_lex_state.indent_level + 1)
    else:
        level_count, indent_levels = _split_by_indent_level(
            indent_str, max_level=clc_lex_state.indent_level)
    level_index = level_count - 1

    while level_index < clc_lex_state.indent_level:
        yield Token.new_borrow_pos(TERMINAL_BLOCK_END, indent_str, token)
        # Note that this is level-triggered instead of edge-triggered -- in
        # other words, we'll just repeatedly set this when dropping out of the
        # indent level. It's equivalent and makes the logic cleaner.
        clc_lex_state.within_embed = False
        clc_lex_state.indent_level -= 1
        recover_indent = False

        # Note: we're not setting within_comment to false here, because
        # de-indenting within a comment is just a syntax error, end of story

    if detect_deeper_indentation:
        while level_index > clc_lex_state.indent_level:
            clc_lex_state.indent_level += 1
            clc_lex_state.pending_embed = False
            yield Token.new_borrow_pos(TERMINAL_BLOCK_BEGIN, indent_str, token)

            # Note that if we were waiting for an embed to start, then this
            # terminal also contains the initial indentation for the first line
            # of the embed block, and we need to recover it
            if pending_embed:
                recover_indent = True

    # Note that we need to re-check this because we might be entering or
    # leaving an embedded block
    if recover_indent:
        # This shouldn't be possible, but hey, just in case...
        if len(indent_levels) < level_count:
            print('Impossible branch!')
            recovered_indentation = ''
        else:
            recovered_indentation = indent_levels[-1]
        yield Token.new_borrow_pos(
            TERMINAL_RECOVERED_INDENT, recovered_indentation, token)


@_token_dispatch(TERMINAL_NL_THEN_EMPTY)
def _process_nl_then_empty(clc_lex_state, token):
    """Lark doesn't support zero-width terminals, so we use a
    special terminal for empty lines that also includes a newline,
    and then we use this to split it into a newline and an empty
    line.
    """
    yield Token.new_borrow_pos(TERMINAL_EOL, token, token)
    yield Token.new_borrow_pos(TERMINAL_EMPTY_LINE, '', token)


@_token_dispatch(TERMINAL_COMMENT_BEGIN)
def _process_comment_begin(clc_lex_state, token):
    """Sets within_comment appropriately, so that we temporarily
    ignore indentation changes within newlines until we exit the
    comment.
    """
    clc_lex_state.within_comment = True
    yield token


@_token_dispatch(TERMINAL_COMMENT_END)
def _process_comment_end(clc_lex_state, token):
    """Resumes normal indentation processing at the end of a comment
    block.
    """
    yield token
    clc_lex_state.within_comment = False


@_token_dispatch(TERMINAL_PENDING_EMBED)
def _process_pending_embed(clc_lex_state, token):
    """Marks us as waiting on an embed line. This allows the
    newline-then-indentation processor to include the recovered
    indentation for the first line of the embed block.
    """
    clc_lex_state.pending_embed = True
    yield token


@_token_dispatch(TERMINAL_EMBED_LINE)
def _process_embed_line(clc_lex_state, token):
    """Disables indent detection within embedded lines. Note that this
    is a bit of the hack; it's a level-detect instead of an edge-detect
    (ie, we re-run this for every embed line).
    """
    clc_lex_state.within_embed = True
    yield token


@_metadata_value_parser(TERMINAL_METADATA_STR)
def _process_metadata_string(token):
    # Strip the quotes and keep the rest
    return token.value.value[1: -1]


@_metadata_value_parser(TERMINAL_METADATA_NULL)
def _process_metadata_null(token):
    return None


@_metadata_value_parser(TERMINAL_METADATA_TRUE)
def _process_metadata_true(token):
    return True


@_metadata_value_parser(TERMINAL_METADATA_FALSE)
def _process_metadata_false(token):
    return False


@_metadata_value_parser(TERMINAL_METADATA_NUMERIC)
def _process_metadata_numeric(token):
    return Decimal(token.value.value)


class _ShimShamPostlex(PostLex):
    """This is a shim class. The ONLY purpose it serves is to pass in
    the always_accept parameter, because that's the only way we have to
    make sure that our derived terminals get kept during grammar
    compilation.
    """
    always_accept = {TERMINAL_NL_THEN_INDENT, TERMINAL_NL_THEN_EMPTY}

    def process(self, stream):
        yield from stream


# TODO: we should rewrite this so that the actual transformer class is
# metaprogrammed, and the individual handlers are done via decorators. (and we
# should move this into a dedicated module!)
class CstTransformer(Transformer):
    """This class transforms a lark parse tree into a cleancopy CST.
    Note that these need to match the grammar EXACTLY, including the
    places where _EOLs are defined.

    Note: keep the method order the same as the definitions in the
    grammar itself, so it's easy to compare them side-by-side.
    """

    def document(self, value):
        version_comment, __, root_node_container = value
        root_node = DocumentNode(
            title_lines=None,
            content_lines=root_node_container.content_lines,
            embed_lines=None,
            metadata_lines=None,
            comment_lines=None)
        return CleancopyDocument(
            version_comment=version_comment,
            document_root=root_node)

    def version_comment(self, value):
        return VersionComment(version=value)

    def node_anchor(self, value):
        content_lines = []

        for line_or_container in value:
            if isinstance(line_or_container, _CommentLinesContainer):
                content_lines.extend(line_or_container.comment_lines)
            else:
                content_lines.append(line_or_container)

        return _NodeAnchorContainer(content_lines=content_lines)

    def pending_node_anchor(self, value):
        title_lines = []
        pending_node_embed = None
        pending_node_content = None
        metadata_lines = None
        # TODO: this needs to be folded into the metadata lines
        comment_lines = None

        for title_or_node in value:
            if isinstance(title_or_node, PendingNodeTitleLine):
                title_lines.append(title_or_node)

            elif isinstance(title_or_node, _PendingNodeContentContainer):
                pending_node_content = title_or_node.content_lines
                metadata_lines = title_or_node.metadata_lines
                comment_lines = title_or_node.comment_lines

            elif isinstance(title_or_node, _PendingNodeEmbedContainer):
                pending_node_embed = title_or_node.embed_lines
                metadata_lines = title_or_node.metadata_lines
                comment_lines = title_or_node.comment_lines

        return DocumentNode(
            title_lines=title_lines,
            content_lines=pending_node_content,
            embed_lines=pending_node_embed,
            comment_lines=comment_lines,
            metadata_lines=metadata_lines)

    def pending_node_content(self, value):
        content_lines = []
        metadata_lines = []
        comment_lines = []

        for parse_tree_child in value:
            if isinstance(parse_tree_child, _NodeAnchorContainer):
                content_lines.extend(parse_tree_child.content_lines)
            elif isinstance(parse_tree_child, _PendingNodeMetadataContainer):
                metadata_lines.extend(parse_tree_child.metadata_lines)
                comment_lines.extend(parse_tree_child.comment_lines)

        return _PendingNodeContentContainer(
            content_lines=content_lines,
            metadata_lines=metadata_lines,
            comment_lines=comment_lines)

    def pending_node_embed(self, value):
        embed_lines = []
        metadata_lines = []
        comment_lines = []

        for tree_child in value:
            if isinstance(tree_child, PendingNodeEmbedTypeAssignmentLine):
                metadata_lines.append(tree_child)
            elif isinstance(tree_child, EmbedLine):
                embed_lines.append(tree_child)
            elif isinstance(tree_child, EmptyLine):
                embed_lines.append(tree_child)
            elif isinstance(tree_child, _PendingNodeMetadataContainer):
                metadata_lines.extend(tree_child.metadata_lines)
                comment_lines.extend(tree_child.comment_lines)

        return _PendingNodeEmbedContainer(
            embed_lines=embed_lines,
            metadata_lines=metadata_lines,
            comment_lines=comment_lines)

    def pending_node_metadata_block(self, value):
        metadata_lines = []
        comment_lines = []
        for line in value:
            if isinstance(line, PendingNodeMetadataLine):
                metadata_lines.append(line)
            elif isinstance(line, _CommentLinesContainer):
                comment_lines.extend(line.comment_lines)
        return _PendingNodeMetadataContainer(
            metadata_lines=metadata_lines,
            comment_lines=comment_lines)

    def node_line_pending_node_empty(self, value):
        return _PENDING_NODE_IS_EMPTY

    def node_line_pending_node_title(self, value):
        __, title_text, __ = value
        return PendingNodeTitleLine(text=str(title_text))

    def node_line_pending_node_metadata(self, value):
        token_key, __, metadata_value, __ = value
        return PendingNodeMetadataLine(
            key=token_key.value,
            value=metadata_value)

    def node_line_pending_node_embed_assignment(self, value):
        __, __, embed_type_assignment, __ = value
        # Note that we're stripping out the quotes for the string here
        return PendingNodeEmbedTypeAssignmentLine(
            value=embed_type_assignment.value.value[1: -1])

    def metadata_value(self, value):
        # This is awkward, but expedient. It's not convenient to expand this
        # within the grammar for two reasons: first, one of our terminals is
        # imported, and second, we use METADATA_VALUE_STR directly within the
        # embed type declaration
        token, = value
        try:
            typecaster = _metadata_value_parser.registry[token.type]
        except KeyError as exc:
            raise InvalidMetadataValue(
                'Invalid token type for metadata value!') from exc

        try:
            return typecaster(token)
        except (ValueError, TypeError) as exc:
            raise InvalidMetadataValue(
                'Text could not be parsed into a valid metadata value!'
            ) from exc

    def node_line_empty(self, value):
        # empty_line, EOL
        token, __ = value
        return EmptyLine(
            # Note that this is including the \n from the previous line, so we
            # need to artificially drop that.
            # start_line=token.line,
            # start_col=token.column,
            start_line=token.end_line,
            start_col=1,
            end_line=token.end_line,
            end_col=token.end_column)

    def node_line_content(self, value):
        token_line, __ = value
        return ContentLine(text=token_line.value)

    def node_line_embed(self, value):
        is_empty = None
        recovered_indentation = None
        embed_line_wrapper = None

        for child_token in value:
            if child_token.type == TERMINAL_EMBED_LINE:
                is_empty = False
                embed_line_wrapper = child_token.value
            elif child_token.type == TERMINAL_RECOVERED_INDENT:
                # Note that this also extracts the value from the token wrapper
                recovered_indentation = child_token.value.value
            elif child_token.type == TERMINAL_EMPTY_LINE:
                is_empty = True

        if is_empty:
            return EmptyLine()

        else:
            text_wrapper = _TokenValueWrapper(
                value=recovered_indentation + embed_line_wrapper.value,
                indent_level=embed_line_wrapper.indent_level)
            return EmbedLine(text=text_wrapper)

    def comment_lines(self, value):
        comment_lines = []
        previous_token_indent = None
        for token in value:
            if token.type == TERMINAL_COMMENT_LINE:
                if previous_token_indent is None:
                    comment_text_wrapper = token.value
                else:
                    old_wrapper = token.value
                    comment_text_wrapper = _TokenValueWrapper(
                        value=previous_token_indent + old_wrapper.value,
                        indent_level=old_wrapper.indent_level)
                    previous_token_indent = None

                comment_lines.append(CommentLine(text=comment_text_wrapper))

            elif token.type == TERMINAL_RECOVERED_INDENT:
                # Note that this also extracts the value from the token wrapper
                previous_token_indent = token.value.value

        return _CommentLinesContainer(comment_lines=comment_lines)


@dataclass
class _CommentLinesContainer:
    comment_lines: list


@dataclass
class _PendingNodeMetadataContainer:
    metadata_lines: list
    comment_lines: list


@dataclass
class _PendingNodeContentContainer:
    content_lines: list
    metadata_lines: list
    comment_lines: list


@dataclass
class _PendingNodeEmbedContainer:
    embed_lines: list
    metadata_lines: list
    comment_lines: list


@dataclass
class _NodeAnchorContainer:
    content_lines: list


def _split_by_indent_level(indentation: str, max_level: Optional[int] = inf):
    """This will split an indentation string into substring(s), one for
    each indent level encountered. It also checks for both partial
    indentation, as well as mixed tabs/spaces within a particular
    indentation level. It returns a tuple: (level_count, [levels]).

    If max_level is given, then it will stop splitting AFTER the
    max_level, leaving the remaining whitespace untouched (as the last
    item in the split). Be sure to check against the level_count here;
    you might be de-indenting, and we DON'T raise in that case!
    """
    level_start_indices = [0]
    char_count = len(indentation)
    current_level = 1
    cursor = 0

    # First we figure out what substrings *should* correspond to given levels
    while cursor < char_count and current_level <= max_level:
        if indentation[cursor] == '\t':
            cursor += 1
        else:
            cursor += INDENT_DEPTH

        level_start_indices.append(cursor)
        current_level += 1
    # Just for clarity: this is no longer the current level, because it just
    # failed the while condition. So this is now the previous current level
    # plus one, which is the same as the level count.
    level_count = current_level

    # Next, we create the actual substrings, and make sure they don't mix
    # tabs and spaces internally. Note that the zeroth indentation level is
    # always the empty string!
    indentation_levels = ['']
    # Note: we need this here so that the level_end binding from the for loop
    # gets the function scope instead of the loop scope
    level_end = 0
    for level_start, level_end in itertools.pairwise(level_start_indices):
        # This must be a tab character; we don't need to do an expensive string
        # operation to get it back
        if level_end - level_start == 1:
            indentation_levels.append('\t')
        else:
            substring = indentation[level_start: level_end]
            if '\t' in substring:
                raise IndentationError(
                    'Cannot mix tabs and spaces within a single indent level!')
            indentation_levels.append(substring)

    # Don't forget that pairwise will only use the last level_start as the
    # endpoint for the second-to-last indentation level!
    last_substring = indentation[level_end:]
    indentation_levels.append(last_substring)

    # Finally, we do some other checks on the last indentation segment, but
    # only if we were asked to split the whole indentation string.
    if max_level is inf:
        # This would indicate that the deepest indentation level is incomplete.
        # We don't need to bother getting a substring for it.
        if cursor != char_count:
            raise IndentationError('Partial indentation is invalid!')
        if len(last_substring) > 1 and '\t' in last_substring:
            raise IndentationError(
                'Cannot mix tabs and spaces within a single indent level!')

    return level_count, indentation_levels


def test():
    from pathlib import Path
    from cleancopy.grammar_binding import parse

    testvector_dir = Path('./tests/testdata/testvectors_clc')
    testvectors = ['sample_1.clc', 'sample_2.clc', 'sample_3.clc']

    parse_results = []
    for filename in testvectors:
        print(f'#################### {filename} ####################')
        test_doc = (testvector_dir / filename).read_text()
        parse_results.append(parse(test_doc))

    return parse_results
