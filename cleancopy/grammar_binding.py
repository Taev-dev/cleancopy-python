"""The grammar binding produces a CST from the grammar, using whatever
third-party parser generator library we decide to use.
"""
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Collection

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
from cleancopy.cst_nodes import ContentLine
from cleancopy.cst_nodes import DocumentNode
from cleancopy.cst_nodes import EmptyLine
from cleancopy.cst_nodes import PendingNodeMetadataLine
from cleancopy.cst_nodes import PendingNodeTitleLine
from cleancopy.cst_nodes import VersionComment
from cleancopy.exceptions import IndentationError

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
TERMINAL_COMMENT_BEGIN = '_COMMENT_BEGIN'
TERMINAL_COMMENT_END = '_COMMENT_END'

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
        print('entering lexer')
        clc_lex_state = _CleancopyLexState(
            indent_level=0, within_comment=False)

        try:
            for next_token in self._do_lex(
                clc_lex_state, lexer_state, parser_state
            ):
                print(f'<{next_token.type}>: {next_token.strip()}')
                # I hate that we're manipulating the state both within and
                # outside of the iterator, but it makes the code much simpler.
                # Hopefully nothing breaks!
                clc_lex_state.last_token = next_token
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


@dataclass
class _CleancopyLexState:
    indent_level: int
    within_comment: bool
    # Set after the first token! Note that this is slightly different than the
    # lark last token, because we modify the tokens being produced during
    # lexing. So these are the tokens we **return** to the parser.
    last_token: Token = None


@_token_dispatch(TERMINAL_NL_THEN_INDENT)
def _process_nl_then_indentation(clc_lex_state, token):
    """Checks for indentation on a newline, potentially yielding
    indent/dedent tokens in addition to the passed token.
    """
    yield Token.new_borrow_pos(TERMINAL_EOL, token, token)
    if clc_lex_state.within_comment:
        # yield Token.new_borrow_pos(TERMINAL_EOL, token, token)
        return

    # Note that we might have a carriage return in addition to the
    # newline, so we discard both like this
    indent_str = token.rsplit('\n', 1)[1]
    indent_level = _get_indent_level(indent_str)

    while indent_level > clc_lex_state.indent_level:
        clc_lex_state.indent_level += 1
        yield Token.new_borrow_pos(TERMINAL_BLOCK_BEGIN, indent_str, token)

    while indent_level < clc_lex_state.indent_level:
        clc_lex_state.indent_level -= 1
        yield Token.new_borrow_pos(TERMINAL_BLOCK_END, indent_str, token)


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
    clc_lex_state.within_comment = False
    yield token


class _ShimShamPostlex(PostLex):
    """This is a shim class. The ONLY purpose it serves is to pass in
    the always_accept parameter, because that's the only way we have to
    make sure that our derived terminals get kept during grammar
    compilation.
    """
    always_accept = {TERMINAL_NL_THEN_INDENT, TERMINAL_NL_THEN_EMPTY}

    def process(self, stream):
        yield from stream


class CstTransformer(Transformer):
    """This class transforms a lark parse tree into a cleancopy CST.
    Note that these need to match the grammar EXACTLY, including the
    places where _EOLs are defined.
    """

    def version_comment(self, value):
        return VersionComment(version=value)

    def document(self, value):
        version_comment, __, root_node_lines = value
        root_node = DocumentNode(
            title_lines=None,
            content_lines=root_node_lines,
            metadata_lines=None)
        return CleancopyDocument(
            version_comment=version_comment,
            document_root=root_node)

    # Starting here, we're adding in some better organization
    #################################

    def node_anchor(self, value):
        # Note that the type of this needs to match up with the if/elif inside
        # pending_node_content
        return list(value)

    def pending_node_anchor(self, value):
        return value[0]

    def pending_node_empty(self, value):
        title_lines = []
        metadata_lines = []

        for lark_parse_tree_child in value:
            if isinstance(lark_parse_tree_child, PendingNodeMetadataLine):
                metadata_lines.append(lark_parse_tree_child)
            elif isinstance(lark_parse_tree_child, PendingNodeTitleLine):
                title_lines.append(lark_parse_tree_child)

        return DocumentNode(
            title_lines=title_lines,
            content_lines=None,
            metadata_lines=metadata_lines)

    def pending_node_content(self, value):
        title_lines = []
        content_lines = []
        metadata_lines = []

        for lark_parse_tree_child in value:
            if isinstance(lark_parse_tree_child, list):
                content_lines.extend(lark_parse_tree_child)
            elif isinstance(lark_parse_tree_child, PendingNodeMetadataLine):
                metadata_lines.append(lark_parse_tree_child)
            elif isinstance(lark_parse_tree_child, PendingNodeTitleLine):
                title_lines.append(lark_parse_tree_child)

        return DocumentNode(
            title_lines=title_lines,
            content_lines=content_lines,
            metadata_lines=metadata_lines)

    def node_line_pending_node_empty(self, value):
        return _PENDING_NODE_IS_EMPTY

    def node_line_pending_node_title(self, value):
        __, title_text, __ = value
        return PendingNodeTitleLine(text=str(title_text))

    def node_line_pending_node_metadata(self, value):
        token_key, __, token_value, __ = value
        return PendingNodeMetadataLine(
            key=token_key.value,
            value=token_value.value)

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


def _get_indent_level(indentation: str):
    tab_count = indentation.count('\t')
    total_whitespace_count = len(indentation)
    spacelike_count = total_whitespace_count - tab_count

    if spacelike_count % INDENT_DEPTH:
        raise IndentationError('Partial indentation is invalid!')

    spacelike_indent_level = spacelike_count // INDENT_DEPTH
    return spacelike_indent_level + tab_count


def test():
    from pathlib import Path
    from cleancopy.grammar_binding import parse
    test_path_1 = Path('./tests/testdata/testvectors_clc/sample_1.clc')
    test_doc_1 = test_path_1.read_text()
    test_path_2 = Path('./tests/testdata/testvectors_clc/sample_2.clc')
    test_doc_2 = test_path_2.read_text()
    return parse(test_doc_1), parse(test_doc_2)
