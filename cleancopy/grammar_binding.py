"""The grammar binding produces a CST from the grammar, using whatever
third-party parser generator library we decide to use.
"""
from copy import copy
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
TERMINAL_INDENT = '_INDENT'
TERMINAL_DEDENT = '_DEDENT'
TERMINAL_COMMENT_BEGIN = '_COMMENT_BEGIN'
TERMINAL_COMMENT_END = '_COMMENT_END'


def parse(text):
    parser_container = Lark(
        _GRAMMAR,
        parser='lalr',
        postlex=CleancopyPostlexer(),
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
        _token_dispatch.registry[terminal_name] = func.__name__
        return func
    return decorator


_token_dispatch.registry = {}


"""
NOTE:
Okay, so, it turns out that the default lexer doesn't look into the
terminals that might result in a postlex additional token, UNLESS it also
matches another found terminal.

As a workaround, I just threw in the temporary terminals as valid parts of
the grammar, even though they'll just be immediately postlexed away. This
got me past the first problem, but now it's failing after the first dedent
token.

I think, maybe, just maybe, the issue is that there's some funkiness where
in order to create the dedent token, something funky is happening and there's
a conflict of sorts between the dedent and the workaround.

I'm not 100% sure, but I think a solution MIGHT be to create a custom lexer.
"""


class CleancopyLexer(Lexer):

    def __init__(
            self,
            lexer_conf,
            states: dict[str, Collection[str]],
            always_accept: Collection[str] = None):

        if always_accept is None:
            always_accept = frozenset()

        terminals_by_name = lexer_conf.terminals_by_name

        # Note that the lark contextual lexer does some regex-based validity
        # checking here if an optional upstream dep is installed

        lexers = self._lexers = {}
        # This lets us reuse the lexer if another parser state accepts the same
        # set of valid next tokens
        lexer_reuse_lookup: dict[frozenset[str], BasicLexer] = {}

        for parser_state, valid_next_tokens in states.items():
            lexer_reuse_key = valid_next_tokens = frozenset(valid_next_tokens)

            if lexer_reuse_key in lexer_reuse_lookup:
                child_lexer = lexer_reuse_lookup[lexer_reuse_key]
            else:
                # This can be slightly different than valid_next_tokens
                # depending on our settings
                child_lexer_tokens = (
                    valid_next_tokens
                    | frozenset(lexer_conf.ignore)
                    | frozenset(always_accept))
                child_lexer_conf = copy(lexer_conf)
                child_lexer_conf.terminals = [
                    terminals_by_name[token] for token in child_lexer_tokens
                    if token in terminals_by_name]
                child_lexer = BasicLexer(child_lexer_conf)
                lexer_reuse_lookup[lexer_reuse_key] = child_lexer

            lexers[parser_state] = child_lexer

        fallback_lexer_conf = copy(lexer_conf)
        fallback_lexer_conf.terminals = list(lexer_conf.terminals)
        fallback_lexer_conf.skip_validation = True
        self._fallback_lexer = BasicLexer(fallback_lexer_conf)

    def lex(self, lexer_state, parser_state):
        try:
            while True:
                try:
                    current_token = parser_state.position
                    contextual_lexer = self._lexers[current_token]
                    yield contextual_lexer.next_token(
                        lexer_state, parser_state)
                except EOFError:
                    break

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


class CleancopyPostlexer(PostLex):
    indent_level: int
    within_comment: bool

    def __init__(self):
        self.within_comment = False
        self.indent_level = 0

    def process(self, token_stream):
        for token in self._process(token_stream):
            print(f'<{token.type}>: {token.strip()}')
            yield token

    def _process(self, token_stream, _token_handlers=_token_dispatch.registry):
        for token in token_stream:
            terminal_type = token.type
            if terminal_type in _token_handlers:
                yield from getattr(self, _token_handlers[terminal_type])(token)
            else:
                yield token

        while self.indent_level > 0:
            self.indent_level -= 1
            yield Token(TERMINAL_DEDENT, '')

    @_token_dispatch(TERMINAL_NL_THEN_INDENT)
    def process_nl_then_indentation(self, token):
        """Checks for indentation on a newline, potentially yielding
        indent/dedent tokens in addition to the passed token.
        """
        yield Token.new_borrow_pos(TERMINAL_EOL, token, token)

        if self.within_comment:
            return

        # Note that we might have a carriage return in addition to the
        # newline, so we discard both like this
        indent_str = token.rsplit('\n', 1)[1]
        indent_level = _get_indent_level(indent_str)

        while indent_level > self.indent_level:
            self.indent_level += 1
            yield Token.new_borrow_pos(TERMINAL_INDENT, indent_str, token)

        while indent_level < self.indent_level:
            self.indent_level -= 1
            yield Token.new_borrow_pos(TERMINAL_DEDENT, indent_str, token)

    @_token_dispatch(TERMINAL_NL_THEN_EMPTY)
    def process_nl_then_empty(self, token):
        """Lark doesn't support zero-width terminals, so we use a
        special terminal for empty lines that also includes a newline,
        and then we use this to split it into a newline and an empty
        line.
        """
        yield Token.new_borrow_pos(TERMINAL_EOL, token, token)
        yield Token.new_borrow_pos(TERMINAL_EMPTY_LINE, '', token)

    @_token_dispatch(TERMINAL_COMMENT_BEGIN)
    def process_comment_begin(self, token):
        """Sets within_comment appropriately, so that we temporarily
        ignore indentation changes within newlines until we exit the
        comment.
        """
        self.within_comment = True
        yield token

    @_token_dispatch(TERMINAL_COMMENT_END)
    def process_comment_end(self, token):
        """Resumes normal indentation processing at the end of a comment
        block.
        """
        self.within_comment = False
        yield token


class CstTransformer(Transformer):
    """This class transforms a lark parse tree into a cleancopy CST."""

    def version_comment(self, value):
        return VersionComment(version=value)

    def empty_line(self, value):
        return EmptyLine(token=value)

    def node_root(self, value):
        return DocumentNode(content_lines=list(value))

    def content_line(self, value):
        return ContentLine(token=value)

    # def document(self):
    #     return ClcDocumentNode()


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
    test_path_2 = Path('./tests/testdata/testvectors_clc/sample_2.clc')
    test_doc_2 = test_path_2.read_text()
    return parse(test_doc_2)
