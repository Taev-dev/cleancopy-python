"""The grammar binding produces a CST from the grammar, using whatever
third-party parser generator library we decide to use.
"""
import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lark import Lark
from lark import Transformer
from lark.common import LexerConf
from lark.common import ParserConf
from lark.lark import LarkOptions
from lark.lark import PostLex
from lark.lexer import ContextualLexer
from lark.lexer import Token
from lark.load_grammar import GrammarBuilder
from lark.parser_frontends import ParsingFrontend

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
    raw_tree = _build_lark_parse_method()(text)
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


# Might need to define actual lark options. See class LarkOptions within
# lark/lark.py
def _build_lark_parse_method():
    postlex = CleancopyPostlexer()
    start_rule = 'document'
    options = LarkOptions(options_dict=dict(
        # debug=False,
        # keep_all_tokens=False,
        # tree_class=None,
        # cache=False,
        postlex=postlex,
        parser='lalr',
        start=[start_rule],
        transformer=None,
        # This is where we insert our custom contextual lexer
        _plugins={'ContextualLexer': ContextualLexer}))
    lexer_conf, parser_conf = _build_lark_confs(
        start=start_rule,
        options=options,
        postlex=postlex)
    lark_frontend = ParsingFrontend(lexer_conf, parser_conf, options)
    return functools.partial(
        lark_frontend.parse, start=start_rule, on_error=None)


def _build_lark_confs(*, start, postlex, options, terminals_to_keep=None):
    """This bypasses the vast majority of... detritus... within lark,
    building both a parser and a lexer configuration directly. This
    makes it easier for us to use our own custom lexer.
    """
    if terminals_to_keep is None:
        terminals_to_keep = set()

    # This compiles the EBNF grammar into BNF
    terminals, rules, ignore_tokens = _GRAMMAR.compile(
        [start], terminals_to_keep)
    lexer_conf = LexerConf(
        terminals=terminals,
        re_module=re,
        ignore=ignore_tokens,
        postlex=postlex,
        # The rest is just duplicating defaults, because this is completely
        # undocumented within lark -- it's faster to just reproduce it here
        # than hunt around in the source code there
        callbacks=None,
        g_regex_flags=0,
        skip_validation=False,
        use_bytes=False)

    lexer_conf.lexer_type = 'contextual'

    callbacks = _get_lexer_callbacks(options.transformer, terminals)
    parser_conf = ParserConf(rules, callbacks, [start])
    parser_conf.parser_type = 'lalr'

    return lexer_conf, parser_conf


def _get_lexer_callbacks(transformer, terminals):
    """This is basically a direct duplicate of the lark source code.
    We're reproducing it here for two reasons: first, it's not part of
    the public lark API, and second, because it documents exactly where
    the "magic" happens that ties the terminal name to the methods to
    be called within the transformer. So we could, in theory, use this
    as a hook to replace the magic make-sure-the-names-are-the-same
    detection with a decorator-based one.
    """
    callbacks = {}
    for terminal in terminals:
        this_terminal_name = terminal.name
        # This is where the "magic" happens: it matches the terminal name with
        # a method defined on the transformer. If it finds one, it defines it
        # as a callback for the lexer for that particular terminal name.
        maybe_callback = getattr(transformer, this_terminal_name, None)
        if maybe_callback is not None:
            callbacks[this_terminal_name] = maybe_callback

    return callbacks


def test():
    from pathlib import Path
    from cleancopy.grammar_binding import parse
    test_path_2 = Path('./tests/testdata/testvectors_clc/sample_2.clc')
    test_doc_2 = test_path_2.read_text()
    return parse(test_doc_2)
