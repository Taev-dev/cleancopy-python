"""This file contains all of the valid CST nodes. The first parse of
a cleancopy file will generate a CST. That CST can then be fed into
a separate parser, which will generate an AST from the CST.

See here for details on how to write the lexer for that second parser:
https://lark-parser.readthedocs.io/en/latest/examples/advanced/
    custom_lexer.html
"""
import textwrap
from dataclasses import dataclass
from typing import Any


@dataclass(kw_only=True)
class CSTNode:
    # Note: though *most* things are single lines, we also assign these to
    # line collections like document nodes, so start_line and end_line aren't
    # ALWAYS the same.
    start_line: int = None
    end_line: int = None
    start_col: int = None
    end_col: int = None


@dataclass(kw_only=True)
class DocumentNode(CSTNode):
    title_lines: Any
    metadata_lines: Any
    content_lines: Any

    def prettify(self):
        all_lines = []

        if self.title_lines is not None:
            for line in self.title_lines:
                all_lines.append(textwrap.indent(str(line), '    '))
        if self.metadata_lines is not None:
            for line in self.metadata_lines:
                all_lines.append(textwrap.indent(str(line), '    '))
        if self.content_lines is not None:
            for line in self.content_lines:
                if hasattr(line, 'prettify'):
                    nested_lines = line.prettify()
                    all_lines.append(textwrap.indent(nested_lines, '    '))
                else:
                    all_lines.append(textwrap.indent(str(line), '    '))

        return '\n'.join(all_lines)


@dataclass(kw_only=True)
class VersionComment(CSTNode):
    version: str


@dataclass(kw_only=True)
class EmptyLine(CSTNode):
    pass


@dataclass(kw_only=True)
class ContentLine(CSTNode):
    text: str


@dataclass(kw_only=True)
class CleancopyDocument:
    version_comment: VersionComment
    document_root: DocumentNode

    def prettify(self):
        return self.document_root.prettify()


@dataclass(kw_only=True)
class PendingNodeTitleLine(CSTNode):
    text: str


@dataclass(kw_only=True)
class PendingNodeMetadataLine(CSTNode):
    key: str
    value: str
