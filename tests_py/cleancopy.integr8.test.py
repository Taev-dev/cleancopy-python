"""This contains a bunch of spot-check snippets: slightly less involved
than the e2e tests that are looking at whole documents, but still not
unit tests.
"""
from __future__ import annotations

from cleancopy import Abstractifier
from cleancopy import parse
from cleancopy.ast import Document
from cleancopy.ast import Paragraph
from cleancopy.ast import RichtextBlockNode
from cleancopy.utils import dedoc

from cleancopy_testutils import doc_prep


class TestDocument:

    def test_doc_has_root(self):
        transformer = Abstractifier()
        result = transformer.convert(parse(b'foo'))

        assert isinstance(result, Document)
        assert isinstance(result.root, RichtextBlockNode)
        assert result.title is None
        assert result.info is None

    def test_root_depth(self):
        transformer = Abstractifier()
        result = transformer.convert(parse(b'foo'))

        assert isinstance(result, Document)
        assert isinstance(result.root, RichtextBlockNode)
        assert result.root.depth == 0

    def test_doc_with_title(self):
        transformer = Abstractifier()
        result = transformer.convert(parse(doc_prep('''
            > Foo
            __doc_meta__: true
            <''')))

        assert isinstance(result, Document)
        assert isinstance(result.root, RichtextBlockNode)
        assert result.title is not None
        assert result.title.content == ['Foo']
        assert result.info is not None
        assert not result.info.metadata

    def test_doc_with_metadata(self):
        transformer = Abstractifier()
        result = transformer.convert(parse(doc_prep('''
            >
            __doc_meta__: true
            foo: 'oof'
            bar: 'rab'
            <''')))

        assert isinstance(result, Document)
        assert isinstance(result.root, RichtextBlockNode)
        assert result.title is None
        assert result.info is not None
        assert 'foo' in result.info.metadata
        assert 'bar' in result.info.metadata


class TestRichtextBlockNode:

    def test_content_is_paragraphs(self):
        transformer = Abstractifier()
        result = dedoc(transformer.convert(parse(doc_prep('''
            foo

            >
                bar'''))))

        assert len(result) == 2
        assert isinstance(result[0], Paragraph)
        assert isinstance(result[1], RichtextBlockNode)
        assert len(result[1].content) == 1
        assert isinstance(result[1].content[0], Paragraph)

    def test_depths(self):
        transformer = Abstractifier()
        result = dedoc(transformer.convert(parse(doc_prep('''
            foo

            >
                bar

                >
                    baz'''))))

        assert isinstance(result[0], Paragraph)
        assert isinstance(result[1], RichtextBlockNode)
        assert result[1].depth == 1
        assert isinstance(result[1].content[1], RichtextBlockNode)
        assert result[1].content[1].depth == 2
