"""This contains a bunch of spot-check snippets: slightly less involved
than the e2e tests that are looking at whole documents, but still not
unit tests.
"""
from __future__ import annotations

from decimal import Decimal
from typing import cast

from cleancopy import Abstractifier
from cleancopy import parse
from cleancopy.ast import BoolDataType
from cleancopy.ast import DecimalDataType
from cleancopy.ast import Document
from cleancopy.ast import EmbeddingBlockNode
from cleancopy.ast import InlineNodeInfo
from cleancopy.ast import IntDataType
from cleancopy.ast import List_
from cleancopy.ast import ListItem
from cleancopy.ast import MentionDataType
from cleancopy.ast import NullDataType
from cleancopy.ast import Paragraph
from cleancopy.ast import ReferenceDataType
from cleancopy.ast import RichtextBlockNode
from cleancopy.ast import RichtextInlineNode
from cleancopy.ast import StrDataType
from cleancopy.ast import TagDataType
from cleancopy.ast import VariableDataType
from cleancopy.spectypes import BlockFormatting
from cleancopy.spectypes import InlineFormatting
from cleancopy.spectypes import ListType
from cleancopy.utils import dedoc

from cleancopy_testutils import doc_prep


class TestDocument:

    def test_doc_has_root(self):
        """Documents must always have a root node. That root node must
        always be a richtext bloci node.
        """
        transformer = Abstractifier()
        result = transformer.convert(parse(b'foo'))

        assert isinstance(result, Document)
        assert isinstance(result.root, RichtextBlockNode)
        assert result.title is None
        assert result.info is None

    def test_root_depth(self):
        """The depth of the root node must always be zero.
        """
        transformer = Abstractifier()
        result = transformer.convert(parse(b'foo'))

        assert isinstance(result, Document)
        assert isinstance(result.root, RichtextBlockNode)
        assert result.root.depth == 0

    def test_doc_with_title(self):
        """Applying the doc meta metadata tag to a node must result in
        it becoming the document's metadata, causing it to apply the
        title from the node to the document itself.
        """
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
        """Applying the doc meta metadata tag to a node must result in
        it becoming the document's metadata, causing it to apply the
        metadata from the node to the document itself.
        """
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

    def test_no_title(self):
        """A node with an empty title must report its title as None.
        """
        transformer = Abstractifier()
        result = dedoc(transformer.convert(parse(doc_prep('''
            >
            <'''))))

        assert len(result) == 1
        assert isinstance(result[0], RichtextBlockNode)
        assert result[0].title is None

    def test_plain_title(self):
        """A node with a plaintext title must result in a single-layer
        inline richtext node, and multiple lines must be collapsed into
        a single content entry.
        """
        transformer = Abstractifier()
        result = dedoc(transformer.convert(parse(doc_prep('''
            > I have a title
            > and it has multiple lines
            > but no richtext
            <'''))))

        assert len(result) == 1
        assert isinstance(result[0], RichtextBlockNode)
        assert result[0].title is not None
        assert result[0].title.content == [
            'I have a title and it has multiple lines but no richtext']

    def test_richtext_title(self):
        """A node with a richtext title must obey the richtext nesting
        and formatting rules.
        """
        transformer = Abstractifier()
        result = dedoc(transformer.convert(parse(doc_prep('''
            > I have a title
            > and it has multiple lines
            > [[**and** ^^some^^ richtext](https://link.example)]
            <'''))))

        assert len(result) == 1
        assert isinstance(result[0], RichtextBlockNode)
        assert result[0].title is not None
        assert len(result[0].title.content) == 2
        assert result[0].title.content[0] == \
            'I have a title and it has multiple lines '

        richtext = result[0].title.content[1]
        assert isinstance(richtext, RichtextInlineNode)

        rec_strs, rec_infos = richtext.recursive_strip()
        assert rec_strs == [['and'], ' ', ['some'], ' richtext']
        assert rec_infos.flatten([0]) == [
            InlineNodeInfo(
                target=StrDataType('https://link.example'),
                sugared=BoolDataType(True)),
            InlineNodeInfo(
                formatting=InlineFormatting.STRONG,
                sugared=BoolDataType(True))]
        assert rec_infos.flatten([2]) == [
            InlineNodeInfo(
                target=StrDataType('https://link.example'),
                sugared=BoolDataType(True)),
            InlineNodeInfo(
                formatting=InlineFormatting.EMPHASIS,
                sugared=BoolDataType(True))]

    def test_content_is_paragraphs(self):
        """All content within a richtext block node not included within
        a child node must be within a paragraph.

        Note that this applies to both the root node and any nested
        nodes, and that the abstractification logic is slightly
        different between them.
        """
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
        """Nested nodes must increase the value of node depth that they
        report.
        """
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

    def test_quote(self):
        """Nodes designated as quotes must have their quote attributes
        set accordingly. These must remain as their ``DataType``
        instances and not be coerced into the underlying values.
        """
        transformer = Abstractifier()
        result = dedoc(transformer.convert(parse(doc_prep('''
            foo

            >
            __formatting__: '__quote__'
            __citation__: 'confucious'
                ## said the wise man...
                bar'''))))

        assert isinstance(result[0], Paragraph)
        assert isinstance(result[1], RichtextBlockNode)
        blocknode = result[1]
        assert blocknode.info is not None
        assert blocknode.info.formatting is not None
        assert blocknode.info.formatting is BlockFormatting.QUOTE
        assert blocknode.info.citation is not None
        assert blocknode.info.citation == StrDataType('confucious')

    def test_arbitrary_metadata(self):
        """Nodes with arbitrary metadata must include that metadata
        within their node info.
        """
        transformer = Abstractifier()
        result = dedoc(transformer.convert(parse(doc_prep('''
            foo

            >
            foo: 'oof'
            bar: 'rab'
                some inner text'''))))

        assert isinstance(result[1], RichtextBlockNode)
        blocknode = result[1]
        assert blocknode.info is not None
        assert 'foo' in blocknode.info.metadata
        assert 'bar' in blocknode.info.metadata
        assert blocknode.info.metadata['foo'] == StrDataType('oof')
        assert blocknode.info.metadata['bar'] == StrDataType('rab')

    def test_metadata_types(self):
        """All node metadata types must be processed according to spec,
        and be included in the nodeinfo's metadata was instances of
        their metadata type.
        """
        transformer = Abstractifier()
        result = dedoc(transformer.convert(parse(doc_prep('''
            foo

            >
            missing: __missing__
            null_: null
            int: 123
            dec: 123.456
            bool: true
            mention: @foo
            mention2: @'foo[]'
            tag: #tag
            tag2: #'tag[]'
            var: %var
            var2: %'var[]'
            ref: &ref
            ref2: &'ref[]'
            <'''))))

        assert isinstance(result[1], RichtextBlockNode)
        blocknode = result[1]
        assert blocknode.info is not None

        assert 'missing' not in blocknode.info.metadata
        assert isinstance(blocknode.info.metadata['null_'], NullDataType)
        assert isinstance(blocknode.info.metadata['int'], IntDataType)
        assert isinstance(blocknode.info.metadata['dec'], DecimalDataType)
        assert isinstance(blocknode.info.metadata['bool'], BoolDataType)
        assert isinstance(blocknode.info.metadata['mention'], MentionDataType)
        assert isinstance(blocknode.info.metadata['mention2'], MentionDataType)
        assert isinstance(blocknode.info.metadata['tag'], TagDataType)
        assert isinstance(blocknode.info.metadata['tag2'], TagDataType)
        assert isinstance(blocknode.info.metadata['var'], VariableDataType)
        assert isinstance(blocknode.info.metadata['var2'], VariableDataType)
        assert isinstance(blocknode.info.metadata['ref'], ReferenceDataType)
        assert isinstance(blocknode.info.metadata['ref2'], ReferenceDataType)

        assert blocknode.info.metadata['null_'].value is None
        assert blocknode.info.metadata['int'].value == 123
        assert blocknode.info.metadata['dec'].value == Decimal('123.456')
        assert blocknode.info.metadata['bool'].value is True
        assert blocknode.info.metadata['mention'].value == 'foo'
        assert blocknode.info.metadata['mention2'].value == 'foo[]'
        assert blocknode.info.metadata['tag'].value == 'tag'
        assert blocknode.info.metadata['tag2'].value == 'tag[]'
        assert blocknode.info.metadata['var'].value == 'var'
        assert blocknode.info.metadata['var2'].value == 'var[]'
        assert blocknode.info.metadata['ref'].value == 'ref'
        assert blocknode.info.metadata['ref2'].value == 'ref[]'


class TestEmbeddedBlockNode:

    def test_empty_node(self):
        """Empty nodes marked as embeddings must still result in
        embedding block nodes.
        """
        transformer = Abstractifier()
        result = dedoc(transformer.convert(parse(doc_prep('''
            >
            __embed__: 'text'
            <'''))))

        assert len(result) == 1
        assert isinstance(result[0], EmbeddingBlockNode)
        assert result[0].content is None

    def test_empty_node_with_metadata(self):
        """Embedding block nodes (empty or otherwise, but we're only
        spot-checking empty nodes) must include any specified arbitrary
        metadata within their nodeinfo metadata.
        """
        transformer = Abstractifier()
        result = dedoc(transformer.convert(parse(doc_prep('''
            >
            __embed__: 'text'
            foo: 'oof'
            <'''))))

        assert len(result) == 1
        assert isinstance(result[0], EmbeddingBlockNode)
        assert result[0].content is None
        assert result[0].info is not None
        assert 'foo' in result[0].info.metadata
        assert result[0].info.metadata['foo'] == StrDataType('oof')

    def test_correct_indentation_handling(self):
        """The indentation of embedding nodes must correctly strip the
        leading indentation from the cleancopy node, but preserve any
        leading indentation beyond it.

        Trailing indentation must be untouched.
        """
        transformer = Abstractifier()
        # NOTE THAT THERE ARE EXTRA SPACES AFTER THE ``foo``!!!!
        result = dedoc(transformer.convert(parse(doc_prep('''
            >
            __embed__: 'text'
                foo    
                    bar
                        baz'''))))  # noqa: W291

        assert len(result) == 1
        assert isinstance(result[0], EmbeddingBlockNode)
        # Note that the leading indentation has been stripped but trailing
        # preserved!
        assert result[0].content == 'foo    \n    bar\n        baz'


class TestLists:

    def test_paragraph_grouping(self):
        """Lists must maintain "allegiance" to surrounding paragraphs
        as according to our line break rules.
        """
        transformer = Abstractifier()
        result = dedoc(transformer.convert(parse(doc_prep('''
            L1
            ++  foo oof
            ++  bar rab
            L2

            p2

            ++  baz zab'''))))

        assert len(result) == 3
        assert all(
            isinstance(paragraph, Paragraph) for paragraph in result)
        result = cast(list[Paragraph], result)

        assert len(result[0].content) == 3
        assert isinstance(result[0].content[0], RichtextInlineNode)
        assert isinstance(result[0].content[1], List_)
        assert isinstance(result[0].content[2], RichtextInlineNode)

        assert len(result[1].content) == 1
        assert isinstance(result[0].content[0], RichtextInlineNode)

        assert len(result[2].content) == 1
        assert isinstance(result[2].content[0], List_)

    def test_unordered(self):
        """Unordered lists must nest correctly, and multiple line
        entries must be correctly converted into single inline richtext
        nodes.
        """
        transformer = Abstractifier()
        result = dedoc(transformer.convert(parse(doc_prep('''
            ++  foo oof
            ++  bar
                ++  rab
            ++  baz
                is a wonderful fruit
                next to a bread with nice cheeze'''))))

        assert len(result) == 1
        assert isinstance(result[0], Paragraph)
        assert len(result[0].content) == 1

        listnode = result[0].content[0]
        assert isinstance(listnode, List_)
        assert listnode.type_ is ListType.UNORDERED

        assert len(listnode.content) == 3
        assert all(
            isinstance(listitem, ListItem) for listitem in listnode.content)
        assert all(listitem.index is None for listitem in listnode.content)

        foo_content = listnode.content[0].content
        assert len(foo_content) == 1
        assert isinstance(foo_content[0], Paragraph)
        assert len(foo_content[0].content) == 1
        assert isinstance(foo_content[0].content[0], RichtextInlineNode)
        assert foo_content[0].content[0].info is None
        assert foo_content[0].content[0].content == ['foo oof']

        bar_content = listnode.content[1].content
        assert len(bar_content) == 1
        assert isinstance(bar_content[0], Paragraph)
        assert len(bar_content[0].content) == 2
        assert isinstance(bar_content[0].content[0], RichtextInlineNode)
        assert bar_content[0].content[0].info is None
        assert bar_content[0].content[0].content == ['bar']
        assert isinstance(bar_content[0].content[1], List_)

        baz_content = listnode.content[2].content
        assert len(baz_content) == 1
        assert isinstance(baz_content[0], Paragraph)
        assert len(baz_content[0].content) == 1
        assert isinstance(baz_content[0].content[0], RichtextInlineNode)
        assert baz_content[0].content[0].info is None
        assert baz_content[0].content[0].content == [
            'baz is a wonderful fruit next to a bread with nice cheeze']

    def test_ordered(self):
        """Ordered lists must also have their content be in paragraph
        form, and the as-written indices must be preserved in their
        index attributes.
        """
        transformer = Abstractifier()
        result = dedoc(transformer.convert(parse(doc_prep('''
            1.. foo oof
            3.. bar
            5.. baz'''))))

        assert len(result) == 1
        assert isinstance(result[0], Paragraph)
        assert len(result[0].content) == 1

        listnode = result[0].content[0]
        assert isinstance(listnode, List_)
        assert listnode.type_ is ListType.ORDERED

        indices = [listitem.index for listitem in listnode.content]
        assert indices == [1, 3, 5]


class TestInlineRichtext:

    def test_paragraphs(self):
        """Richtext lines must be correctly grouped into paragraphs
        based on the spec's empty-line rules on the root node.
        """
        transformer = Abstractifier()
        result = dedoc(transformer.convert(parse(doc_prep('''
            sometimes when in the course of human events

            it is necessary to split things into multiple paragraphs

            and these should be divided
            as seen fit
            by the author

            of the document
            '''))))

        assert len(result) == 4
        assert isinstance(result[2], Paragraph)
        assert isinstance(result[2].content[0], RichtextInlineNode)
        assert result[2].content[0].content == [
            'and these should be divided as seen fit by the author']

    def test_paragraphs_within_node(self):
        """Richtext lines must be correctly grouped into paragraphs
        based on the spec's empty-line rules within a non-root node.
        """
        transformer = Abstractifier()
        result = dedoc(transformer.convert(parse(doc_prep('''
            > some node
                sometimes when in the course of human events

                it is necessary to split things into multiple paragraphs

                and these should be divided
                as seen fit
                by the author

                of the document'''))))

        assert len(result) == 1
        assert isinstance(result[0], RichtextBlockNode)
        nested_content = result[0].content

        assert len(nested_content) == 4
        assert isinstance(nested_content[2], Paragraph)
        assert isinstance(nested_content[2].content[0], RichtextInlineNode)
        assert nested_content[2].content[0].content == [
            'and these should be divided as seen fit by the author']

    def test_simple_formatting(self):
        """Formatting sugars must be correctly applied to the actual
        text. Their nesting must be per-spec, and not include any
        extraneous info=None nodes.
        """
        transformer = Abstractifier()
        result = dedoc(transformer.convert(parse(doc_prep('''
            sometimes we want to have a little
            **strength**
            in our message, a little
            ^^emphasis^^
            in our voice, a little
            ~~nonsense~~
            in our tone,
            and a little
            __groundedness__ in our perspective

            other times we just want to [[link something and cut the
            pretense, bruv](https://link.example)]'''))))

        assert len(result) == 2
        assert isinstance(result[0], Paragraph)
        assert isinstance(result[0].content[0], RichtextInlineNode)
        rec_strs, _ = result[0].content[0].recursive_strip()
        assert rec_strs == [
            'sometimes we want to have a little ',
            ['strength'],
            ' in our message, a little ',
            ['emphasis'],
            ' in our voice, a little ',
            ['nonsense'],
            ' in our tone, and a little ',
            ['groundedness'],
            ' in our perspective']

    def test_nested_formatting(self):
        """Nested formatting sugars must correctly stack contexts and
        avoid extraneous info=None nodes.
        """
        transformer = Abstractifier()
        result = dedoc(transformer.convert(parse(doc_prep('''
            [[**all ^^the __things__^^**](https://link.example)]'''))))

        assert len(result) == 1
        assert isinstance(result[0], Paragraph)
        assert len(result[0].content) == 1
        assert isinstance(result[0].content[0], RichtextInlineNode)

        rec_strs, rec_infos = result[0].content[0].recursive_strip()
        assert rec_strs == [['all ', ['the ', ['things']]]]
        flat_infos = rec_infos.flatten([0, 1, 1])
        assert flat_infos == [
            InlineNodeInfo(
                target=StrDataType('https://link.example'),
                sugared=BoolDataType(True)),
            InlineNodeInfo(
                formatting=InlineFormatting.STRONG,
                sugared=BoolDataType(True)),
            InlineNodeInfo(
                formatting=InlineFormatting.EMPHASIS,
                sugared=BoolDataType(True)),
            InlineNodeInfo(
                formatting=InlineFormatting.UNDERLINE,
                sugared=BoolDataType(True)),]

    def test_plain_link_no_wrapped_empty(self):
        """A sugared link must not include an empty info=None node
        wrapping the actual content of the link.
        """
        transformer = Abstractifier()
        result = dedoc(transformer.convert(parse(doc_prep('''
            [[foo](https://link.example)]'''))))

        assert len(result) == 1
        assert isinstance(result[0], Paragraph)
        assert len(result[0].content) == 1
        assert isinstance(result[0].content[0], RichtextInlineNode)

        rec_strs, rec_infos = result[0].content[0].recursive_strip()
        assert rec_strs == ['foo']
        flat_infos = rec_infos.flatten([])
        assert flat_infos == [
            InlineNodeInfo(
                target=StrDataType('https://link.example'),
                sugared=BoolDataType(True))]
