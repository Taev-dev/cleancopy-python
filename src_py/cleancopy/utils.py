from cleancopy.ast import BlockNode
from cleancopy.ast import Document
from cleancopy.ast import Paragraph


def dedoc(document: Document) -> list[Paragraph | BlockNode]:
    """Given a simple document with no title and no metadata, extracts
    the content from the root node and returns it.

    Useful if you want to embed a document in another document, or for
    testing.
    """
    if document.title is not None:
        raise ValueError('Dedoc requires a document without a title!')
    if document.info is not None:
        raise ValueError('Dedoc requires a document without info!')

    return document.root.content
