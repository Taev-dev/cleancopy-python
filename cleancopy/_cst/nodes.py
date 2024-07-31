"""After getting the result of a parse from treesitter, we convert
everything into using instances of these objects instead. This gives us
stronger typing, a more consistent API, and a convenient place to put
any needed helper methods, in a way that doesn't get exposed as part
of the package API.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from decimal import Decimal
from typing import Literal

from cleancopy.spectypes import InlineFormatting
from cleancopy.spectypes import ListType

# Note: URIs are automatically converted to strings; they're only separate in
# the CST because sugared strings are a strict subset of strings and need to
# be differentiated from the other target types in the grammar itself
type CSTMetadataTarget = (
    CSTMetadataStr | CSTMetadataMention | CSTMetadataTag | CSTMetadataVariable
    | CSTMetadataReference)
type CSTMetadataValue = (
    CSTMetadataStr | CSTMetadataInt | CSTMetadataDecimal | CSTMetadataBool
    | CSTMetadataNull | CSTMetadataMention | CSTMetadataTag
    | CSTMetadataVariable | CSTMetadataReference | CSTMetadataMissing)


@dataclass(kw_only=True)
class CSTNode:
    """Base class for all CST node types.
    TODO: put line and column infos here!
    """


@dataclass(kw_only=True)
class CSTDocument(CSTNode):
    root: CSTDocumentNode[CSTDocumentNodeContentRichtext]


@dataclass(kw_only=True)
class CSTDocumentNode[
        C: CSTDocumentNodeContentRichtext | CSTDocumentNodeContentEmbedding
        ](CSTNode):
    """The base class for both richtext and embedded nodes."""
    title: list[CSTRichtext | CSTEmptyLine]
    metadata: list[CSTMetadataAssignment | CSTAnnotation | CSTEmptyLine]
    content: C | None


@dataclass(kw_only=True)
class CSTDocumentNodeTitle(CSTNode):
    """This class is used purely for disambiguation when dealing with
    the output of treesitter. Its content gets elided into the parent
    CSTDocumentNode title attribute.
    """
    content: list[CSTRichtext | CSTEmptyLine]

    def add_line(
            self,
            line: CSTRichtext | CSTEmptyLine
            ) -> None:
        """This is a convenience method that handles merging multiple
        consecutive CSTRichtext instances into single values with
        line breaks between them.
        """
        if isinstance(line, CSTRichtext) and self.content:
            previous_line = self.content[-1]
            if isinstance(previous_line, CSTRichtext):
                previous_line.content.append(CSTLineBreak(content=None))
                previous_line.content.extend(line.content)
            else:
                self.content.append(line)

        else:
            self.content.append(line)


@dataclass(kw_only=True)
class CSTDocumentNodeMetadata(CSTNode):
    """This class is used purely for disambiguation when dealing with
    the output of treesitter. Its content gets elided into the parent
    CSTDocumentNode metadata attribute.
    """
    content: list[CSTMetadataAssignment | CSTAnnotation | CSTEmptyLine]

    def add_line(
            self,
            line: CSTMetadataAssignment | CSTAnnotation | CSTEmptyLine
            ) -> None:
        """This is a convenience method that handles merging multiple
        consecutive CSTAnnotation instances into single values with
        line breaks between them.
        """
        if isinstance(line, CSTAnnotation) and self.content:
            previous_line = self.content[-1]
            if isinstance(previous_line, CSTAnnotation):
                previous_line.content.append(CSTLineBreak(content=None))
                previous_line.content.extend(line.content)
            else:
                self.content.append(line)

        else:
            self.content.append(line)


@dataclass(kw_only=True)
class CSTDocumentNodeContentRichtext(CSTNode):
    content: list[
        CSTEmptyLine | CSTList | CSTAnnotation | CSTRichtext | CSTDocumentNode]


@dataclass(kw_only=True)
class CSTDocumentNodeContentEmbedding(CSTNode):
    content: str


@dataclass(kw_only=True)
class CSTEmptyLine(CSTNode):
    content: str


@dataclass(kw_only=True)
class CSTLineBreak(CSTNode):
    content: None


@dataclass(kw_only=True)
class CSTList(CSTNode):
    type_: ListType
    content: list[CSTListItem]


@dataclass(kw_only=True)
class CSTListItemOLIndexContainer(CSTNode):
    """This class is used purely for disambiguation when dealing with
    the output of treesitter. Its content gets elided into the parent
    CSTListItem index attribute.
    """
    value: int


@dataclass(kw_only=True)
class CSTListItem(CSTNode):
    index: int | None
    content: list[CSTEmptyLine | CSTList | CSTAnnotation | CSTRichtext]


@dataclass(kw_only=True)
class CSTRichtext(CSTNode):
    content: list[
        str | CSTFormattingMarker | CSTFmtBracketLink | CSTFmtBracketMetadata
        | CSTLineBreak]


@dataclass(kw_only=True)
class CSTFormattingMarker(CSTNode):
    marker: InlineFormatting
    was_autoclose: bool = field(default=False, kw_only=True)


@dataclass(kw_only=True)
class CSTAnnotation(CSTNode):
    content: list[str | CSTLineBreak]


@dataclass(kw_only=True)
class CSTFmtBracketMetadataContainer(CSTNode):
    """This class is used purely for disambiguation when dealing with
    the output of treesitter. Its content gets elided into the parent
    CSTInlineMetadata
    """
    content: list[CSTMetadataAssignment]


@dataclass(kw_only=True)
class CSTFmtBracketMetadata(CSTNode):
    content: CSTRichtext
    metadata: list[CSTMetadataAssignment]


@dataclass(kw_only=True)
class CSTFmtBracketLink(CSTNode):
    content: None | CSTRichtext
    target: CSTMetadataTarget


@dataclass
class CSTMetadataAssignment(CSTNode):
    key: CSTMetadataStr
    value: CSTMetadataValue


@dataclass
class CSTMetadataStr(CSTNode):
    value: str
    # Note: gets set to None for sugared strings
    quote_style: None | Literal[1] | Literal[2] = field(kw_only=True)


@dataclass
class CSTMetadataInt(CSTNode):
    value: int


@dataclass
class CSTMetadataDecimal(CSTNode):
    value: Decimal


@dataclass
class CSTMetadataBool(CSTNode):
    value: bool


@dataclass
class CSTMetadataNull(CSTNode):
    value: None


@dataclass
class CSTMetadataMissing(CSTNode):
    value: None


@dataclass
class CSTMetadataMention(CSTNode):
    value: CSTMetadataStr


@dataclass
class CSTMetadataTag(CSTNode):
    value: CSTMetadataStr


@dataclass
class CSTMetadataVariable(CSTNode):
    value: CSTMetadataStr


@dataclass
class CSTMetadataReference(CSTNode):
    value: CSTMetadataStr
