from dataclasses import dataclass
from typing import Any


@dataclass
class DocumentNode:
    content_lines: Any
    metadata_lines: Any


@dataclass
class VersionComment:
    version: str


@dataclass
class EmptyLine:
    token: Any


@dataclass
class ContentLine:
    token: Any


@dataclass
class CleancopyDocument:
    version_comment: VersionComment
    document_root: DocumentNode
