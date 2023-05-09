from lark.exceptions import LarkError


class IndentationError(LarkError):
    """Raised when there is a problem with the indentation."""


class InvalidMetadataValue(LarkError):
    """Raised when the metadata value is invalid."""
