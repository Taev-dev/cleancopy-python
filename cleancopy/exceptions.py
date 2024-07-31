class CleancopyException(Exception):
    """This, or a subclass of this, is raised for all expected internal
    exceptions that are used for control flow in the library.
    """


class InvalidCleancopy(Exception):
    """This, or a subclass of this, is raise for all expected external
    exceptions -- ie, any problems with the document itself.
    """


class MultipleDocumentMetadatas(InvalidCleancopy):
    """Raised when we encounter multiple nodes marked as document
    metadata.
    """
