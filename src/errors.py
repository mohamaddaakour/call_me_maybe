"""Application-specific exceptions with user-facing messages."""


class ApplicationError(Exception):
    """Base class for expected errors that should not show a traceback."""


class InputFileError(ApplicationError):
    """Raised when an input file cannot be read or validated."""


class OutputFileError(ApplicationError):
    """Raised when the result file cannot be written."""


class GenerationNotImplementedError(ApplicationError):
    """Raised for non-empty workloads until generation is implemented."""
