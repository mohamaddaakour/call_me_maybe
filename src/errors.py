class InputFileError(Exception):
    """Report a clear, recoverable problem with an input file."""


class DecodingError(Exception):
    """Report that the model could not produce a constrained call."""
