"""New type for argparser, and an associated error."""
from pathlib import Path


class DirectoryCheckerError(Exception):
    """An error from trying to convert a command line string to a directory."""

    def __init__(self, message, path, *args: object) -> None:
        super().__init__(message, *args)
        self.path = path


class DirectoryChecker(object):
    """Type to check directory status as soon as parsed by command line"""

    def __init__(self, exists=True) -> None:
        self.exists = exists

    def __call__(self, string):
        string = Path(string)
        if self.exists and not string.is_dir():
            msg = f"Not a directory : {string}"
            raise DirectoryCheckerError(message=msg, path=string)
        else:
            return string
