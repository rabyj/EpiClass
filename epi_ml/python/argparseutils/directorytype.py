from pathlib import Path


class DirectoryTypeError(Exception):
    """An error from trying to convert a command line string to a directory."""
    def __init__(self, message, path, *args: object) -> None:
        super().__init__(message, *args)
        self.path = path


class DirectoryType(object):
    """Type to check directory status as soon as parsed"""
    def __init__(self, exist=True):
        self._exist = exist

    def __call__(self, string):
        string = Path(string)
        if self._exist and not string.is_dir():
            msg = f"Not a directory : {string}"
            raise DirectoryTypeError(message=msg, path=string)
        else:
            return string
