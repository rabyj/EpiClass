import argparse
from pathlib import Path

class DirectoryType(object):
    def __init__(self, exist=True):
        self._exist = exist

    def __call__(self, string):
        string = Path(string)
        if self._exist and not string.is_dir():
            msg = f"Not a directory : {string}"
            raise argparse.ArgumentTypeError(msg)
        else:
            return string