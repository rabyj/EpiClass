import os.path
import argparse

class DirectoryType(object):
    def __init__(self, exist=True):
        self._exist = exist

    def __call__(self, string):
        if self._exist and not os.path.isdir(string):
            msg = "{0} is not a directory".format(string)
            raise argparse.ArgumentTypeError(msg)
        else:
            return string