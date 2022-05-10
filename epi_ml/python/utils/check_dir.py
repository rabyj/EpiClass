"""Utility script to check a directory and create it if needed.

Intended to be used before launching any main training/prediction scripts
to verify that the logging directory is present, and create it if not.
This permits proper redirection of stdout and stderr to files in bash to said directory,
in the following scripts.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, "/home/rabyj/project-rabyj/sources/epi_ml/epi_ml/python/argparseutils")

from directorytype import DirectoryType, DirectoryTypeError


def parse_arguments(args: list) -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dir', type=DirectoryType(), help='A directory.')
    arg_parser.add_argument('--exists', action="store_true", help='Specifies that the directory should already exist.')
    return arg_parser.parse_args(args)

def main(args):
    """main called from command line, edit to change behavior"""

    try:
        cli = parse_arguments(args)
    except DirectoryTypeError as dir_err:
        faulty_path = dir_err.path
        if cli.exists:
            raise dir_err from None # make
        else:
            for parent in reversed(faulty_path.parents):
                parent.mkdir(mode=0o2750, exist_ok=True)
            faulty_path.mkdir(mode=0o2750, exist_ok=True)

            print(f"Created missing logdir and needed parents : {faulty_path}")

            # Reverify everything was done properly
            cli = parse_arguments(args)


if __name__ == "__main__":
    main(sys.argv[1:])
