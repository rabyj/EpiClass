"""Slightly modify argparse to print help on any error."""
import argparse
import sys


class DefaultHelpParser(argparse.ArgumentParser):
    """Modified ArgumentParser."""

    def error(self, message):
        sys.stderr.write(f"error: {message}\n")
        self.print_help()
        sys.exit(2)
