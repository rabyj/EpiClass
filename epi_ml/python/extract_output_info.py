"""
Extract specific information from what epiML prints.

Useful to get several results for the Excel sheet at once.
"""
import argparse
import io
import sys

def parse_arguments(args: list) -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('epiML_output', type=argparse.FileType('r'), help='File to extract info from.')
    return arg_parser.parse_args(args)


class EpiMLOutputReader():
    """Read epiML output files and extract their information."""

    SIZE_TOKEN = "validation"
    EXAMPLES_TOKEN = "Examples"
    TRAINING_TOKEN = "epoch"
    TRAINING_METRICS_TOKEN = "Training"
    VALIDATION_METRICS_TOKEN = "Validation"
    HYPERPARAMS_TOKENS = frozenset([
        "Nb", "Layers", "batch_size:", "early_stop_limit:", "is_training:", "keep_prob:",
        "l1_scale:", "l2_scale:", "learning_rate:", "measure_frequency:", "training_epochs:"
    ])

    def __init__(self):
        self._file = None
        self._info = {}
        self._tokens = self._init_tokens()
        self._switcher = self._init_switcher()
        self._current_line = ""

    def _init_tokens(self):
        """Return set of all tokens"""
        return set([
            self.SIZE_TOKEN, self.EXAMPLES_TOKEN, self.TRAINING_TOKEN,
            self.TRAINING_METRICS_TOKEN, self.VALIDATION_METRICS_TOKEN
        ]).union(self.HYPERPARAMS_TOKENS)

    def _init_switcher(self):
        """Return dict defining methods associated with tokens."""
        return {
            self.SIZE_TOKEN : self._read_set_sizes,
            self.EXAMPLES_TOKEN : self._read_examples,
            self.TRAINING_TOKEN : self._read_training,
            self.TRAINING_METRICS_TOKEN : self._read_training_metrics,
            self.VALIDATION_METRICS_TOKEN : self._read_validation_metrics
        }

    def read_file(self, file: io.IOBase):
        """Read file and extract important information."""
        self._file = file
        self._info = {} #empty if another file was read before

        file.seek(0)
        while True:
            try:
                self._current_line = next(file)
            except StopIteration:
                break

            first_word = self._current_line.strip('\n').split(' ', 1)[0]
            if first_word in self._tokens:
                self._read_section(first_word)

    def _read_section(self, token):
        """Choose section reading method."""
        if token in self.HYPERPARAMS_TOKENS:
            self._read_hyperparams()
        else:
            try:
                reader = self._switcher[token]
                reader()
            except KeyError:
                raise ValueError("Invalid token")

    def _read_hyperparams(self):
        print("Reading hypers")
        pass

    def _read_set_sizes(self):
        print("Reading set sizes")
        pass

    def _read_examples(self):
        print("Reading examples")
        pass

    def _read_training(self):
        print("Reading training")
        pass

    def _read_training_metrics(self):
        print("Reading training metrics")
        pass

    def _read_validation_metrics(self):
        print("Reading validation metrics")
        pass


def main(args):
    """Miaw"""

    options = parse_arguments(args)

    reader = EpiMLOutputReader()
    reader.read_file(options.epiML_output)


if __name__ == "__main__":
    main(sys.argv[1:])
