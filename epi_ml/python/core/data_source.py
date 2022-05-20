"""Module for reading source data files."""
from pathlib import Path

HDF5_RESOLUTION = {
    "1kb":1000,
    "10kb":10000,
    "100kb":100000,
    "1mb":1000000
}

class EpiDataSource(object):
    """Used to contain source files."""
    def __init__(self, hdf5: Path, chromsize: Path, metadata: Path):
        self._hdf5 = hdf5
        self._chromsize = chromsize
        self._metadata = metadata
        self.check_paths()

    @property
    def hdf5_file(self) -> Path:
        """Return hdf5 file path."""
        return self._hdf5

    @property
    def chromsize_file(self) -> Path:
        """Return chromsize file path."""
        return self._chromsize

    @property
    def metadata_file(self) -> Path:
        """Return metadata file path."""
        return self._metadata

    def check_paths(self):
        """Make sure files exist. Raise error otherwise"""
        for path in [self._hdf5, self._chromsize, self._metadata]:
            if not path.is_file():
                raise OSError(f"File does not exist : {path}.\n Expected file at : {path.resolve()}")

    def hdf5_resolution(self):
        """Return resolution as an integer."""
        with open(self.hdf5_file, "r", encoding="utf-8") as my_file:
            first_path = Path(next(my_file).rstrip())
            resolution_string = first_path.name.split('_')[1]
        return HDF5_RESOLUTION[resolution_string]

    def load_chrom_sizes(self):
        """Return sorted list with chromosome (name, size) pairs. This order
        is the same as the order of chroms in the concatenated signals.
        """
        with open(self.chromsize_file, "r", encoding="utf-8") as my_file:
            pairs = [line.rstrip('\n').split() for line in my_file]
        return sorted([(name, int(size)) for name, size in pairs])
