import io
import os.path

HDF5_RESOLUTION = {
    "1kb":1000,
    "10kb":10000,
    "100kb":100000,
    "1mb":1000000
}

class EpiDataSource(object):
    """Used to contain source files."""
    def __init__(self, hdf5: io.IOBase, chromsize: io.IOBase, metadata: io.IOBase):
        self._hdf5 = hdf5
        self._chromsize = chromsize
        self._metadata = metadata

    @property
    def hdf5_file(self) -> io.IOBase:
        return self._hdf5

    @property
    def chromsize_file(self) -> io.IOBase:
        return self._chromsize

    @property
    def metadata_file(self) -> io.IOBase:
        return self._metadata

    def hdf5_resolution(self):
        """Return resolution as an integer."""
        self.hdf5_file.seek(0)
        first_path = next(self.hdf5_file).rstrip('\n')
        resolution_string = os.path.basename(first_path).split('_')[1]
        self.hdf5_file.seek(0)
        return HDF5_RESOLUTION[resolution_string]

    def load_chrom_sizes(self):
        """Return sorted list with chromosome (name, size) pairs. This order
        is the same as the order of chroms in the concatenated signals.
        """
        self.chromsize_file.seek(0)
        pairs = [line.rstrip('\n').split() for line in self.chromsize_file]
        return sorted([(name, int(size)) for name, size in pairs])
