import io

class EpiDataSource(object):
    """used to load metadata"""
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

