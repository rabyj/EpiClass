"""Module for reading source data files."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Tuple

import h5py

HDF5_RESOLUTION = {"1kb": 1000, "10kb": 10000, "100kb": 100000, "1mb": 1000000}


class EpiDataSource:
    """Used to contain source files."""

    def __init__(self, hdf5: Path, chromsize: Path, metadata: Path):
        self._hdf5 = hdf5
        self._chromsize = chromsize
        self._metadata = metadata
        self.check_paths()

    @property
    def hdf5_file(self) -> Path:
        """Return hdf5 list file path."""
        return self._hdf5

    @property
    def chromsize_file(self) -> Path:
        """Return chromsize file path."""
        return self._chromsize

    @property
    def metadata_file(self) -> Path:
        """Return metadata file path."""
        return self._metadata

    def check_paths(self) -> None:
        """Make sure files exist. Raise error otherwise"""
        for path in [self._hdf5, self._chromsize, self._metadata]:
            if not path.is_file():
                raise OSError(
                    f"File does not exist : {path}.\n Expected file at : {path.resolve()}"
                )

    @staticmethod
    def get_file_list(hdf5_list_path: Path) -> List[Path]:
        """Return list of hdf5 files."""
        with open(hdf5_list_path, "r", encoding="utf-8") as my_file:
            return [Path(line.rstrip("\n")) for line in my_file]

    def hdf5_resolution(self) -> int:
        """Return resolution as an integer."""
        with open(self.hdf5_file, "r", encoding="utf-8") as my_file:
            first_path = Path(next(my_file).rstrip())
            try:
                resolution = self.get_file_hdf5_resolution(first_path)
            except (KeyError, FileNotFoundError) as err:
                warnings.warn(f"{err}. Seeking resolution from filename.")
                try:
                    resolution = self.get_resolution_from_filename(first_path)
                except KeyError as err2:
                    raise KeyError(
                        f"Filename does not contain resolution: {first_path}"
                    ) from err2
        return resolution

    @staticmethod
    def get_resolution_from_filename(path: Path) -> int:
        """Return resolution as an integer."""
        resolution_string = path.name.split("_")[1]
        return HDF5_RESOLUTION[resolution_string]

    @staticmethod
    def get_file_hdf5_resolution(hdf5_file: Path) -> int:
        """Return resolution as an integer."""
        with h5py.File(hdf5_file, "r") as h5_file:
            try:
                resolution = int(h5_file.attrs["bin"][0])  # type: ignore
            except KeyError as err:
                raise KeyError(
                    f"Resolution not found in {hdf5_file}. (attribute 'bin' does not exist)"
                ) from err
        return resolution

    @staticmethod
    def load_external_chrom_file(chrom_file: Path | str) -> List[Tuple[str, int]]:
        """Return sorted list with chromosome (name, size) pairs."""
        with open(chrom_file, "r", encoding="utf-8") as my_file:
            pairs = [line.rstrip("\n").split() for line in my_file]
        return sorted([(name, int(size)) for name, size in pairs])

    def load_chrom_sizes(self) -> List[Tuple[str, int]]:
        """Return sorted list with chromosome (name, size) pairs. This order
        is the same as the order of chroms in the concatenated signals.
        """
        return self.load_external_chrom_file(self.chromsize_file)
