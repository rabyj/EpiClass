"""EpiAtlas data treatment testing module."""
from __future__ import annotations

from pathlib import Path
from typing import List

import h5py

from src.python.core.data_source import EpiDataSource
from src.python.core.epiatlas_treatment import EpiAtlasTreatment
from src.python.core.hdf5_loader import Hdf5Loader
from src.python.core.metadata import Metadata


class EpiatlasTreatmentTestData(object):
    """Create and handle mock/test EpiAtlasTreatment."""

    def __init__(self, metadata_path: Path, md5_list_path: Path, label_category: str):
        self.dir = Path(__file__).parent.resolve()
        self.chroms_file = (
            self.dir.parents[2] / "input-format/hg38.noy.chrom.sizes"
        )  # src/input_format/
        self.chroms = Hdf5Loader.load_chroms(self.chroms_file)

        tmp_hdf5 = self.create_temp_hdf5s(md5_list_path.resolve())

        self.datasource = self.create_mock_datasource(
            metadata_path, tmp_hdf5, self.chroms_file
        )

        self.ea_handler = EpiAtlasTreatment(
            datasource=self.datasource,
            label_category=label_category,
            min_class_size=3,
            n_fold=2,
        )

    def create_temp_hdf5s(
        self, md5_list_path: Path, name="_100kb_all_none_value.hdf5"
    ) -> List[Path]:
        """Create temporary files and returns paths"""
        tmp_files = []
        with open(md5_list_path, "r", encoding="utf8") as md5_list:
            for md5 in md5_list.readlines():
                md5 = md5.strip()
                tmp_file = Path("/tmp") / (md5 + name)
                tmp_files.append(tmp_file)
                self.write_mock_hdf5(tmp_file, md5)

        return tmp_files

    def write_mock_hdf5(self, path: Path, md5: str):
        """Write a hdf5 file to the given path with the expected general structure."""
        with h5py.File(name=path, mode="w") as f:
            grp = f.create_group(md5)
            for chrom in self.chroms:
                grp.create_dataset(name=chrom, data=[1, 2], dtype=int)
        f.close()

    @staticmethod
    def create_temp_file_list(temp_files: List[Path]) -> Path:
        """Create a file containing a list of given paths.

        Returns path of created file.
        """
        tmp_file = Path("/tmp/hdf5s.list")
        with open(tmp_file, "w", encoding="utf-8") as f:
            for path in temp_files:
                f.write(f"{path}\n")

        return tmp_file

    def create_mock_datasource(
        self, metadata: Path, tmp_hdf5s: List[Path], chroms_file: Path
    ) -> EpiDataSource:
        """Return a datasource object for testing purposes."""
        return EpiDataSource(
            hdf5=EpiatlasTreatmentTestData.create_temp_file_list(tmp_hdf5s),
            chromsize=chroms_file,
            metadata=metadata,
        )


# standalone
def create_test_metadata(metadata_source_path: Path, md5_list: Path):
    """Create a metadata json file with a subset of information, for testing purposes."""
    my_metadata = Metadata(metadata_source_path)

    with open(md5_list, "r", encoding="utf8") as f:
        md5_set = set([md5.strip() for md5 in f.readlines()])

    for md5 in list(my_metadata.md5s):
        if md5 not in md5_set:
            del my_metadata[md5]

    my_metadata.save(md5_list.parent / (md5_list.stem + "-metadata.json"))


def main():
    """Create test data."""
    test_set = "test-epilap-empty-biotype-n40"
    current_dir = Path(__file__).parent.resolve()
    md5_list = current_dir / f"{test_set}.md5"
    metadata_path = current_dir / f"{test_set}-metadata.json"

    label_category = "biomaterial_type"

    test_data = EpiatlasTreatmentTestData(metadata_path, md5_list, label_category)


if __name__ == "__main__":
    main()
