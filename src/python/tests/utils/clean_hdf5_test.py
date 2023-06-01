"""Tests for the `clean_hdf5` module."""
from pathlib import Path

import pytest

from epi_ml.utils.clean_hdf5 import (
    is_position_in_blacklist,
    load_bed,
    main,
    preprocess_bed,
    process_file,
)

CURRENT_DIR = Path(__file__).parent.resolve()


def test_is_position_in_blacklist():
    """Tests the `is_position_in_blacklist` function."""
    # Sample blacklist bed intervals (0-based start, 1-based end coordinates)
    blacklist_bed = [
        ("chr1", 10, 20),
        ("chr2", 5, 15),
        ("chr2", 25, 30),
        ("chr3", 10, 15),
    ]

    # Preprocess the blacklist bed
    chromosome_intervals = preprocess_bed(blacklist_bed)

    # Test cases with region overlap
    test_cases = [
        ("chr1", 5, 12, True),  # Region overlaps with chr1 interval
        ("chr1", 9, 20, True),  # Region overlaps with chr1 interval
        ("chr1", 15, 18, True),  # Region overlaps with chr1 interval
        ("chr1", 20, 22, False),  # Region does not overlap with chr1 interval
        ("chr2", 5, 8, True),  # Region overlaps with chr2 interval
        ("chr2", 15, 18, False),  # Region does not overlap with chr2 interval
        ("chr2", 14, 18, True),  # Region overlap with chr2 interval
        ("chr2", 30, 35, False),  # Region does not overlap with chr2 interval
        ("chr3", 10, 15, True),  # Region overlaps with chr3 interval
        ("chr3", 12, 14, True),  # Region overlaps with chr3 interval
        ("chr3", 5, 8, False),  # Region does not overlap with chr3 interval
        ("chr3", 5, 8, False),  # Region does not overlap with chr3 interval
        ("chr4", 10, 20, False),  # No intervals for chr4
    ]

    for chromosome, start, end, expected in test_cases:
        print(f"Testing {chromosome}:{start}-{end}...")
        result = is_position_in_blacklist(start, end, chromosome_intervals, chromosome)
        assert result == expected, print(f"Expected {expected}, got {result}")


@pytest.fixture(name="test_hdf5")
def test_hdf5_file_path() -> Path:
    """Returns the path to a test HDF5 file."""
    return Path(
        CURRENT_DIR
        / "../fixtures/89a0dcb635f0e9740f587931437b69f1_100kb_all_none_value.hdf5"
    )


@pytest.fixture(name="test_bed")
def test_bed_file_path() -> Path:
    """Returns the path to a test bed file."""
    return Path(CURRENT_DIR / "../fixtures/hg38_unified_blacklist.bed")


def test_load_bed(test_bed):
    """Tests the `load_bed` function."""
    bed_data = load_bed(test_bed)
    # Add your assertions here based on what you expect in your bed file
    assert bed_data is not None
    assert len(bed_data) > 0


def test_preprocess_bed(test_bed):
    """Tests the `preprocess_bed` function."""
    bed_data = load_bed(test_bed)
    processed_data = preprocess_bed(bed_data)
    # Add your assertions here based on what you expect after processing your bed file
    assert processed_data is not None
    assert len(processed_data) > 0


def test_process_file(tmpdir, test_bed, test_hdf5):
    """Tests the `process_file` function."""
    bed_data = load_bed(test_bed)
    processed_data = preprocess_bed(bed_data)

    output_dir = Path(tmpdir.mkdir("sub"))
    process_file(test_hdf5, processed_data, output_dir)

    output_file_path = output_dir / (test_hdf5.stem + "_0blklst.hdf5")
    # Make assertions about the output file
    assert output_file_path.is_file()


def test_main(tmpdir, test_bed, test_hdf5, mocker):
    """Tests the `main` function."""
    tmpdir = Path(tmpdir)
    hdf5_list_file = tmpdir / "hdf5_list.txt"
    hdf5_list_file.write_text(str(test_hdf5))

    # Create a simple function to replace argument parsing
    def mock_parse_arguments():
        class Args:
            """Mock arguments."""

            hdf5_list = hdf5_list_file
            bed_filter = test_bed
            output_dir = tmpdir
            n_jobs = 1

        return Args()

    # Replace the original parse_arguments function with our mock
    mocker.patch(
        "epi_ml.utils.clean_hdf5.parse_arguments", side_effect=mock_parse_arguments
    )

    # Run the main function
    main()

    # Make assertions about the output directory and files
    output_file_path = tmpdir / (test_hdf5.stem + "_0blklst.hdf5")
    assert output_file_path.is_file()
