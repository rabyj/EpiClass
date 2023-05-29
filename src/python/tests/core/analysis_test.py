"""Test analysis code"""
import pytest

from epi_ml.core.analysis import bins_to_bed_ranges


def test_bins_to_bed_ranges() -> None:
    """Tests the function bins_to_bed_ranges().

    This function tests bins_to_bed_ranges() with various input arguments, including single bin indexes,
    multiple bin indexes, and out of range bin indexes. The test checks if the function correctly converts
    bin indexes to chromosome ranges and raises an IndexError when a bin index is out of range.
    """
    chroms = [("chr1", 100), ("chr2", 150), ("chr3", 300), ("chrX", 120)]
    resolution = 100

    # Test single bin indexes
    assert bins_to_bed_ranges([0], chroms, resolution) == [("chr1", 0, 100)]
    assert bins_to_bed_ranges([1], chroms, resolution) == [("chr2", 0, 100)]
    assert bins_to_bed_ranges([2], chroms, resolution) == [("chr2", 100, 150)]
    assert bins_to_bed_ranges([3], chroms, resolution) == [("chr3", 0, 100)]
    assert bins_to_bed_ranges([4], chroms, resolution) == [("chr3", 100, 200)]
    assert bins_to_bed_ranges([5], chroms, resolution) == [("chr3", 200, 300)]
    assert bins_to_bed_ranges([6], chroms, resolution) == [("chrX", 0, 100)]
    assert bins_to_bed_ranges([7], chroms, resolution) == [("chrX", 100, 120)]

    # Test multiple bin indexes
    assert bins_to_bed_ranges([0, 2, 4], chroms, resolution) == [
        ("chr1", 0, 100),
        ("chr2", 100, 150),
        ("chr3", 100, 200),
    ]
    assert bins_to_bed_ranges([5, 1, 3], chroms, resolution) == [
        ("chr2", 0, 100),
        ("chr3", 0, 100),
        ("chr3", 200, 300),
    ]

    # Test out of range bin indexes
    with pytest.raises(IndexError):
        bins_to_bed_ranges([8], chroms, resolution)
    with pytest.raises(IndexError):
        bins_to_bed_ranges([-1], chroms, resolution)
    with pytest.raises(IndexError):
        bins_to_bed_ranges([1, 2, 8], chroms, resolution)
