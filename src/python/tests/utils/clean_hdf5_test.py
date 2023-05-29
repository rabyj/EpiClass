"""Tests for the `clean_hdf5` module."""
from epi_ml.utils.clean_hdf5 import is_position_in_blacklist, preprocess_bed


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
