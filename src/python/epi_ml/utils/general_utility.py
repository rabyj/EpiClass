"""For lost little functions that don't fit anywhere else."""
from pathlib import Path
from typing import List


def write_hdf5_paths_to_file(
    md5s: List[str], prefix: str, suffix: str, path: Path
) -> None:
    """Write a list of md5s to a file, with a prefix and suffix."""
    with open(path, "w", encoding="utf8") as f:
        for md5 in md5s:
            f.write(f"{prefix}{md5}{suffix}\n")


def write_md5s_to_file(md5s: List[str], logdir: str, name: str) -> Path:
    """Write a list of md5s to a file. Return filepath."""
    filename = Path(logdir) / f"{name}.md5"
    with open(filename, "w", encoding="utf8") as f:
        for md5 in md5s:
            f.write(f"{md5}\n")
    return filename
