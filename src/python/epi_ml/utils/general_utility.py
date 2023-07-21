"""For lost little functions that don't fit anywhere else."""
from pathlib import Path
from typing import Iterable


def write_hdf5_paths_to_file(
    md5s: Iterable[str], parent: str, suffix: str, filepath: Path
) -> None:
    """Write a list of md5s to a file, with a prefix and suffix.
    Path(prefix)/md5{suffix}.hdf5
    """
    with open(filepath, "w", encoding="utf8") as f:
        for md5 in md5s:
            line = Path(parent) / (md5 + f"{suffix}.hdf5\n")
            f.write(str(line))


def write_md5s_to_file(md5s: Iterable[str], logdir: str, name: str) -> Path:
    """Write a list of md5s to a file. Return filepath."""
    filename = Path(logdir) / f"{name}.md5"
    with open(filename, "w", encoding="utf8") as f:
        for md5 in md5s:
            f.write(f"{md5}\n")
    return filename
