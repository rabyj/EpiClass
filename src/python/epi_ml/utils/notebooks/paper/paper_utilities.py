"""Utility functions for the paper notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from epi_ml.core.metadata import Metadata

ASSAY: str = "assay_epiclass"
CELL_TYPE: str = "harmonized_sample_ontology_intermediate"
ASSAY_MERGE_DICT: Dict[str, str] = {
    "mrna_seq": "rna_seq",
    "wgbs-pbat": "wgbs",
    "wgbs-standard": "wgbs",
}


class IHECColorMap:
    """Class to handle IHEC color map."""

    def __init__(self, base_fig_dir: Path):
        self.base_fig_dir = base_fig_dir
        self.ihec_colormap_name = "IHEC_EpiATLAS_IA_colors_Mar18_2024.json"
        self.ihec_color_map = self.get_IHEC_color_map(
            base_fig_dir, self.ihec_colormap_name
        )
        self.assay_color_map = self.create_assay_color_map(self.ihec_color_map)
        self.cell_type_color_map = self.create_cell_type_color_map(self.ihec_color_map)

    @staticmethod
    def get_IHEC_color_map(folder: Path, name: str) -> List[Dict]:
        """Get the IHEC color map."""
        color_map_path = folder / name
        with open(color_map_path, "r", encoding="utf8") as color_map_file:
            ihec_color_map = json.load(color_map_file)
        return ihec_color_map

    @staticmethod
    def create_assay_color_map(ihec_color_map: List[Dict]) -> Dict[str, str]:
        """Create a rbg color map for ihec core assays."""
        colors = dict(ihec_color_map[0]["experiment"][0].items())
        for name, color in list(colors.items()):
            rbg = color.split(",")
            colors[name.lower()] = f"rgb({rbg[0]},{rbg[1]},{rbg[2]})"

        colors["rna_seq"] = colors["rna-seq"]
        return colors

    @staticmethod
    def create_cell_type_color_map(ihec_color_map: List[Dict]) -> Dict[str, str]:
        """Read the rbg color map for ihec cell types."""
        colors = dict(
            ihec_color_map[3]["harmonized_sample_ontology_intermediate"][0].items()
        )
        for name, color in list(colors.items()):
            rbg = color.split(",")
            colors[name] = f"rgb({rbg[0]},{rbg[1]},{rbg[2]})"
        return colors


def merge_similar_assays(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to merge rna-seq/wgbs categories, included prediction score."""
    df = df.copy(deep=True)
    try:
        df["rna_seq"] = df["rna_seq"] + df["mrna_seq"]
        df["wgbs"] = df["wgbs-standard"] + df["wgbs-pbat"]
    except KeyError as exc:
        raise ValueError(
            "Wrong results dataframe, label category is not assay specific."
        ) from exc
    df.drop(columns=["mrna_seq", "wgbs-standard", "wgbs-pbat"], inplace=True)
    df["True class"].replace(ASSAY_MERGE_DICT, inplace=True)
    df["Predicted class"].replace(ASSAY_MERGE_DICT, inplace=True)

    try:
        df[ASSAY].replace(ASSAY_MERGE_DICT, inplace=True)
    except KeyError:
        pass

    # Recompute Max pred if it exists
    classes = df["True class"].unique()
    if "Max pred" in df.columns:
        df["Max pred"] = df[classes].max(axis=1)
    return df


def return_metadata(version: str, paper_dir: Path | str) -> Metadata:
    """Return metadata for a specific version.

    Example of epiRR unique to v1: IHECRE00003355.2
    """
    paper_dir = Path(paper_dir)
    if version not in ["v1", "v2", "v2-encode"]:
        raise ValueError("Version must be one of v1, v2, v2-encode")

    names = {
        "v1": "hg38_2023-epiatlas_dfreeze_formatted_JR.json",
        "v2": "hg38_2023-epiatlas-dfreeze-pospurge-nodup_filterCtl.json",
        "v2-encode": "hg38_2023-epiatlas-dfreeze_v2.1_w_encode_noncore_2.json",
    }
    metadata = Metadata(paper_dir / "data" / "metadata" / names[version])
    return metadata
