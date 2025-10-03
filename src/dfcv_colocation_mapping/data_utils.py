import os
import yaml
import logging

import numpy as np
import pandas as pd
import geopandas as gpd

import subprocess
import humanize

import textwrap
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon

logging.basicConfig(level=logging.INFO)


def get_conflict_source(conflict_exposure_source):
    if conflict_exposure_source == "ACLED (population_best)":
        return "acled"
    elif conflict_exposure_source == "ACLED (WBG calculation)":
        return "wbg_acled"
    elif conflict_exposure_source == "UCDP":
        return "ucdp"


def get_exposure(exposure):
    if exposure == "absolute":
        return "exposure"
    elif exposure == "relative":
        return "exposure_relative"
    elif exposure == "intensity_weighted_relative":
        return "intensity_weighted_exposure_relative"


def match_shape(src1: np.ndarray, src2: np.ndarray) -> np.ndarray:
    """Align `src2` to the shape of `src1` by cropping or zero-padding."""
    src1_rows, src1_cols = src1.shape
    src2_rows, src2_cols = src2.shape

    # Crop to src1 shape
    rows = min(src1_rows, src2_rows)
    cols = min(src1_cols, src2_cols)
    src2 = src2[:rows, :cols]

    # Pad if needed
    if src2.shape != (src1_rows, src1_cols):
        pad_rows = src1_rows - src2.shape[0]
        pad_cols = src1_cols - src2.shape[1]
        src2 = np.pad(
            src2,
            ((0, pad_rows), (0, pad_cols)),
            mode="constant",
            constant_values=0,
        )

    return src1, src2


def _minmax_scale(data: pd.Series) -> pd.Series:
    """
    Performs Min-Max scaling on a NumPy array or Pandas Series, scaling values to [0, 1].

    Args:
        data (np.ndarray or pd.Series): The input data to be scaled.

    Returns:
        np.ndarray or pd.Series: The scaled data with values between 0 and 1.

    Raises:
        TypeError: If `data` is not a NumPy array or Pandas Series.
        ValueError: If `data` is empty.
    """

    # Ensure input is a valid type
    if not isinstance(data, (np.ndarray, pd.Series)):
        raise TypeError(
            f"Input must be np.ndarray or pd.Series, got {type(data).__name__}"
        )

    # Ensure data is not empty
    if len(data) == 0:
        raise ValueError("Input data is empty. Cannot perform scaling.")

    # Compute min and max, ignoring NaNs
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)

    # Handle case where all values are identical
    if max_val == min_val:
        return np.zeros_like(data, dtype=float)

    # Perform Min-Max scaling
    scaled_data = (data - min_val) / (max_val - min_val)

    return scaled_data


def _humanize(value, number=None) -> str:
    """
    Converts a numeric value into a human-readable string with compact formatting.

    Args:
        value (float | int): The numeric value to format.
        number (optional): Placeholder for future use. Currently unused.

    Returns:
        str: Human-readable string representation of the number.

    Raises:
        TypeError: If `value` is not a number (int or float).
    """

    # Ensure input is numeric
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"Value must be int or float, got {type(value).__name__}"
        )

    # Negative values are represented as "0"
    if value <= 0:
        return "0"

    # Large numbers (10 and above)
    if value >= 10:
        # Choose formatter
        if value >= 1_000_000:
            formatter = "%.1f"  # e.g., 1.2M
        elif value >= 100_000:
            formatter = "%.0f"  # e.g., 120K
        else:
            formatter = "%.1f"  # e.g., 12.3K

        text = humanize.intword(value, formatter)
        text = text.replace(" thousand", "k")
        text = text.replace(" million", "M")
        text = text.replace(" billion", "G")

        # Remove trailing .0 for K, M, and B
        text = text.replace(".0k", "k")
        text = text.replace(".0M", "M")
        text = text.replace(".0G", "G")

        return text

    # Small numbers (<10)
    if value < 1:
        return f"{value:.2f}"  # e.g., 0.12
    elif value.is_integer():
        return f"{int(value)}"  # e.g., 5
    else:
        return f"{value:.1f}"  # e.g., 5.2


def _fill_holes(geometry) -> object:
    """
    Removes interior holes from Polygon or MultiPolygon geometries.

    Args:
        geometry (Polygon | MultiPolygon | object):
            The input geometry to process.
            - If Polygon, returns a new Polygon with only the exterior ring.
            - If MultiPolygon, returns a MultiPolygon with holes removed from each Polygon.
            - Other geometry types are returned unchanged.

    Returns:
        object: Geometry with holes removed if Polygon/MultiPolygon,
                otherwise returns the input geometry unchanged.

    Raises:
        ValueError: If the input geometry is invalid (e.g., None or empty).
    """

    # Ensure input geometry is valid
    if geometry is None or geometry.is_empty:
        raise ValueError("Invalid geometry: input is None or empty.")

    # If Polygon, reconstruct using only its exterior (removes holes)
    if isinstance(geometry, Polygon):
        return Polygon(geometry.exterior)

    # If MultiPolygon, apply hole removal to each sub-polygon
    elif isinstance(geometry, MultiPolygon):
        return MultiPolygon([Polygon(p.exterior) for p in geometry.geoms])

    # Return other geometry types as is (e.g., LineString, Point, etc.)
    return geometry


def _merge_data(
    full_data: gpd.GeoDataFrame, columns: list = [], how: str = "inner"
) -> gpd.GeoDataFrame:
    """
    Merges multiple GeoDataFrames or DataFrames into a single GeoDataFrame.

    Args:
        full_data (list):
            List of GeoDataFrames or DataFrames to merge.
            The first element is used as the base, and others are merged sequentially.
        columns (list, optional):
            List of column names to merge on. Defaults to [].
        how (str, optional):
            Type of merge to perform. Defaults to "inner".
            Options: {"left", "right", "outer", "inner"}.

    Returns:
        gpd.GeoDataFrame: The merged GeoDataFrame.

    Raises:
        ValueError: If `full_data` is empty or not a list of DataFrames/GeoDataFrames.
        KeyError: If merge columns are missing in one of the DataFrames.
    """

    # Ensure we have at least one dataset to merge
    if not full_data or not isinstance(full_data, list):
        raise ValueError(
            "`full_data` must be a non-empty list of DataFrames or GeoDataFrames."
        )

    # Use the first dataset as the base
    merged = full_data[0].copy()

    # Iteratively merge the remaining datasets
    for data in full_data[1:]:
        # Check if all merge columns exist in both datasets
        if not set(columns).issubset(data.columns) or not set(
            columns
        ).issubset(merged.columns):
            raise KeyError(
                f"Merge columns {columns} not found in one of the DataFrames."
            )

        merged = pd.merge(merged.copy(), data, on=columns, how=how)

    # Ensure result is a GeoDataFrame if geometry column is preserved
    if "geometry" in columns:
        merged = gpd.GeoDataFrame(merged, geometry="geometry")

    return merged


def read_config(config_file: str) -> dict:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration data as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file contains invalid syntax or cannot be parsed.
    """

    # Check if the file exists before trying to open
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    try:
        # Open the YAML configuration file in read mode
        with open(config_file, "r") as file:
            # Parse the YAML content into a Python dictionary
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        # Raise error if the YAML is invalid
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    return config
