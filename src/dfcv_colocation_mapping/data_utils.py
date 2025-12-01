import os
import math
import yaml
import logging
import subprocess

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon

import re
import humanize
import textwrap
from stop_words import get_stop_words

logging.basicConfig(level=logging.INFO)


def aggregate_data(
    data,
    weights,
    aggregation: str = "arithmetic_mean",
    p: float = 0.5,
    epsilon: float = 0.00001,
):
    weights = weights / weights.sum()
    if aggregation == "power_mean":
        return ((data**p).multiply(weights, axis=1).sum(axis=1)) ** (1 / p)

    elif aggregation == "geometric_mean":
        return (data + epsilon).pow(weights, axis=1).prod(axis=1)

    elif aggregation == "arithmetic_mean":
        return data.multiply(weights, axis=1).sum(axis=1)


def get_deepest_values(dictionary):
    deepest_values = []
    for value in dictionary.values():
        if isinstance(value, dict):
            deepest_values.extend(get_deepest_values(value))
        else:
            deepest_values.extend(value)
    return deepest_values


def match_shape(src1: np.ndarray, src2: np.ndarray) -> np.ndarray:
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


def capitalize(string: str):
    word_list = re.split(" ", string)
    final = [word_list[0].capitalize()]
    stop_words = get_stop_words("en") + ["IDPs"]

    for word in word_list[1:]:
        newline = False
        if word.startswith("\n"):
            word = word[1:]
            newline = True
        if word not in stop_words:
            word = word.capitalize()
            word = "\n" + word if newline else word
        final.append(word)
    final = " ".join(final)
    return final


def strip_parens(string: str) -> str:
    return re.sub(r"\([^)]*\)", "", string).strip()


def find_matching_alias(
    var: str, alias_map: dict, return_var: bool = False
) -> str:
    for alias_key, alias_value in alias_map.items():
        if alias_key in var:
            if return_var:
                return alias_key
            return strip_parens(alias_value)
    return


def _minmax_scale(data: pd.Series) -> pd.Series:
    """
    Performs Min-Max scaling on a NumPy array or Pandas Series, scaling values to [0, 1].

    Args:
        data (np.ndarray or pd.Series): The input data to be scaled.

    Returns:
        np.ndarray or pd.Series: The scaled data with values between 0 and 1.
    """
    # Compute min and max, ignoring NaNs
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)

    # Handle case where all values are identical
    if max_val == min_val:
        return np.zeros_like(data, dtype=float)

    # Perform Min-Max scaling
    scaled_data = (data - min_val) / (max_val - min_val)

    return scaled_data


def _nice_round(x):
    if x <= 5:
        return 5
    elif x <= 10:
        return 10
    elif x <= 50:
        return math.ceil(x / 10) * 10
    elif x <= 100:
        return math.ceil(x / 50) * 50
    elif x <= 500:
        return math.ceil(x / 100) * 100
    elif x <= 1000:
        return math.ceil(x / 500) * 500
    else:
        return math.ceil(x / 1000) * 1000


def _humanize(value, number=None) -> str:
    """
    Converts a numeric value into a human-readable string with compact formatting.

    Args:
        value (float | int): The numeric value to format.
        number (optional): Placeholder for future use. Currently unused.

    Returns:
        str: Human-readable string representation of the number.
    """
    if value <= 0:
        return "0"

    if value >= 10:
        # Choose formatter
        if value >= 1_000_000:
            formatter = "%.1f"
        elif value >= 100_000:
            formatter = "%.0f"
        else:
            formatter = "%.1f"

        text = humanize.intword(value, formatter)
        text = text.replace(" thousand", "k")
        text = text.replace(" million", "M")
        text = text.replace(" billion", "B")

        # Remove trailing .0 for K, M, and B
        text = text.replace(".0k", "k")
        text = text.replace(".0M", "M")
        text = text.replace(".0B", "B")

        return text

    # Small numbers (<10)
    if value < 1:
        if value < 0.1:
            if value < 0.01:
                return f"{value:.4f}"
            return f"{value:.3f}"
        return f"{value:.2f}"
    elif value.is_integer():
        return f"{int(value)}"
    else:
        return f"{value:.1f}"


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
    full_data: gpd.GeoDataFrame, columns: list = [], how: str = "left"
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
    """
    # Use the first dataset as the base
    merged = full_data[0].copy()

    # Iteratively merge the remaining datasets
    for data in full_data[1:]:
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
    """
    try:
        # Open the YAML configuration file in read mode
        with open(config_file, "r") as file:
            # Parse the YAML content into a Python dictionary
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        # Raise error if the YAML is invalid
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    return config
