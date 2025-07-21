import os
import yaml
import logging

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio as rio
import rasterio.mask
import subprocess

logging.basicConfig(level=logging.INFO)


def calculate_multihazard_score(
    data: gpd.GeoDataFrame,
    config_file: dict = None,
    conflict_column: str = "dfcv_conflict",
    suffixes = ["exposure_relative", "exposure"]
):
    if config_file is None:
        config = read_config(config_file)

    for suffix in suffixes:
        mhs, total_weight = 0, 0
        for hazard, weight in config["weights"].items():
            if suffix is not None:
                hazard = f"{hazard}_{suffix}"

            #if hazard in 
            mhs = mhs + (data[hazard] * (weight))
            total_weight += weight

        mhs = mhs / (total_weight)

        mhs_name = "mhs"
        if suffix is not None:
            mhs_name = f"{mhs_name}_{suffix}"
        data[mhs_name] = mhs

        mhsc_name = f"mhs_{conflict_column}"
        if suffix is not None:
            mhsc_name = f"{mhsc_name}_{suffix}"

        data[mhsc_name] = data[mhs_name] * data[f"{conflict_column}_{suffix}"]

    return data


def _merge_data(
    full_data: gpd.GeoDataFrame, columns: list = [], how: str = "inner"
) -> gpd.GeoDataFrame:
    merged = full_data[0]

    for data in full_data[1:]:
        if not set(data.columns) <= set(merged.columns):
            merged = pd.merge(merged, data, on=columns, how=how)

    if "geometry" in columns:
        merged = gpd.GeoDataFrame(merged, geometry="geometry")

    return merged


def _clip_raster(
    global_tif: str, local_tif: str, admin: gpd.GeoDataFrame, nodata: list = []
) -> rio.io.DatasetReader:

    # Return existing raster if the clipped file already exists
    if not os.path.exists(local_tif):
        with rio.open(global_tif) as src:
            if src.nodata is not None:
                nodata = [src.nodata] + nodata

            # Reproject the admin boundaries if CRS differs
            if src.crs != admin.crs:
                admin0 = admin.to_crs(src.crs)

            # Extract the country boundary geometry for clipping
            shape = [admin.iloc[0]["geometry"]]

            # Perform raster clipping using rasterio.mask
            out_image, out_transform = rio.mask.mask(
                src, shape, crop=True, all_touched=True
            )
            for val in nodata:
                out_image[out_image == val] = -1

            # Update raster metadata to reflect changes
            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": -1,
                }
            )

        # Save the clipped raster to the specified output path
        with rasterio.open(local_tif, "w", **out_meta) as dest:
            dest.write(out_image)

    # Return the clipped raster
    return rio.open(local_tif)


def read_config(config_file: str):
    if config_file is None:
        config_file = "configs/config.yaml"

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config
