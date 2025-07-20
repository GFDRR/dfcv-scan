import os
import yaml
import logging

import numpy as np
import pandas as pd
import geopandas as gpd

from osgeo import gdal
import rasterio as rio
import rasterio.mask
import rasterstats

import subprocess

logging.basicConfig(level=logging.INFO)


def _aggregate_acled(
    admin: gpd.GeoDataFrame,
    acled_file: str,
    asset_file: str,
    agg_file: str,
    exposure_raster: str,
    exposure_vector: str,
    adm_level: str = "ADM3",
):
    acled = gpd.read_file(acled_file)
    if not os.path.exists(agg_file):
        _aggregate_acled_exposure(acled, admin, agg_file, adm_level)
    agg = gpd.read_file(agg_file)

    if not os.path.exists(exposure_vector):
        acled_tif = _customize_acled_exposure(acled_file, asset_file)

        out_tif = _calculate_exposure(
            asset_file, acled_tif, exposure_raster, threshold=1
        )
        data = _calculate_zonal_stats(
            out_tif,
            adm_level=adm_level,
            column="conflict_exposure",
            prefix="dfcv",
            stats_agg=["sum"],
            out_file=exposure_vector,
        )
    exposure = gpd.read_file(exposure_vector)

    merge_columns = ["iso_code", adm_level, f"{adm_level}_ID", "geometry"]
    acled = _merge_data([agg, exposure], columns=merge_columns)

    return acled


def _customize_acled_exposure(
    acled_file: str,
    asset_file: str,
    temp_dir: str = "temp",
    meter_crs: str = "EPSG:3857",
    crs: str = "EPSG:4326",
) -> str:
    def get_buffer_size(event, fatality):
        if (event != "Strategic developments"):
            if (
                (event == "Riots")
                | ((event == "Violence against civilians") & (fatality == 0))
            ):
                return 2000
            return 5000
        return 0

    out_dir = "".join(acled_file.split("/")[:-1])
    filename = acled_file.split("/")[-1].split(".")[0] + "_TEMP.geojson"
    temp_file = os.path.join(out_dir, filename)

    if not os.path.exists(temp_file):
        data = gpd.read_file(acled_file)
        data["values"] = 1
        data["buffer_size"] = data.apply(
            lambda x: get_buffer_size(x.event_type, x.fatalities), axis=1
        )
        data["geometry"] = data.to_crs(meter_crs).apply(
            lambda x: x.geometry.buffer(x.buffer_size), axis=1
        )
        data = data.set_crs(meter_crs, allow_override=True).to_crs(crs)
        data.to_file(temp_file)

    filename = acled_file.replace(".geojson", ".tif")
    out_file = os.path.join(out_dir, filename)

    if not os.path.exists(out_file):
        with rio.open(asset_file) as src:
            out_image = src.read(1)
            out_image = np.zeros(out_image.shape)

            out_meta = src.meta.copy()
            with rio.open(out_file, "w", **out_meta) as dest:
                dest.write(out_image, 1)

        subprocess.call(
            ["gdal_rasterize", "-burn", "1", temp_file, out_file], shell=True
        )

    return out_file


def _aggregate_acled_exposure(
    acled: gpd.GeoDataFrame,
    admin: gpd.GeoDataFrame,
    agg_file: str,
    adm_level: str = "ADM3",
):
    def _nansumwrapper(a, **kwargs):
        if np.isnan(a).all():
            return np.nan
        else:
            return np.nansum(a, **kwargs)

    agg = acled.sjoin(admin, how="left", predicate="intersects")
    agg = agg.drop(["index_right"], axis=1)

    pop_sum = _aggregate_data(
        agg,
        adm_level=adm_level,
        agg_col="population_best",
        agg_func=lambda x: _nansumwrapper(x),
    )
    event_count = _aggregate_data(
        agg, adm_level=adm_level, agg_col="conflict_count", agg_func="count"
    )
    null_pop_event_count = _aggregate_data(
        agg[agg["population_best"].isna()],
        adm_level=adm_level,
        agg_col="null_conflict_count",
        agg_func="count",
    )
    agg = _merge_data(
        [admin, pop_sum, event_count, null_pop_event_count],
        columns=[f"{adm_level}_ID"],
        how="left",
    )

    agg["conflict_exposure"] = agg["population_best"] / (
        agg["conflict_count"] - agg["null_conflict_count"].fillna(0)
    )
    agg.to_file(agg_file)


def _generate_exposure(
    asset_file: str, local_file: str, exposure_file: str, threshold: float
) -> None:
    resampled_file = f"{local_file.split('.')[0]}_RESAMPLED.tif"

    if not os.path.exists(exposure_file):
        if not os.path.exists(resampled_file):
            _resample_raster(asset_file, local_file, resampled_file)

        _calculate_exposure(asset_file, resampled_file, exposure_file, threshold)


def _calculate_exposure(
    asset_file: str, hazard_file: str, exposure_file: str, threshold: float
) -> str:
    with rio.open(asset_file, "r") as src1, rio.open(hazard_file, "r") as src2:
        asset = src1.read(1)
        hazard = src2.read(1)

        binary = (hazard >= threshold).astype(int)
        exposure = asset * binary
        out_meta = src1.meta.copy()

    out_file = exposure_file.replace("EXPOSURE", "BINARY")
    with rio.open(out_file, "w", **out_meta) as dst:
        dst.write(binary, 1)

    with rio.open(exposure_file, "w", **out_meta) as dst:
        dst.write(exposure, 1)

    return exposure_file


def _resample_raster(asset_file: str, in_file: str, out_file: str) -> str:
    asset = gdal.Open(asset_file, 0)
    geoTransform = asset.GetGeoTransform()
    x_res = geoTransform[1]
    y_res = -geoTransform[5]

    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * (asset.RasterXSize - 1)
    miny = maxy + geoTransform[5] * (asset.RasterYSize - 1)
    out_bounds = [minx, miny, maxx, maxy]

    # call gdal Warp
    kwargs = {
        "format": "GTiff",
        "xRes": x_res,
        "yRes": y_res,
        "targetAlignedPixels": True,
        "outputBounds": out_bounds,
    }
    ds = gdal.Warp(out_file, in_file, **kwargs)
    return out_file


def _aggregate_data(
    data: gpd.GeoDataFrame,
    adm_level: str = "ADM3",
    agg_col: str = None,
    agg_func: str = "sum",
) -> gpd.GeoDataFrame:

    agg_name = f"{adm_level}_ID"
    if agg_func == "count":
        agg = data.groupby([agg_name], dropna=False).size().reset_index()
    else:
        agg = (
            data.groupby([agg_name], dropna=False)
            .agg({agg_col: agg_func})
            .reset_index()
        )

    agg.columns = [agg_name, agg_col]
    return agg


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


def _calculate_zonal_stats(
    asset_tif: str,
    column: str,
    out_file: str = None,
    adm_level: str = "ADM3",
    stats_agg: list = ["sum"],
    add_stats: list = None,
    suffix: str = None,
    prefix: str = None,
) -> gpd.GeoDataFrame:

    name = os.path.basename(asset_tif).split(".")[0]
    iso_code = name.split("_")[0]

    out_dir = os.path.abspath(os.path.join(asset_tif, os.pardir))
    if out_file is None:
        out_file = os.path.join(out_dir, f"{name}_{adm_level}.geojson")

    if not os.path.exists(out_file):
        admin_file = os.path.join(out_dir, f"{iso_code}_{adm_level}.geojson")
        admin = gpd.read_file(admin_file)
        original_crs = admin.crs

        with rio.open(asset_tif) as src:
            if admin.crs != src.crs:
                admin = admin.to_crs(src.crs)
                admin.to_file(admin_file)

        stats = rasterstats.zonal_stats(
            admin_file,
            asset_tif,
            stats=stats_agg,
            all_touched=True,
            add_stats=add_stats,
        )
        stats = pd.DataFrame(stats)
        if "custom" in stats:
            stats = stats["custom"].astype(float)

        if admin.crs != original_crs:
            admin = admin.to_crs(original_crs)
            admin.to_file(admin_file)

        data = gpd.read_file(admin_file)

        column_name = column.lower()
        if suffix is not None:
            column_name = f"{column.lower()}_{suffix}"
        if prefix is not None:
            column_name = f"{prefix}_{column.lower()}"

        data[column_name] = stats
        logging.info(f"Data saved to {out_file}.")
        data.to_file(out_file)

    return gpd.read_file(out_file)


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


def _limit_filter(data: gpd.GeoDataFrame, limit: dict):
    for column in limit:
        if isinstance(limit[column], dict):
            for subtype in limit[column]:
                temp1 = data[data[column] != subtype]
                temp2 = data[data[column] == subtype]
                for subsubtype in limit[column][subtype]:
                    subset = temp2[
                        temp2[subsubtype].isin(limit[column][subtype][subsubtype])
                    ]

            data = gpd.GeoDataFrame(pd.concat([temp1, subset]))
        elif isinstance(limit[column], list):
            data = data[data[column].isin(limit[column])]
    return data


def _exclude_filter(data: gpd.GeoDataFrame, exclude: dict):
    for column in exclude:
        data = data[~data[column].isin(exclude[column])]
    return data


def read_config(config_file: str):
    if config_file is None:
        config_file = "configs/config.yaml"

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config
