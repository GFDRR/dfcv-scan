# Standard library
import datetime
import itertools
import json
import logging
import os
import re
import shutil
import urllib.request
import warnings
import zipfile
from functools import reduce
from pathlib import Path
from warnings import simplefilter

# Third-party libraries
import geojson
import importlib_resources
import numpy as np
import pandas as pd
import pycountry
import requests
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

# GIS and geospatial
import geopandas as gpd
import rasterio as rio
import rasterio.mask
import rasterstats
from osgeo import gdal, gdalconst

# Analysis, math, and statistics
from scipy.stats.mstats import gmean

# Other specialized libraries
import ahpy
import bs4
import osmnx as ox
from dtmapi import DTMApi

# Local package import
from dfcv_colocation_mapping import data_utils

pd.set_option("future.no_silent_downcasting", True)
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
ox.settings.max_query_area_size = 500000000000000
logging.basicConfig(level=logging.INFO, force=True)
io_logger = logging.getLogger("pyogrio._io")
io_logger.setLevel(logging.WARNING)

WARNING = "\033[31m"
RESET = "\033[0m"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class DatasetManager:
    def __init__(
        self,
        iso_code: str,
        adm_level: str = "ADM3",
        group: str = "Region",
        adm_source: str = "geoboundary",
        meter_crs: str = "EPSG:3857",
        crs: str = "EPSG:4326",
        # Conflict Configuration
        acled_key: str = None,
        acled_country: str = None,
        conflict_start_date: str = None,
        conflict_end_date: str = None,
        conflict_last_n_years: int = 10,
        # Displacement configuration
        dtm_key: str = None,
        dtm_adm_level: str = None,
        idmc_key: str = None,
        displacement_start_date: str = None,
        displacement_end_date: str = None,
        displacement_last_n_years: int = 10,
        # Asset and hazard configurations
        resample_worldcover: bool = True,
        fathom_year: int = 2020,
        fathom_rp: int = 50,
        jrc_rp: int = 100,
        mhs_aggregation: str = "power_mean",
        # Global variable names
        acled_name: str = "acled",
        ucdp_name: str = "ucdp",
        fathom_name: str = "fathom",
        global_name: str = "global",
        # Config file locations
        config_file: str = None,
        dtm_cred_file: str = None,
        idmc_cred_file: str = None,
        acled_cred_file: str = None,
        adm_config_file: str = None,
        osm_config_file: str = None,
        acled_config_file: str = None,
        # Download configurations
        data_dir: str = "data",
        overwrite: bool = False,
    ):
        # === Core country information ===
        self.iso_code = iso_code
        self.adm_level = adm_level
        self.adm_source = adm_source
        self.crs = crs
        self.meter_crs = meter_crs
        self.data_dir = data_dir
        self.overwrite = overwrite

        # === Conflict configurations ===
        self.acled_key = acled_key
        self.acled_country = acled_country
        self.conflict_start_date = self._get_start_date(
            conflict_start_date, conflict_last_n_years
        )
        self.conflict_end_date = self._get_end_date(conflict_end_date)

        # === Displacement configurations ===
        self.dtm_adm_level = self._get_dtm_adm_level(dtm_adm_level)
        self.displacement_start_date = self._get_start_date(
            displacement_start_date, displacement_last_n_years
        )
        self.displacement_end_date = self._get_end_date(displacement_end_date)

        # === Asset and hazard configurations ===
        self.resample_worldcover = resample_worldcover
        self.fathom_year = fathom_year
        self.fathom_rp = fathom_rp
        self.jrc_rp = jrc_rp

        # Locate default configuration files if not provided
        self.config_file = self._resolve_config_path(
            config_file, "data_config.yaml"
        )
        self.acled_cred_file = self._resolve_config_path(
            acled_cred_file, "acled_creds.yaml"
        )
        self.acled_config_file = self._resolve_config_path(
            acled_config_file, "acled_config.yaml"
        )
        self.dtm_cred_file = self._resolve_config_path(
            dtm_cred_file, "dtm_creds.yaml"
        )
        self.idmc_cred_file = self._resolve_config_path(
            idmc_cred_file, "dtm_creds.yaml"
        )
        self.adm_config_file = self._resolve_config_path(
            adm_config_file, "adm_config.yaml"
        )
        self.osm_config_file = self._resolve_config_path(
            osm_config_file, "osm_config.yaml"
        )

        # Load main config
        self.config = data_utils.read_config(self.config_file)
        self.osm_config = data_utils.read_config(self.osm_config_file)
        self.acled_config = data_utils.read_config(self.acled_config_file)
        self.adm_config = data_utils.read_config(self.adm_config_file)
        self.country = self._get_country_name(self.iso_code)

        # Load credentials
        self.acled_key = self._load_creds(
            self.acled_cred_file, "access_token", acled_key
        )
        self.dtm_key = self._load_creds(self.dtm_cred_file, "dtm_key", dtm_key)
        self.idmc_key = self._load_creds(
            self.idmc_cred_file, "idmc_key", idmc_key
        )

        self.acled_hierarchy = self.acled_config["acled_hierarchy"]
        self.acled_selected = self.acled_config["acled_selected"]

        # Uppercase standard labels
        self.global_name = global_name.upper()
        self.fathom_name = fathom_name.upper()
        self.acled_name = acled_name.upper()
        self.ucdp_name = ucdp_name.upper()

        # Prepare directories for storing data
        self.data_dir = os.path.join(os.getcwd(), data_dir)
        self.local_dir = os.path.join(self.data_dir, iso_code)
        self.global_dir = os.path.join(self.data_dir, self.global_name)

        # Build file path for asset layer
        self.asset_names, self.asset_files = self._get_asset_names_and_files()
        self.mhs_aggregation = mhs_aggregation

    def _get_country_name(self, iso_code: str):
        country = pycountry.countries.get(alpha_3=iso_code).name
        for config_iso_code in self.config["country_map_code"]:
            if iso_code == config_iso_code:
                return self.config["country_map_code"][iso_code]
        if country is None:
            raise ValueError(f"{WARNING}Invalid ISO code: {iso_code}{RESET}")
        return country

    def _resolve_config_path(self, provided_path, filename):
        resources = importlib_resources.files("dfcv_colocation_mapping")
        return provided_path or resources.joinpath("configs", filename)

    def _load_creds(self, file_path: str, key_field: str, default: str = None):
        if os.path.exists(file_path):
            creds = data_utils.read_config(file_path)
            return creds.get(key_field)
        return default

    def _get_start_date(self, start_date, last_n_years):
        if start_date is None:
            start_date = (
                datetime.date.today() - relativedelta(years=last_n_years)
            ).isoformat()

        return start_date

    def _get_end_date(self, end_date):
        if end_date is None:
            end_date = datetime.date.today().isoformat()

        return end_date

    def _get_dtm_adm_level(self, dtm_adm_level):
        if dtm_adm_level is None:
            if int(self.adm_level[-1]) > 2:
                dtm_adm_level = "ADM2"
            else:
                dtm_adm_level = self.adm_level

        return dtm_adm_level

    def _get_asset_names_and_files(self):
        asset_names = []
        asset_files = []

        for asset in self.config["asset_data"]:
            asset = asset.replace(f"{self.global_name.lower()}_", "")
            asset_file = self._build_filename(
                self.iso_code, asset, self.local_dir, ext="tif"
            )
            asset_names.append(asset)
            asset_files.append(asset_file)

        return asset_names, asset_files

    def _download_with_aggregate(self, func, *args, **kwargs):
        base = func(*args, **kwargs)
        agg = (
            func(*args, **kwargs, aggregate=True) if base is not None else None
        )
        return base, agg

    def _assign_grouping(self, iso_code, data, config):
        group = None
        if iso_code not in config:
            return data, group

        config = config[iso_code]
        group = config["group"]

        # Only add grouping column if it doesn't exist yet
        if group not in data.columns:
            adm_level = config["adm_level"]
            grouping = config["grouping"]

            # Map administrative level to group
            data[group] = data[adm_level].map(grouping)

        return data, group

    def download_datasets(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.local_dir, exist_ok=True)
        os.makedirs(self.global_dir, exist_ok=True)

        # Load geoboundaries, fallback to GADM if primary source fails
        logging.info(f"Loading {self.adm_level} geoboundaries...")
        self.geoboundary = self.download_geoboundary_with_attempts()
        self.merge_columns = list(self.geoboundary.columns)

        # Load hazard layers
        logging.info("Loading asset layers...")
        self.assets = self.download_assets()

        logging.info("Loading hazard layers...")
        self.hazards = self.download_hazards()

        logging.info(
            f"Loading conflict data from {self.conflict_start_date} to {self.conflict_end_date}..."
        )
        # Load acled conflict data
        logging.info("Loading ACLED data...")
        self.acled, self.acled_agg = self._download_with_aggregate(
            self.download_acled
        )

        # Load ucdp conflict data
        logging.info("Loading UCDP data...")
        self.ucdp, self.ucdp_agg = self._download_with_aggregate(
            self.download_ucdp
        )

        logging.info(
            f"Loading displacement data from {self.displacement_start_date} to {self.displacement_end_date}..."
        )
        logging.info("Loading IOM DTM data...")
        self.dtm, self.dtm_agg = self._download_with_aggregate(
            self.download_dtm, self.dtm_adm_level, filtered=True
        )

        logging.info("Loading IDMC data...")
        self.idmc_gidd_conflict, self.idmc_gidd_conflict_agg = (
            self._download_with_aggregate(
                self.download_idmc_gidd, cause="conflict"
            )
        )
        self.idmc_gidd_disaster, self.idmc_gidd_disaster_agg = (
            self._download_with_aggregate(
                self.download_idmc_gidd, cause="disaster"
            )
        )

        # Compute multi-hazard scores
        logging.info("Calculating Multihazard Scores...")
        self.data = self._combine_datasets()
        self.data = self._calculate_multihazard_score(self.data)

        logging.info("Downloading OSM...")
        self.osm = self.download_osm()

    def _combine_datasets(self) -> gpd.GeoDataFrame:
        data = []

        # Add assets and hazard datasets (replace NaN with 0)
        for dataset in [self.assets, self.hazards]:
            if dataset is not None:
                data.append(dataset.fillna(0))

        # Define optional datasets with validity conditions
        optional_datasets = [
            (self.dtm_agg, self.dtm_adm_level == self.adm_level),
            (self.idmc_gidd_conflict_agg, True),
            (self.idmc_gidd_disaster_agg, True),
            (self.acled_agg, True),
            (self.ucdp_agg, True),
        ]

        # Add datasets that are non-null, non-empty, and meet conditions
        for ds, condition in optional_datasets:
            if ds is not None and len(ds) > 0 and condition:
                data.append(ds)

        # Merge and finalize
        data = data_utils._merge_data(data, columns=self.merge_columns)
        data = self._calculate_idmc_idp_total(data)

        return data

    def _calculate_multihazard_score(
        self,
        data: gpd.GeoDataFrame,
        conflict_columns: list = ["wbg_acled", "ucdp"],
        suffixes: list = [
            "exposure",
            "exposure_relative",
            "intensity_weighted_exposure_relative",
        ],
        aggregation: str = "power_mean",
        p: float = 0.5,
        epsilon: float = 0.00001,
    ) -> gpd.GeoDataFrame:
        """
        Calculates multi-hazard scores (MHS) across hazard types using arithmetic,
        geometric, or power mean aggregation. Optionally multiplies MHS by conflict exposure.
        """

        # Ensure relative exposure columns exist for all hazard columns
        for asset_name in self.asset_names:
            for column in data.columns:
                if "relative" not in column and asset_name in column:
                    colname = f"{column}_relative"
                    if "exposure" in column and colname not in data.columns:
                        data[column] = (
                            data[column].astype(float).fillna(np.nan)
                        )
                        data[colname] = data[column].div(
                            data[asset_name].where(
                                data[asset_name] != 0, np.nan
                            )
                        )

        # Scale worldcover columns if needed
        for column in data.columns:
            if "worldcover" in column and "relative" not in column:
                data[column] = data[column] * 0.01

        # Prepare hazard categories and unified dict
        hazard_dicts = [
            self.config["hazards"][cat] for cat in self.config["hazards"]
        ]
        categories = list(self.config["hazards"].keys())

        all_hazards = {k: v for d in hazard_dicts for k, v in d.items()}
        hazard_dicts.append(all_hazards)
        categories.append("all")

        # Loop through each suffix to calculate MHS
        for suffix in suffixes:
            for hazard_dict, category in zip(hazard_dicts, categories):
                for asset in self.asset_names:
                    # Collect available hazard columns for this asset/suffix
                    hazards, hazard_cols = [], []
                    for hazard in hazard_dict:
                        hazard_col = f"{hazard}_{asset}_{suffix}"
                        if hazard_col in data.columns:
                            if not (data[hazard_col] == 0).all():
                                hazard_cols.append(hazard_col)
                                hazards.append(hazard)

                    if len(hazards) == 0:
                        continue

                    # Align weights to existing hazard columns
                    weights = np.array(
                        [hazard_dict[hazard] for hazard in hazards]
                    )
                    weights = weights / weights.sum()

                    # Compute MHS using selected aggregation
                    if aggregation == "power_mean":
                        mhs = (
                            (data[hazard_cols] ** p)
                            .multiply(weights, axis=1)
                            .sum(axis=1)
                        ) ** (1 / p)

                    elif aggregation == "geometric_mean":
                        mhs = (
                            (data[hazard_cols] + epsilon)
                            .pow(weights, axis=1)
                            .prod(axis=1)
                        )

                    elif aggregation == "arithmetic_mean":
                        mhs = (
                            data[hazard_cols]
                            .multiply(weights, axis=1)
                            .sum(axis=1)
                        )

                    # Add scaled MHS column
                    mhs_name = f"mhs_{category}_{asset}_{suffix}"
                    data[mhs_name] = data_utils._minmax_scale(mhs)

                    # Multiply MHS by conflict exposure (scaled)
                    for conflict_column in conflict_columns:
                        conflict_col = f"{conflict_column}_{asset}_{suffix}"
                        if conflict_col in data.columns:
                            conflict_scaled = data_utils._minmax_scale(
                                data[conflict_col]
                            )
                            mhsc_name = f"mhs_{category}_{conflict_column}_{asset}_{suffix}"
                            data[mhsc_name] = data[mhs_name] * conflict_scaled

        return data

    def download_geoboundary_with_attempts(self, attempts: int = 3):
        for i in range(attempts):
            try:
                return self.download_geoboundary(
                    self.adm_source, self.adm_level
                )
            except Exception as err:
                logging.info(err)
                logging.info(
                    f"Loading geoboundaries failed. Trying again {i+1}/{attempts}"
                )
                pass

        logging.info("Loading geoboundaries failed. Trying with GADM...")
        return self.download_geoboundary("gadm", self.adm_level)

    def download_geoboundary(
        self, adm_source: str, adm_level: str, overwrite: bool = False
    ) -> gpd.GeoDataFrame:

        # Build output filenames
        out_file = self._build_filename(
            self.iso_code,
            f"{adm_source}_{adm_level}",
            self.local_dir,
            ext="geojson",
        )
        gadm_file = self._build_filename(
            self.iso_code, f"gadm_{adm_level}", self.local_dir, ext="geojson"
        )

        # Prefer GADM file if exists
        if not overwrite and os.path.exists(gadm_file):
            adm_source = "gadm"
            out_file = gadm_file

        # Download only if file doesn't exist or overwrite is True
        elif overwrite or not os.path.exists(out_file):
            # Download GADM dataset
            if adm_source == "gadm":
                try:
                    self.download_url(
                        adm_source,
                        dataset_name=f"{adm_source}_{adm_level}",
                        ext="geojson",
                    )
                    geoboundary = gpd.read_file(out_file)
                except Exception as e:
                    raise FileNotFoundError(
                        f"Failed to download or read GADM data: {str(e)}"
                    )

                # Rename columns to standard format
                rename = dict()
                for index in range(int(adm_level[-1]) + 1):
                    if index == 0:
                        rename[f"GID_{index}"] = "iso_code"
                    else:
                        rename[f"GID_{index}"] = f"ADM{index}_ID"
                        rename[f"NAME_{index}"] = f"ADM{index}"

                geoboundary = geoboundary.rename(columns=rename)
                all_columns = list(rename.values()) + ["geometry"]
                geoboundary = geoboundary[all_columns]

                geoboundary.to_file(out_file)
                logging.info(
                    f"Geoboundary file saved to {os.path.basename(out_file)}."
                )

            elif adm_source == "geoboundary":
                # Download GeoBoundaries dataset
                gbhumanitarian_url = self.config["urls"]["gbhumanitarian_url"]
                gbopen_url = self.config["urls"]["gbopen_url"]
                level = int(adm_level[-1])

                # Download each administrative level
                datasets = []
                for index in range(1, level + 1):
                    adm_level = f"ADM{index}"
                    intermediate_file = self._build_filename(
                        self.iso_code,
                        f"{adm_source}_{adm_level}",
                        self.local_dir,
                        ext="geojson",
                    )

                    if not os.path.exists(intermediate_file):
                        # Try GBHumanitarian URL first
                        url = (
                            f"{gbhumanitarian_url}{self.iso_code}/{adm_level}/"
                        )
                        try:
                            r = requests.get(url)
                            download_path = r.json()["gjDownloadURL"]
                        except Exception:
                            # Fallback to GBOpen URL if GBHumanitarian URL fails
                            try:
                                url = (
                                    f"{gbopen_url}{self.iso_code}/{adm_level}/"
                                )
                                r = requests.get(url)
                                download_path = r.json()["gjDownloadURL"]
                            except Exception as e:
                                raise requests.RequestException(
                                    f"Failed to download {adm_level} boundaries: {str(e)}"
                                )

                        # Download and save the GeoJSON data
                        try:
                            geoboundary = requests.get(download_path).json()
                            with open(intermediate_file, "w") as file:
                                geojson.dump(geoboundary, file)
                        except Exception as e:
                            raise FileNotFoundError(
                                f"Failed to save GeoJSON file {os.path.basename(intermediate_file)}: {str(e)}"
                            )

                    try:
                        # Read the downloaded GeoJSON into a GeoDataFrame
                        geoboundary = gpd.read_file(intermediate_file)
                        geoboundary["iso_code"] = self.iso_code

                        # Select relevant columns and rename them
                        if (
                            "shapeName" in geoboundary.columns
                            and "shapeID" in geoboundary.columns
                        ):
                            geoboundary = geoboundary[
                                [
                                    "iso_code",
                                    "shapeName",
                                    "shapeID",
                                    "geometry",
                                ]
                            ]
                            geoboundary.columns = [
                                "iso_code",
                                adm_level,
                                f"{adm_level}_ID",
                                "geometry",
                            ]

                        # Save geoboundary with renamed columns
                        datasets.append(geoboundary)
                        geoboundary.to_file(intermediate_file)
                        logging.info(
                            f"Geoboundary file saved to {os.path.basename(intermediate_file)}."
                        )
                    except Exception as e:
                        raise FileNotFoundError(
                            f"Failed to read GeoJSON file {os.path.basename(intermediate_file)}: {str(e)}"
                        )

                # Merge multiple levels
                geoboundary = datasets[-1].to_crs(self.meter_crs)
                columns = geoboundary.columns

                # Iterate through the remaining DataFrames and perform joins
                for index in reversed(range(level - 1)):
                    current = datasets[index].to_crs(self.meter_crs)
                    join_columns = [
                        f"ADM{index+1}_ID",
                        f"ADM{index+1}",
                        "geometry",
                    ]
                    joined = geoboundary.sjoin(
                        current[join_columns], predicate="intersects"
                    ).drop(columns=["index_right"])
                    joined = joined.to_crs(self.meter_crs)

                    # Calculate the intersection area and percentage overlap
                    adm = join_columns[0]
                    joined["intersection_area"] = joined.apply(
                        lambda row: row.geometry.intersection(
                            current[current[adm] == row[adm]].iloc[0].geometry
                        ).area,
                        axis=1,
                    )
                    joined["overlap_percentage"] = (
                        joined["intersection_area"]
                        / joined["geometry"].area
                        * 100
                    )

                    # Filter for the desired overlap percentage
                    geoboundary = joined[joined["overlap_percentage"] >= 50]
                    columns = list(columns) + list(join_columns[:-1])
                    geoboundary = geoboundary[columns]
            else:
                raise ValueError(
                    f"{WARNING}adm_source '{adm_source}' not recognized. Use 'gadm' or 'geoboundary'.{RESET}"
                )

            geoboundary.to_crs(self.crs).to_file(out_file)

        # Load final geoboundary and update attributes
        geoboundary = gpd.read_file(out_file).to_crs(self.crs)

        out_file = self._build_filename(
            self.iso_code,
            adm_level,
            self.local_dir,
            ext="geojson",
        )
        group = None
        if not overwrite and not os.path.exists(out_file):
            geoboundary, group = self._assign_grouping(
                self.iso_code, geoboundary, self.adm_config
            )
            geoboundary.to_crs(self.crs).to_file(out_file)
            logging.info(
                f"Geoboundary file saved to {os.path.basename(out_file)}."
            )

        geoboundary = gpd.read_file(out_file)

        if adm_level == self.adm_level:
            self.group = group
            self.admin_file = out_file
            self.adm_source = adm_source
            self.merge_columns = list(geoboundary.columns)

        return geoboundary

    def download_osm(self) -> gpd.GeoDataFrame:
        out_file = self._build_filename(
            self.iso_code, "OSM", self.local_dir, ext="geojson"
        )

        if os.path.exists(out_file):
            return gpd.read_file(out_file)

        osm = []
        ox.settings.log_console = True

        categories = self.osm_config["keywords"]
        for category in (
            pbar := tqdm(categories, total=len(categories), dynamic_ncols=True)
        ):
            pbar.set_description(f"Processing {category}")
            tags = categories[category]
            admin_bounds = self.geoboundary.dissolve("iso_code")[
                "geometry"
            ].envelope.values[0]
            data = ox.features.features_from_polygon(
                admin_bounds, tags
            ).to_crs(self.meter_crs)
            data.geometry = data.geometry.centroid
            data = data.to_crs(self.crs)

            data["category"] = category.replace("_", " ").title()
            osm.append(data[["geometry", "amenity", "category"]])

        osm = gpd.GeoDataFrame(pd.concat(osm)).reset_index().to_crs(self.crs)
        osm = osm.rename(
            columns={"category": "osm_category", "amenity": "osm_amenity"}
        )
        osm = osm.sjoin(self.geoboundary, how="left", predicate="intersects")
        osm = osm.drop(["index_right"], axis=1)
        osm.to_file(out_file, driver="GeoJSON")
        return osm

    def download_idmc_gidd(
        self,
        cause: str = "conflict",
        name: str = "idmc",
        adm_level: str = None,
        filtered: bool = True,
        aggregate: bool = False,
    ):
        gidd_file = self._build_filename(
            self.iso_code, f"{name}_{cause}", self.local_dir, ext="geojson"
        )
        filtered_file = self._build_filename(
            self.iso_code,
            f"{name}_{cause}_filtered",
            self.local_dir,
            ext="geojson",
        )

        if self.overwrite or not os.path.exists(gidd_file):
            try:
                idmc_gidd_url = self.config["urls"]["idmc_gidd_url"].format(
                    self.idmc_key, self.iso_code, cause
                )
                self._download_url_progress(idmc_gidd_url, gidd_file)
            except:
                logging.info(
                    f"{WARNING}WARNING: IDMC data failed to download.{RESET}"
                )
                logging.info(
                    f"{WARNING}Please ensure you IDMC API key is correct.{RESET}"
                )
                return

            idmc_gidd = gpd.read_file(gidd_file, use_arrow=True)
            idmc_gidd = idmc_gidd.sjoin(
                self.geoboundary, how="left", predicate="intersects"
            )
            idmc_gidd = idmc_gidd.drop(["index_right"], axis=1)
            idmc_gidd.to_file(gidd_file)

            if filtered:
                if len(idmc_gidd) == 0:
                    return

                if self.overwrite or not os.path.exists(filtered_file):
                    idmc_gidd_filtered = idmc_gidd.copy()

                    if (
                        "Start date" in idmc_gidd_filtered.columns
                        and "End date" in idmc_gidd_filtered.columns
                    ):
                        idmc_gidd_filtered["Start date"] = pd.to_datetime(
                            idmc_gidd_filtered["Start date"]
                        )
                        idmc_gidd_filtered["End date"] = pd.to_datetime(
                            idmc_gidd_filtered["End date"]
                        )

                        idmc_gidd_filtered = idmc_gidd_filtered[
                            (
                                idmc_gidd_filtered["Start date"]
                                >= self.displacement_start_date
                            )
                            & (
                                idmc_gidd_filtered["End date"]
                                <= self.displacement_end_date
                            )
                        ]

                    elif "Stock date" in idmc_gidd_filtered.columns:
                        idmc_gidd_filtered["Stock date"] = pd.to_datetime(
                            idmc_gidd_filtered["Stock date"]
                        )
                        idmc_gidd_filtered = idmc_gidd_filtered[
                            (
                                idmc_gidd_filtered["Stock date"]
                                >= self.displacement_start_date
                            )
                            & (
                                idmc_gidd_filtered["Stock date"]
                                <= self.displacement_end_date
                            )
                        ]
                    idmc_gidd_filtered.to_file(filtered_file)

        idmc_gidd = gpd.read_file(gidd_file, use_arrow=True)
        if filtered and os.path.exists(filtered_file):
            idmc_gidd = gpd.read_file(filtered_file, use_arrow=True)

        if aggregate:
            if len(idmc_gidd) == 0:
                return

            if adm_level is None:
                adm_level = self.adm_level

            agg_file = self._build_filename(
                self.iso_code,
                f"{name}_{cause}_{adm_level}",
                self.local_dir,
                ext="geojson",
            )

            if self.overwrite or not os.path.exists(agg_file):
                idp_column = "Total figures"
                idmc_gidd_agg = self._aggregate_data(
                    idmc_gidd[[adm_level, f"{adm_level}_ID", idp_column]],
                    agg_col=idp_column,
                    agg_func="sum",
                    adm_level=adm_level,
                )
                idmc_gidd_agg = idmc_gidd_agg.rename(
                    columns={idp_column: f"idmc_{cause}_idp_total"}
                )
                admin = self.geoboundary
                idmc_gidd_agg = data_utils._merge_data(
                    [admin, idmc_gidd_agg],
                    columns=[f"{adm_level}_ID"],
                    how="left",
                )
                idmc_gidd_agg.to_file(agg_file)

            idmc_gidd_agg = gpd.read_file(agg_file)
            return idmc_gidd_agg

        return idmc_gidd

    def _calculate_idmc_idp_total(
        self,
        data,
        col: str = "idmc_idp_total",
        disaster_col: str = "idmc_disaster_idp_total",
        conflict_col: str = "idmc_conflict_idp_total",
    ):
        if conflict_col in data.columns and disaster_col in data.columns:
            data[col] = data[conflict_col].add(
                data[disaster_col], fill_value=0
            )
        elif conflict_col in data.columns:
            data[col] = data[conflict_col]
        elif disaster_col in data.columns:
            data[col] = data[disaster_col]

        return data

    def download_dtm(
        self,
        dtm_adm_level: str = "ADM2",
        idp_column: str = "numPresentIdpInd",
        year: int = None,
        filtered: bool = False,
        aggregate: bool = False,
    ):
        raw_file = self._build_filename(
            self.iso_code,
            f"DTM_{dtm_adm_level}_RAW",
            self.local_dir,
            ext="csv",
        )
        filtered_file = self._build_filename(
            self.iso_code,
            f"DTM_{dtm_adm_level}_FILTERED",
            self.local_dir,
            ext="csv",
        )

        if self.dtm_key is None:
            return
        else:
            try:
                api = DTMApi(subscription_key=self.dtm_key)
                self.dtm_countries = api.get_all_countries()
            except:
                logging.info(
                    f"{WARNING}WARNING: Network connection to dtm.iom.int could not be established.{RESET}"
                )
                logging.info(
                    f"{WARNING}WARNING: DTM data failed to download.{RESET}"
                )
                return

        if (
            self.overwrite
            or not os.path.exists(raw_file)
            or not os.path.exists(filtered_file)
        ):
            try:
                country_name = self.dtm_countries[
                    self.dtm_countries["admin0Pcode"] == self.iso_code
                ]["admin0Name"].values[0]
            except:
                logging.info(
                    f"{WARNING}WARNING: No DTM data available for {self.iso_code} ({self.country}).{RESET}"
                )
                return

            try:
                adm = self.download_geoboundary(
                    adm_source="geoboundary",
                    adm_level=dtm_adm_level,
                    overwrite=False,
                )
            except Exception as e:
                logging.info(e)
                adm = self.download_geoboundary(
                    adm_source="gadm", adm_level=dtm_adm_level
                )

            dtm = None
            if dtm_adm_level == "ADM1":
                dtm = api.get_idp_admin1_data(
                    CountryName=country_name,
                    FromReportingDate=self.displacement_start_date,
                    ToReportingDate=self.displacement_end_date,
                )

            elif dtm_adm_level == "ADM2":
                dtm = api.get_idp_admin2_data(
                    CountryName=country_name,
                    FromReportingDate=self.displacement_start_date,
                    ToReportingDate=self.displacement_end_date,
                )

            if len(dtm) == 0:
                logging.info(
                    f"{WARNING}WARNING: No DTM data available for {self.iso_code} ({self.country}).{RESET}"
                )
                return

            dtm.to_csv(raw_file)

            if filtered:
                dtm_filtered = dtm.copy()
                dtm_filtered.yearReportingDate = (
                    dtm_filtered.yearReportingDate.astype(int)
                )
                max_year = dtm_filtered.yearReportingDate.max()
                year = max_year if year is None else min(year, max_year)

                dtm_filtered = dtm_filtered[
                    dtm_filtered.yearReportingDate == year
                ]
                dtm_filtered.roundNumber = dtm_filtered.roundNumber.astype(int)
                dtm_filtered = dtm_filtered[
                    dtm_filtered.roundNumber == dtm_filtered.roundNumber.max()
                ]
                dtm_filtered.to_csv(filtered_file)

        dtm = pd.read_csv(raw_file)
        if filtered:
            dtm = pd.read_csv(filtered_file)

        if aggregate:
            geojson_file = self._build_filename(
                self.iso_code,
                f"DTM_{dtm_adm_level}",
                self.local_dir,
                ext="geojson",
            )

            dtm_adm_level_num = dtm_adm_level[-1]
            column = f"admin{dtm_adm_level_num}Name"
            dtm_agg = self._aggregate_data(
                dtm[[column, idp_column]],
                agg_col=idp_column,
                agg_func="sum",
                adm_level=column,
            )
            dtm_agg = dtm_agg.rename(columns={idp_column: "dtm_idp_total"})

            adm = self.geoboundary
            dtm_agg = adm.merge(
                dtm_agg, left_on=dtm_adm_level, right_on=column, how="left"
            )
            dtm_agg.to_crs(self.crs).to_file(geojson_file)
            dtm_agg = gpd.read_file(geojson_file)
            return dtm_agg

        return dtm

    def download_ucdp(self, aggregate: bool = False):
        local_file = self._build_filename(
            self.iso_code, self.ucdp_name, self.local_dir, ext="geojson"
        )
        global_file = self._build_filename(
            self.global_name, self.ucdp_name, self.global_dir, ext="csv"
        )

        if self.overwrite or not os.path.exists(global_file):
            try:
                dataset = f"{self.global_name}_{self.ucdp_name}".lower()
                self.download_url(dataset=dataset, ext="csv")
            except:
                logging.info(
                    f"{WARNING}WARNING: UCDP Data failed to download.{RESET}"
                )
                return

        if self.overwrite or not os.path.exists(local_file):
            ucdp = pd.read_csv(global_file, low_memory=False)
            ucdp["country"] = ucdp["country"].apply(
                lambda x: re.sub(r"\s*\([^)]*\)", "", x)
            )
            ucdp["country"] = ucdp["country"].str.strip()

            country = self.country
            if self.iso_code == "COD":
                country = "DR Congo"

            ucdp = ucdp[ucdp["country"] == country]
            ucdp["date_start"] = pd.to_datetime(ucdp["date_start"])
            ucdp = ucdp[ucdp["date_start"] >= self.conflict_start_date]
            ucdp = ucdp[ucdp["date_start"] <= self.conflict_end_date]

            type_of_violence_map = {
                1: "State-based conflict",
                2: "Non-state conflict",
                3: "One-sided violence",
            }
            ucdp["type_of_violence"] = ucdp["type_of_violence"].replace(
                type_of_violence_map
            )

            if len(ucdp) == 0:
                logging.info(f"No UCDP data found for {self.iso_code}.")
                return

            ucdp = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(
                    ucdp["longitude"], ucdp["latitude"], crs=self.crs
                ),
                data=ucdp,
            )

            # Spatial join ACLED events with admin boundaries
            admin = self.geoboundary
            ucdp = ucdp.sjoin(admin, how="left", predicate="intersects")
            ucdp = ucdp.drop(["index_right"], axis=1)

            ucdp.to_file(local_file, driver="GeoJSON")
            logging.info(f"Saving UCDP to {local_file}")

        ucdp = gpd.read_file(local_file)

        if aggregate:
            ucdp = self._aggregate_ucdp(ucdp, local_file)

        return ucdp

    def _aggregate_ucdp(self, ucdp, local_file: str):
        ucdp_agg = None
        admin = self.geoboundary

        for asset_name, asset_file in (
            pbar := tqdm(
                zip(self.asset_names, self.asset_files),
                total=len(self.asset_names),
            )
        ):
            pbar.set_description(f"Processing {asset_name}")
            column = f"{self.ucdp_name.lower()}_{asset_name}_exposure"
            exposure_raster = self._build_filename(
                self.iso_code,
                f"{self.ucdp_name}_{asset_name}_exposure",
                self.local_dir,
                ext="tif",
            )
            exposure_vector = self._build_filename(
                self.iso_code,
                f"{self.ucdp_name}_{asset_name}_exposure_{self.adm_level}",
                self.local_dir,
                ext="geojson",
            )

            if self.overwrite or not os.path.exists(exposure_vector):
                out_tif = self._calculate_custom_conflict_exposure(
                    local_file,
                    asset_file,
                    asset_name=asset_name,
                    conflict_src="ucdp",
                )
                out_tif, _ = self._calculate_exposure(
                    asset_file, out_tif, exposure_raster, threshold=1
                )

                self._calculate_zonal_stats(
                    out_tif,
                    column=column,
                    stats_agg=["sum"],
                    out_file=exposure_vector,
                )

            # Read exposure vector and clean zero values
            ucdp_agg_sub = gpd.read_file(exposure_vector)
            ucdp_agg_sub.loc[ucdp_agg_sub[column] == 0, column] = None

            ucdp_agg = (
                ucdp_agg_sub
                if ucdp_agg is None
                else data_utils._merge_data(
                    [ucdp_agg, ucdp_agg_sub], columns=self.merge_columns
                )
            )

        final_exposure_vector = self._build_filename(
            self.iso_code,
            f"{self.ucdp_name}_exposure_{self.adm_level}",
            self.local_dir,
            ext="geojson",
        )
        # Aggregate total conflict events
        column = "total_conflict_count"
        event_count = self._aggregate_data(
            ucdp, agg_col=column, agg_func="count"
        )
        event_count = event_count.rename(columns={column: f"ucdp_{column}"})
        event_count = data_utils._merge_data(
            [admin, event_count],
            columns=[f"{self.adm_level}_ID"],
            how="left",
        )
        fatalities_count = self._aggregate_data(
            ucdp, agg_col="best", agg_func="sum"
        )
        fatalities_count = fatalities_count.rename(
            columns={"best": "ucdp_total_fatalities"}
        )
        fatalities_count = data_utils._merge_data(
            [admin, fatalities_count],
            columns=[f"{self.adm_level}_ID"],
            how="left",
        )
        ucdp = data_utils._merge_data(
            [event_count, fatalities_count, ucdp_agg],
            columns=self.merge_columns,
        )
        self._calculate_conflict_stats(ucdp, source="ucdp")
        ucdp.to_file(final_exposure_vector)

        return ucdp

    def download_acled(
        self,
        population: str = "full",
        aggregate: bool = False,
        exposure_column: str = "population_best",
    ) -> gpd.GeoDataFrame:

        # Build file paths
        acled_dict = dict()
        raw_file = self._build_filename(
            self.iso_code, self.acled_name, self.local_dir, ext="geojson"
        )
        acled_agg_file = self._build_filename(
            self.iso_code,
            f"{self.acled_name}_{self.adm_level}",
            self.local_dir,
            ext="geojson",
        )

        # Download ACLED data
        if self.overwrite or not os.path.exists(raw_file):
            logging.info(f"Downloading ACLED data for {self.iso_code}...")

            params = dict(
                country=self.country,
                event_date=f"{self.conflict_start_date}|{self.conflict_end_date}",
                event_date_where="BETWEEN",
                population=population,
                page=1,
            )
            headers = {"Authorization": f"Bearer {self.acled_key}"}

            # Paginate through ACLED API results
            acled_url = self.config["urls"]["acled_url"]
            len_subdata = -1
            data = []
            while len_subdata != 0:
                try:
                    logging.info(f"Reading ACLED page {params['page']}...")
                    response = requests.get(
                        acled_url, headers=headers, params=params
                    )
                    if response.status_code != 200:
                        logging.info(
                            f"{WARNING}WARNING: ACLED data failed to download.{RESET}"
                        )
                        logging.info(
                            f"{WARNING}ACLED Response Code: {response.status_code}{RESET}"
                        )
                        return

                    subdata = pd.DataFrame(response.json()["data"])
                    data.append(subdata)
                    len_subdata = len(subdata)
                    params["page"] = params["page"] + 1

                except Exception as e:
                    logging.info(
                        f"{WARNING}WARNING: ACLED data failed to download.{RESET}"
                    )
                    logging.info(f"{WARNING}{e}{RESET}")
                    return

            # Concatenate all pages
            data = pd.concat(data)
            if len(data) == 0:
                logging.info(
                    f"{WARNING}WARNING: No ACLED data returned for {self.country}.{RESET}"
                )
                return

            # Convert to GeoDataFrame
            data = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(
                    data["longitude"], data["latitude"], crs=self.crs
                ),
                data=data,
            )

            # Clean and standardize columns
            if exposure_column in data.columns:
                data[exposure_column] = (
                    data[exposure_column]
                    .replace({"": np.nan})
                    .astype(np.float64)
                )

            # Spatial join ACLED events with admin boundaries
            data = data.sjoin(
                self.geoboundary, how="left", predicate="intersects"
            )
            data = data.drop(["index_right"], axis=1)

            data.to_file(raw_file)
            logging.info(f"ACLED file saved to {raw_file}.")

        # Read ACLED data from file
        self.acled_raw = gpd.read_file(raw_file).to_crs(self.crs)

        if aggregate and os.path.exists(acled_agg_file):
            return gpd.read_file(acled_agg_file)

        full_data = []
        for asset_name, asset_file in (
            pbar := tqdm(
                zip(self.asset_names, self.asset_files),
                total=len(self.asset_names),
            )
        ):
            pbar.set_description(f"Processing {asset_name}")

            filtered_file = self._build_filename(
                self.iso_code,
                f"{self.acled_name}_{asset_name}_FILTERED",
                self.local_dir,
                ext="geojson",
            )
            acled = self._filter_acled(
                self.acled_raw, self.acled_hierarchy, filtered_file
            )

            if aggregate:
                agg_file = self._build_filename(
                    self.iso_code,
                    f"{self.acled_name}_{asset_name}_{self.adm_level}",
                    self.local_dir,
                    ext="geojson",
                )
                acled = self._aggregate_acled(
                    acled_file=filtered_file,
                    agg_file=agg_file,
                    asset_name=asset_name,
                    asset_file=asset_file,
                )
                full_data.append(acled)

            else:
                acled = gpd.read_file(filtered_file)
                acled_dict[asset_name] = acled

        if len(full_data) > 0:
            if self.overwrite or not os.path.exists(acled_agg_file):
                acled = data_utils._merge_data(
                    full_data, columns=self.merge_columns
                )
                acled.to_file(acled_agg_file)
            acled = gpd.read_file(acled_agg_file)
            return acled

        return acled_dict

    def _filter_acled(
        self, data: pd.DataFrame, hierarchy: dict = None, out_file: str = None
    ) -> pd.DataFrame:
        valid_rows = []

        if self.overwrite or not os.path.exists(out_file):
            if hierarchy is None:
                hierarchy = self.acled_hierarchy

            # Loop through hierarchy structure
            for disorder_type, event_dict in hierarchy.items():
                for event_type, sub_events in event_dict.items():
                    for sub_event in sub_events:
                        valid_rows.append(
                            (disorder_type, event_type, sub_event)
                        )

            # Convert valid combinations to a DataFrame
            valid_df = pd.DataFrame(
                valid_rows,
                columns=["disorder_type", "event_type", "sub_event_type"],
            )

            # Inner merge to keep only valid combinations
            filtered = data.merge(
                valid_df,
                on=["disorder_type", "event_type", "sub_event_type"],
                how="inner",
            )
            filtered.to_file(out_file)

        filtered = gpd.read_file(out_file)
        return filtered

    def _aggregate_acled(
        self,
        acled_file: str,
        agg_file: str,
        asset_name: str,
        asset_file: str,
        prefix: str = "wbg",
    ):
        # Read the ACLED raw data if it exists
        if not os.path.exists(acled_file):
            raise FileNotFoundError(f"ACLED file not found: {acled_file}")
        acled = gpd.read_file(acled_file)

        # Aggregate ACLED events if the aggregated file does not exist
        if self.overwrite or not os.path.exists(agg_file):
            agg = self._aggregate_acled_exposure(acled, asset_name)

            full_data = [agg]
            exposure_raster = self._build_filename(
                self.iso_code,
                f"{self.acled_name}_{asset_name}_exposure",
                self.local_dir,
                ext="tif",
            )
            exposure_vector = self._build_filename(
                self.iso_code,
                f"{self.acled_name}_{asset_name}_exposure_{self.adm_level}",
                self.local_dir,
                ext="geojson",
            )

            column = f"{self.acled_name.lower()}_{asset_name}_exposure"
            if self.overwrite or not os.path.exists(exposure_vector):
                acled_tif = self._calculate_custom_conflict_exposure(
                    acled_file,
                    asset_file,
                    asset_name=asset_name,
                    conflict_src="acled",
                )
                out_tif, _ = self._calculate_exposure(
                    asset_file, acled_tif, exposure_raster, threshold=1
                )
                self._calculate_zonal_stats(
                    out_tif,
                    column=column,
                    prefix=prefix,
                    stats_agg=["sum"],
                    out_file=exposure_vector,
                )

            # Read exposure vector and clean zero values
            exposure_var = prefix + "_" + column
            exposure = gpd.read_file(exposure_vector)
            exposure.loc[exposure[exposure_var] == 0, exposure_var] = None
            full_data.append(exposure)

            # Merge aggregated ACLED and exposure data
            acled = data_utils._merge_data(
                full_data, columns=self.merge_columns
            )
            acled.to_file(agg_file)

        acled = gpd.read_file(agg_file)
        return acled

    def _calculate_custom_conflict_exposure(
        self,
        conflict_file: str,
        asset_file: str,
        asset_name: str,
        conflict_src: str,
        temp_name: str = "temp",
        buffer_size: int = 3000,
        meter_crs: str = "EPSG:3857",
    ) -> str:
        # Check that the conflict file exists
        if not os.path.exists(conflict_file):
            raise FileNotFoundError(
                f"conflict file not found: {conflict_file}"
            )

        # Helper function to determine buffer size based on event type and fatalities
        def get_buffer_size(event, fatality):
            if event != "Strategic developments":
                if (event == "Riots") | (
                    (event == "Violence against civilians") & (fatality == 0)
                ):
                    return 2000
                return 5000
            return 0

        # Create temporary buffered GeoJSON filename
        filename = (
            os.path.basename(conflict_file).split(".")[0]
            + f"_{temp_name.upper()}.geojson"
        )
        temp_file = os.path.join(self.local_dir, filename)

        # Create temporary raster file for buffered data
        if self.overwrite or not os.path.exists(temp_file):
            data = gpd.read_file(conflict_file)
            data["values"] = 1

            # Get buffer size depending on conflict data source
            if conflict_src == "acled":
                data["buffer_size"] = data.apply(
                    lambda x: get_buffer_size(x.event_type, x.fatalities),
                    axis=1,
                )
            elif conflict_src == "ucdp":
                data["buffer_size"] = buffer_size

            # Apply buffer using meter CRS
            data["geometry"] = data.to_crs(meter_crs).apply(
                lambda x: x.geometry.buffer(x.buffer_size), axis=1
            )
            data = data.set_crs(meter_crs, allow_override=True).to_crs(
                self.crs
            )
            data.to_file(temp_file)

        # Define output raster path
        out_file = os.path.join(
            self.local_dir,
            conflict_file.replace(".geojson", f"_{asset_name.upper()}.tif"),
        )

        # Rasterize if raster does not exist
        if self.overwrite or not os.path.exists(out_file):
            try:
                # Create empty raster based on asset template
                with rio.open(asset_file) as src:
                    out_image = src.read(1)
                    out_image = np.zeros(out_image.shape)

                    out_meta = src.meta.copy()
                    with rio.open(out_file, "w", **out_meta) as dest:
                        dest.write(out_image, 1)

                os.system(f"gdal_rasterize -at -burn 1 {temp_file} {out_file}")

            except Exception as e:
                raise RuntimeError(f"Error creating exposure raster: {e}")

        return out_file

    def _aggregate_acled_exposure(
        self, acled: gpd.GeoDataFrame, asset: str
    ) -> gpd.GeoDataFrame:

        # Helper function to sum while ignoring all-NaN arrays
        def _nansumwrapper(a, **kwargs):
            if np.isnan(a).all():
                return np.nan
            else:
                return np.nansum(a, **kwargs)

        admin = self.geoboundary

        # Aggregate population sum
        pop_sum = self._aggregate_data(
            acled,
            agg_col="population_best",
            agg_func=lambda x: _nansumwrapper(x),
        )
        pop_sum = pop_sum.rename(
            columns={"population_best": f"acled_{asset}_population_best"}
        )

        # Aggregate total conflict events
        event_count = self._aggregate_data(
            acled, agg_col="conflict_count", agg_func="count"
        )
        event_count = event_count.rename(
            columns={"conflict_count": f"acled_{asset}_conflict_count"}
        )

        # Aggregate total conflict events
        fatalities_count = self._aggregate_data(
            acled, agg_col="fatalities", agg_func="sum"
        )
        fatalities_count = fatalities_count.rename(
            columns={"fatalities": f"acled_{asset}_fatalities"}
        )

        # Aggregate conflict events where population_best is missing
        null_pop_event_count = self._aggregate_data(
            acled[acled["population_best"].isna()],
            agg_col="null_conflict_count",
            agg_func="count",
        )
        null_pop_event_count = null_pop_event_count.rename(
            columns={
                "null_conflict_count": f"acled_{asset}_null_conflict_count"
            }
        )

        # Merge all aggregated data with admin boundaries
        acled = data_utils._merge_data(
            [
                admin,
                pop_sum,
                event_count,
                fatalities_count,
                null_pop_event_count,
            ],
            columns=[f"{self.adm_level}_ID"],
            how="left",
        )

        # Calculate population-weighted conflict exposure
        col_base = f"acled_{asset}"
        exposure_var = f"{col_base}_exposure"

        denominator = acled[f"{col_base}_conflict_count"] - acled[
            f"{col_base}_null_conflict_count"
        ].fillna(0)
        acled[exposure_var] = (
            acled[f"{col_base}_population_best"] / denominator
        )
        acled.loc[acled[exposure_var] == 0, exposure_var] = None

        acled = self._calculate_conflict_stats(
            acled, source="acled", asset=asset
        )
        return acled

    def _calculate_conflict_stats(self, data, source, asset: str = "total"):
        col_base = f"{source}_{asset}"
        per_conflict = f"{col_base}_fatalities_per_conflict"

        data[per_conflict] = (
            data[f"{col_base}_fatalities"] / data[f"{col_base}_conflict_count"]
        ).replace([np.inf, -np.inf], np.nan)

        return data

    def _download_url_progress(self, url, output_path):
        desc = os.path.basename(output_path)
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=desc
        ) as t:
            urllib.request.urlretrieve(
                url, filename=output_path, reporthook=t.update_to
            )

    def download_url(
        self, dataset: str, dataset_name: str = None, ext: str = "tif"
    ) -> str:
        if dataset_name is None:
            dataset_name = dataset.replace(f"{self.global_name.lower()}_", "")

        global_file = self._build_filename(
            self.global_name, dataset_name, self.global_dir, ext=ext
        )

        url_name = f"{dataset}_url"
        if url_name in self.config["urls"]:
            if dataset == "gadm":
                url = self.config["urls"][url_name].format(
                    self.iso_code, self.adm_level[-1]
                )
            elif "wildfire" in url_name:
                date_today = datetime.date.today().strftime("%Y-%m-%d")
                url = self.config["urls"][url_name].format(date_today)
            else:
                url = self.config["urls"][url_name].format(
                    self.iso_code, self.iso_code.lower()
                )

        # Check if the dataset is global
        if self.global_name.lower() in dataset:
            # Download if not already present
            if self.overwrite or not os.path.exists(global_file):
                logging.info(f"Downloading {url}...")
                if url.endswith(".zip"):
                    self.download_zip(
                        url, dataset, out_file=global_file, ext=ext
                    )
                elif url.endswith(".tif") or (".tif" in url):
                    self._download_url_progress(url, global_file)

            # Clip raster to country boundary if applicable
            local_file = self._build_filename(
                self.iso_code, dataset_name, self.local_dir, ext=ext
            )
            if ext == "tif":
                nodata = self.config.get("nodata", {}).get(dataset, [])
                admin = self.geoboundary.dissolve(by="iso_code")
                self._clip_raster(global_file, local_file, admin, nodata)

        else:
            # For non-global datasets, just download locally
            local_file = self._build_filename(
                self.iso_code, dataset_name, self.local_dir, ext
            )
            if self.overwrite or not os.path.exists(local_file):
                if url.endswith(".zip"):
                    self.download_zip(
                        url, dataset, out_file=local_file, ext=ext
                    )
                elif url.endswith(".tif") or (".tif" in url):
                    self._download_url_progress(url, local_file)

        return local_file

    def download_zip(
        self, url: str, dataset: str, out_file: str, ext: str = "tif"
    ) -> None:
        # Decide output directory (global vs local)
        out_dir = self.global_dir if "global" in dataset else self.local_dir
        zip_file = os.path.join(out_dir, f"{dataset}.zip")
        zip_dir = os.path.join(out_dir, dataset)

        # Download and extract ZIP if not already done
        if not os.path.exists(zip_file) and not os.path.exists(zip_dir):
            # urllib.request.urlretrieve(url, zip_file)
            self._download_url_progress(url, zip_file)
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(zip_dir)
            os.remove(zip_file)

        if ext == "tif":
            # Look for GeoTIFF files
            tif_files = [
                file for file in os.listdir(zip_dir) if file.endswith(".tif")
            ]

            if len(tif_files) == 0:
                # If no .tif, convert .grd file to GeoTIFF using GDAL
                grd_files = [
                    file
                    for file in os.listdir(zip_dir)
                    if file.endswith(".grd")
                ]
                if len(grd_files) > 0:
                    grd_file = grd_files[0]
                    tif_file = os.path.join(
                        zip_dir, grd_file.replace(".grd", ".tif")
                    )
                    os.system(
                        f"gdal_translate -a_srs EPSG:4326 {os.path.join(zip_dir, grd_file)} {tif_file}"
                    )
            else:
                tif_file = tif_files[0]

            os.system(
                f"gdal_translate -a_srs EPSG:4326 {os.path.join(zip_dir, tif_file)} {out_file}"
            )
            shutil.rmtree(zip_dir)

        elif ext == "geojson":
            # Look for GeoJSON files
            geojson_files = [
                file
                for file in os.listdir(zip_dir)
                if file.endswith(".geojson")
            ]

            # If no .geojson, convert from .json feature collection
            if len(geojson_files) == 0:
                json_file = [
                    file
                    for file in os.listdir(zip_dir)
                    if file.endswith(".json")
                ][0]
                json_file = os.path.join(zip_dir, json_file)
                with open(json_file, encoding="utf-8") as data:
                    features = json.load(data)["features"]

                geojson = gpd.GeoDataFrame.from_features(features)
                geojson = geojson.set_crs(self.crs)
                geojson.to_file(out_file)
                logging.info(f"Geojson file saved to {out_file}.")

            # Get geojson file if it exists
            else:
                geojson_file = geojson_files[0]
                shutil.copyfile(os.path.join(zip_dir, geojson_file), out_file)

            shutil.rmtree(zip_dir)

        elif ext == "csv":
            csv_files = [
                file for file in os.listdir(zip_dir) if file.endswith(".csv")
            ]
            shutil.copyfile(os.path.join(zip_dir, csv_files[0]), out_file)

    def download_fathom(self, name: str, ext: str = "tif") -> str:
        fathom_dir = os.path.join(self.local_dir, self.fathom_name)

        # If processed dataset doesn't exist, generate it
        name = f"{self.iso_code}_{name}_rp{self.fathom_rp}".upper()
        raw_file = os.path.join(fathom_dir, f"{name}.{ext}")
        local_file = os.path.join(self.local_dir, f"{name}.{ext}")

        # If processed file doesn't exist, build from VRT
        if self.overwrite or not os.path.exists(local_file):
            flood_dir = os.path.join(
                fathom_dir,
                name.replace("_" + self.fathom_name, "").upper(),
                str(self.fathom_year),
                f"1in{self.fathom_rp}",
            )
            merged_file = os.path.join(fathom_dir, f"{name}.vrt")
            self._merge_tifs(f"{flood_dir}/*.{ext}", merged_file, raw_file)

            # Clip raster to admin boundary
            admin = self.geoboundary.dissolve(by="iso_code")
            nodata = self.config["nodata"][name.lower()]
            self._clip_raster(raw_file, local_file, admin, nodata)

        return local_file

    def download_jrc(self, name: str):
        out_dir = os.path.join(self.global_dir, name)
        os.makedirs(out_dir, exist_ok=True)

        url_name = f"{name}_url"
        url = self.config["urls"][url_name].format(self.jrc_rp)
        r = requests.get(url)
        data = bs4.BeautifulSoup(r.text, "html.parser")

        logging.info(f"Downloading global flood data from {url}")
        links = [
            link["href"]
            for link in data.find_all("a")
            if "depth.tif" in link["href"]
        ]
        for link in tqdm(links, total=len(links), dynamic_ncols=True):
            out_file = os.path.join(out_dir, link)
            if not os.path.exists(out_file):
                self._download_url_progress(url + link, out_file)

        vrt_file = os.path.join(self.global_dir, f"{name.upper()}.vrt")
        global_file = os.path.join(self.global_dir, f"{name.upper()}.tif")

        if not os.path.exists(global_file):
            logging.info(
                "Generating flood map. Hang tight, this might take a while..."
            )
            self._merge_tifs(f"{out_dir}/*.tif", vrt_file, global_file)
            logging.info(f"Flood map saved to {global_file}.")

        local_file = self._build_filename(
            self.iso_code,
            name.replace("global_", ""),
            self.local_dir,
            ext="tif",
        )
        if not os.path.exists(local_file):
            admin = self.geoboundary.dissolve(by="iso_code")
            self._clip_raster(global_file, local_file, admin)

        return local_file

    def download_worldcover(
        self,
        land_cover_class: str,
        name: str = "worldcover",
        year: int = 2021,
        resample: bool = True,
        overwrite: bool = False,
    ):
        admin = self.geoboundary.dissolve(by="iso_code").to_crs(self.crs)

        # Load worldcover grid
        worldcover_url = self.config["urls"][f"{name}_url"]
        url = f"{worldcover_url}/esa_worldcover_grid.geojson"
        grid = gpd.read_file(url).to_crs(self.crs)

        # Get grid tiles intersecting AOI
        tiles = gpd.overlay(grid, admin, how="intersection")

        # Map Code source: https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf
        map_code = self.config["worldcover_map_code"]

        # Select version tag, based on the year
        version = {2020: "v100", 2021: "v200"}[year]

        local_file = self._build_filename(
            self.iso_code,
            f"{name.upper()}_{land_cover_class.upper()}",
            self.local_dir,
            ext="tif",
        )

        if not os.path.exists(local_file):
            worldcover_file = self._build_filename(
                self.iso_code,
                name.upper(),
                self.local_dir,
                ext="tif",
            )

            if not os.path.exists(worldcover_file):
                out_dir = os.path.join(self.local_dir, name)
                os.makedirs(out_dir, exist_ok=True)

                # Download TIF files
                for tile in tqdm(tiles.ll_tile):
                    raw_name = (
                        f"ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
                    )
                    filename = os.path.join(out_dir, raw_name)
                    if not os.path.exists(filename):
                        url = (
                            f"{worldcover_url}/{version}/{year}/map/{raw_name}"
                        )
                        r = requests.get(url, allow_redirects=True)
                        with open(filename, "wb") as f:
                            f.write(r.content)

                logging.info(
                    f"Generating worldcover map for {self.country}. Hang tight, this might take a while..."
                )
                vrt_file = os.path.join(
                    self.local_dir, f"{self.iso_code}_{name.upper()}.vrt"
                )
                temp_file = os.path.join(
                    self.local_dir, f"{self.iso_code}_{name.upper()}_TEMP.tif"
                )
                self._merge_tifs(f"{out_dir}/*.tif", vrt_file, temp_file)

                if resample:
                    resampled_file = os.path.join(
                        self.local_dir,
                        f"{self.iso_code}_{name.upper()}_RESAMPLED.tif",
                    )
                    worldpop_file = self.download_url("worldpop", ext="tif")
                    temp_file = self._resample_raster(
                        worldpop_file, temp_file, resampled_file
                    )

                admin = self.geoboundary.dissolve(by="iso_code")
                self._clip_raster(temp_file, worldcover_file, admin)

            code = map_code[land_cover_class]
            local_file = self._mask_raster_by_code(
                worldcover_file, local_file, code
            )

        return local_file

    def _mask_raster_by_code(self, raster_file: str, out_file: str, code: int):
        # Ensure both hazard and asset rasters exist
        if not os.path.exists(raster_file):
            raise FileNotFoundError(f"Raster file not found: {raster_file}")

        # Open both asset and hazard rasters
        with rio.open(raster_file, "r") as src:
            # Asset raster values
            raster = src.read(1)
            raster[raster != code] = 0
            raster[raster == code] = 1

            out_meta = src.meta.copy()
            out_meta.update(count=1, dtype="int16")

        # Save exposure raster
        with rio.open(out_file, "w", **out_meta) as dst:
            dst.write(raster.astype(out_meta["dtype"]), 1)

        return out_file

    def download_assets(self, name: str = "asset") -> gpd.GeoDataFrame:
        datasets = self.config[f"{name}_data"]

        # Output file path for merged hazard data
        full_data_file = self._build_filename(
            self.iso_code,
            f"{name}_{self.adm_level}",
            self.local_dir,
            ext="geojson",
        )

        # If overwrite is set or file does not exist, regenerate it
        if self.overwrite or not os.path.exists(full_data_file):
            full_data = None

            for index, dataset in enumerate(datasets):
                logging.info(
                    f"({index+1}/{len(datasets)}) Processing {dataset}..."
                )

                # Download raster dataset (GeoTIFF)
                if "worldcover" in dataset:
                    land_cover_class = dataset.split("_")[-1]
                    local_file = self.download_worldcover(
                        land_cover_class=land_cover_class,
                        resample=self.resample_worldcover,
                    )
                else:
                    local_file = self.download_url(dataset, ext="tif")

                dataset_name = dataset.replace("global_", "")

                # Zonal statistics for base hazard raster
                stats_agg = ["sum"]
                data = self._calculate_zonal_stats(
                    local_file,
                    column=dataset_name,
                    stats_agg=stats_agg,
                )

                # Merge into cumulative dataset
                if full_data is None:
                    full_data = data.copy()
                elif not set(data.columns).issubset(set(full_data.columns)):
                    full_data = data_utils._merge_data(
                        [full_data, data], columns=self.merge_columns
                    )

            # Save merged hazard dataset
            full_data.to_file(full_data_file)
            logging.info(f"Data saved to {full_data_file}.")

        # Always load and return data in correct CRS
        full_data = gpd.read_file(full_data_file).to_crs(self.crs)

        return full_data

    def download_hazards(self, name: str = "hazard") -> gpd.GeoDataFrame:
        datasets = self.config[f"{name}_data"]

        # Output file path for merged hazard data
        full_data_file = self._build_filename(
            self.iso_code,
            f"{name}_{self.adm_level}",
            self.local_dir,
            ext="geojson",
        )

        # If overwrite is set or file does not exist, regenerate it
        if self.overwrite or not os.path.exists(full_data_file):
            full_data = None

            for index, dataset in enumerate(datasets):
                logging.info(
                    f"({index+1}/{len(datasets)}) Processing {dataset}..."
                )

                # Download raster dataset (GeoTIFF)
                if "fathom" in dataset:
                    local_file = self.download_fathom(dataset)
                elif "jrc2" in dataset:
                    local_file = self.download_jrc(dataset)
                else:
                    local_file = self.download_url(dataset, ext="tif")

                dataset_name = dataset.replace("global_", "")
                for asset_name, asset_file in (
                    pbar := tqdm(
                        zip(self.asset_names, self.asset_files),
                        total=len(self.asset_names),
                    )
                ):
                    pbar.set_description(f"Processing {asset_name}")

                    exposure_file = self._build_filename(
                        self.iso_code,
                        f"{dataset_name}_{asset_name}_exposure",
                        self.local_dir,
                        ext="tif",
                    )
                    weighted_exposure_file = self._build_filename(
                        self.iso_code,
                        f"{dataset_name}_{asset_name}_intensity_weighted_exposure",
                        self.local_dir,
                        ext="tif",
                    )

                    # Generate exposure rasters (skip asset rasters)
                    self._generate_exposure(
                        asset_name,
                        asset_file,
                        local_file,
                        exposure_file,
                        self.config["threshold"][dataset],
                    )

                    # Zonal statistics for base hazard raster
                    stats_agg = ["mean"]
                    data = self._calculate_zonal_stats(
                        local_file,
                        column=dataset_name,
                        stats_agg=stats_agg,
                    )

                    # Merge into cumulative dataset
                    if full_data is None:
                        full_data = data.copy()
                    elif not set(data.columns).issubset(
                        set(full_data.columns)
                    ):
                        full_data = data_utils._merge_data(
                            [full_data, data], columns=self.merge_columns
                        )

                    exposure = self._calculate_zonal_stats(
                        exposure_file,
                        column=dataset_name,
                        suffix=f"{asset_name}_exposure",
                    )
                    weighted_exposure = self._calculate_zonal_stats(
                        weighted_exposure_file,
                        column=dataset_name,
                        suffix=f"{asset_name}_intensity_weighted_exposure",
                    )
                    full_data = data_utils._merge_data(
                        [full_data, exposure, weighted_exposure],
                        columns=self.merge_columns,
                    )

            # Save merged hazard dataset
            full_data.to_file(full_data_file)
            logging.info(f"Data saved to {full_data_file}.")

        # Always load and return data in correct CRS
        full_data = gpd.read_file(full_data_file).to_crs(self.crs)

        return full_data

    def _generate_exposure(
        self,
        asset: str,
        asset_file: str,
        local_file: str,
        exposure_file: str,
        threshold: float,
    ) -> None:

        # Ensure the input raster exists before proceeding
        if not os.path.exists(local_file):
            raise FileNotFoundError(
                f"Input raster file not found: {local_file}"
            )

        # Only generate exposure if it hasn't already been computed
        if self.overwrite or not os.path.exists(exposure_file):
            resampled_file = local_file.replace(
                ".tif", f"_{asset.upper()}_RESAMPLED.tif"
            )

            # Resample raster if resampled version does not already exist
            if self.overwrite or not os.path.exists(resampled_file):
                try:
                    self._resample_raster(
                        asset_file, local_file, resampled_file
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to resample raster: {e}")

            try:
                self._calculate_exposure(
                    asset_file, resampled_file, exposure_file, threshold
                )
            except Exception as e:
                raise RuntimeError(f"Failed to calculate exposure: {e}")

    def _resample_raster(
        self, asset_file: str, in_file: str, out_file: str
    ) -> str:
        # Check that both the input file and the reference asset file exist
        if not os.path.exists(in_file):
            raise FileNotFoundError(f"Input raster file not found: {in_file}")

        if not os.path.exists(asset_file):
            raise FileNotFoundError(f"Asset file not found: {asset_file}")

        # Open the asset raster (reference for resolution and bounds)
        asset = gdal.Open(asset_file, 0)
        if asset is None:
            raise RuntimeError(f"Failed to open asset raster: {asset_file}")

        # Extract geotransform info: resolution and bounding box
        geoTransform = asset.GetGeoTransform()
        x_res = geoTransform[1]
        y_res = -geoTransform[5]

        minx = geoTransform[0]
        maxy = geoTransform[3]
        maxx = minx + geoTransform[1] * (asset.RasterXSize - 1)
        miny = maxy + geoTransform[5] * (asset.RasterYSize - 1)
        out_bounds = [minx, miny, maxx, maxy]

        # Set up warp parameters for resampling
        kwargs = {
            "format": "GTiff",
            "xRes": x_res,
            "yRes": y_res,
            "targetAlignedPixels": True,
            "outputBounds": out_bounds,
        }

        # Perform the warp/resampling
        ds = gdal.Warp(out_file, in_file, **kwargs)
        if ds is None:
            raise RuntimeError(f"GDAL Warp failed for input: {in_file}")

        return out_file

    def _calculate_exposure(
        self,
        asset_file: str,
        hazard_file: str,
        exposure_file: str,
        threshold: float,
    ) -> tuple[str, str]:

        # Ensure both hazard and asset rasters exist
        if not os.path.exists(hazard_file):
            raise FileNotFoundError(
                f"Hazard raster file not found: {hazard_file}"
            )

        if not os.path.exists(asset_file):
            raise FileNotFoundError(
                f"Asset raster file not found: {asset_file}"
            )

        # Open both asset and hazard rasters
        with (
            rio.open(asset_file, "r") as src1,
            rio.open(hazard_file, "r") as src2,
        ):
            # Asset raster values
            asset = src1.read(1)
            asset[asset < 0] = 0

            # Hazard raster values
            hazard = src2.read(1)
            if "drought" not in hazard_file.lower():
                hazard[hazard < 0] = 0

            if "heat_stress" in hazard_file.lower():
                hazard = hazard / 100

            # Scale hazard values to [0, 1] for weighting
            asset_binary = asset.copy()
            asset_binary[asset_binary > 0] = 1
            asset_binary, hazard = data_utils.match_shape(asset_binary, hazard)
            hazard_scaled = data_utils._minmax_scale(hazard * asset_binary)

            # Binary raster: hazard above threshold = 1, else 0
            if "drought" in hazard_file.lower():
                binary = (hazard < threshold).astype(int)
            else:
                binary = (hazard >= threshold).astype(int)

            # Exposure: asset presence masked by hazard exceedance
            exposure = asset * binary

            # Weighted exposure: exposure scaled by hazard intensity
            weighted_exposure = exposure * hazard_scaled

            # Copy metadata from asset raster to preserve georeferencing
            out_meta = src1.meta.copy()

            out_meta.update({"dtype": rio.float32, "driver": "GTiff"})

        # Save binary exposure raster
        binary_file = exposure_file.replace("EXPOSURE", "BINARY")
        with rio.open(binary_file, "w", **out_meta) as dst:
            dst.write(binary, 1)

        # Save intensity-weighted exposure raster
        weighted_exposure_file = exposure_file.replace(
            "EXPOSURE", "INTENSITY_WEIGHTED_EXPOSURE"
        )
        with rio.open(weighted_exposure_file, "w", **out_meta) as dst:
            dst.write(weighted_exposure, 1)

        # Save exposure raster
        with rio.open(exposure_file, "w", **out_meta) as dst:
            dst.write(exposure, 1)

        return exposure_file, weighted_exposure_file

    def _aggregate_data(
        self,
        data: gpd.GeoDataFrame,
        agg_col: str = None,
        agg_func: str = "sum",
        adm_level: str = None,
    ) -> gpd.GeoDataFrame:

        # Define administrative ID column
        if adm_level is None:
            adm_level = self.adm_level

        agg_name = f"{adm_level}_ID"
        if agg_name not in data.columns:
            agg_name = adm_level

        # Perform aggregation
        if agg_func == "count":
            # Count number of rows per admin unit
            agg = data.groupby([agg_name], dropna=False).size().reset_index()
        else:
            # Apply specified aggregation function to agg_col
            data = data.copy()
            data[agg_col] = data[agg_col].astype(float)
            agg = (
                data.groupby([agg_name], dropna=False)
                .agg({agg_col: agg_func})
                .reset_index()
            )

        # Rename columns to standard format
        agg.columns = [agg_name, agg_col]

        return agg

    def _merge_tifs(self, in_files, vrt_file, tif_file):
        os.system(f"gdalbuildvrt {vrt_file} {in_files}")
        os.system(
            f"gdal_translate -co TILED=YES -co COMPRESS=LZW -co BIGTIFF=YES -co NUM_THREADS=ALL_CPUS --config GDAL_CACHEMAX 512 {vrt_file} {tif_file}"
        )

    def _clip_raster(
        self,
        global_tif: str,
        local_tif: str,
        admin: gpd.GeoDataFrame,
        nodata: list = [],
    ) -> rio.io.DatasetReader:

        # Ensure the input raster exists
        if not os.path.exists(global_tif):
            raise FileNotFoundError(
                f"{WARNING}Global raster not found: {global_tif}{RESET}"
            )

        # Return existing raster if the clipped file already exists
        if not os.path.exists(local_tif):
            with rio.open(global_tif) as src:
                if src.nodata is not None:
                    nodata = [src.nodata] + nodata

                # Reproject the admin boundaries if CRS differs
                if src.crs != admin.crs:
                    admin = admin.to_crs(src.crs)

                # Extract the country boundary geometry for clipping
                shape = [admin.iloc[0]["geometry"]]

                # Perform raster clipping using rasterio.mask
                out_image, out_transform = rio.mask.mask(
                    src, shape, crop=True, all_touched=True
                )

                out_meta = src.meta.copy()
                dtype = out_meta["dtype"]
                for val in nodata:
                    out_image[out_image == val] = 0

                # Update raster metadata to reflect changes

                out_meta.update(
                    {
                        "dtype": dtype,
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "nodata": 0,
                    }
                )

            # Save the clipped raster to the specified output path
            with rio.open(local_tif, "w", **out_meta) as dest:
                dest.write(out_image)

        # Return the clipped raster
        return rio.open(local_tif)

    def _calculate_zonal_stats(
        self,
        in_file: str,
        column: str,
        out_file: str = None,
        stats_agg: list = ["sum"],
        add_stats: list = None,
        suffix: str = None,
        prefix: str = None,
    ) -> gpd.GeoDataFrame:
        # Extract base name from raster file
        name = os.path.basename(in_file).split(".")[0]

        # Generate default output path if not provided
        if out_file is None:
            out_file = os.path.join(
                self.local_dir, f"{name}_{self.adm_level}.geojson"
            )

        if self.overwrite or not os.path.exists(out_file):
            admin_file = self.admin_file
            admin = self.geoboundary
            original_crs = admin.crs

            # Reproject admin boundaries if CRS does not match raster
            with rio.open(in_file) as src:
                if admin.crs != src.crs:
                    admin = admin.to_crs(src.crs)
                    admin.to_file(admin_file)
                    logging.info(f"Admin file saved to {admin_file}.")

            # Compute zonal statistics
            stats = rasterstats.zonal_stats(
                admin_file,
                in_file,
                stats=stats_agg,
                all_touched=True,
                add_stats=add_stats,
            )
            stats = pd.DataFrame(stats)
            if "custom" in stats:
                stats = stats["custom"].astype(float)

            # Reproject admin back to original CRS
            if admin.crs != original_crs:
                admin = admin.to_crs(original_crs)
                admin.to_file(admin_file)
                logging.info(f"Admin file saved to {admin_file}.")

            # Load admin boundaries and add zonal statistics column
            data = gpd.read_file(admin_file)
            column_name = column.lower()
            if suffix is not None:
                column_name = f"{column.lower()}_{suffix}"
            if prefix is not None:
                column_name = f"{prefix}_{column.lower()}"

            data[column_name] = stats
            data[column_name] = data[column_name].astype("float64")

            # Save results to GeoJSON
            data.to_file(out_file)

        return gpd.read_file(out_file)

    def _build_filename(self, prefix, suffix, out_dir, ext="geojson") -> str:
        # Construct and return the full file path
        return os.path.join(
            out_dir, f"{prefix.upper()}_{suffix.upper()}.{ext}"
        )
