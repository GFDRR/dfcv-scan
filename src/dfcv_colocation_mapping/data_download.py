import os
import re
import json
import shutil
import zipfile
import requests
import logging
import warnings
import itertools
import urllib.request
import importlib_resources
from functools import reduce

import datetime
from pathlib import Path
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np

import geojson
from osgeo import gdal, gdalconst
import geopandas as gpd
import rasterio as rio
import rasterio.mask
import rasterstats
import pycountry

import ahpy
import bs4
import osmnx as ox
from dtmapi import DTMApi

from scipy.stats.mstats import gmean
from dfcv_colocation_mapping import data_utils

from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

ox.settings.max_query_area_size = 500000000000000
logging.basicConfig(level=logging.INFO, force=True)
io_logger = logging.getLogger("pyogrio._io")
io_logger.setLevel(logging.WARNING)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class DatasetManager:
    def __init__(
        self,
        iso_code: str,
        adm_source: str = "geoboundary",
        acled_key: str = None,
        acled_limit: str = None,
        acled_exclude: str = None,
        acled_country: str = None,
        acled_name: str = "acled",
        ucdp_name: str = "ucdp",
        conflict_start_date: str = None,
        conflict_end_date: str = None,
        conflict_last_n_years: int = 10,
        dtm_key: str = None,
        dtm_adm_level: str = None,
        idmc_key: str = None,
        displacement_start_date: str = None,
        displacement_end_date: str = None,
        displacement_last_n_years: int = 10,
        jrc_rp: int = 100,
        jrc_version: str = None,
        resample_worldcover: bool = True,
        fathom_year: int = 2020,
        fathom_rp: int = 50,
        fathom_threshold: int = 50,
        fathom_name: str = "fathom",
        adm_level: str = "ADM3",
        datasets: list = None,
        data_dir: str = "data",
        meter_crs: str = "EPSG:3857",
        crs: str = "EPSG:4326",
        global_name: str = "global",
        overwrite: bool = False,
        group: str = "Region",
        mhs_aggregation: str = "power_mean",
        config_file: str = None,
        dtm_file: str = None,
        idmc_file: str = None,
        acled_file: str = None,
        osm_config_file: str = None,
        adm_config_file: str = None,
        download: bool = False,
    ):
        """Initialize a DatasetManager object for a given country.

        Args:
            iso_code (str): ISO3 country code.
            adm_source (str, optional): Administrative boundary source. Defaults to 'geoboundary'.
            acled_key (str, optional): ACLED API key. Defaults to None.
            acled_email (str, optional): ACLED account email. Defaults to None.
            conflict_start_date (str, optional): Conflict start date. This is set to 10 years ago from today if none. Defaults to None.
            conflict_end_date (str, optional): Conflict end date. This is set to the current date if none. Defaults to None.
            acled_country (str, optional): ACLED country code. Defaults to None.
            acled_name (str, optional): Label for ACLED dataset. Defaults to "acled".
            fathom_year (int, optional): Year for Fathom data. Defaults to 2020.
            fathom_rp (int, optional): Return period for Fathom data. Defaults to 50.
            fathom_threshold (int, optional): Threshold for Fathom impact. Defaults to 50.
            fathom_name (str, optional): Label for Fathom dataset. Defaults to "fathom".
            adm_level (str, optional): Administrative level to use. Defaults to "ADM3".
            datasets (list, optional): List of additional datasets to combine. Defaults to None.
            data_dir (str, optional): Base directory for storing data. Defaults to "data".
            meter_crs (str, optional): Projected CRS for distance-based calculations. Defaults to "EPSG:3857".
            crs (str, optional): Geographic CRS for the datasets. Defaults to "EPSG:4326".
            global_name (str, optional): Name of global data directory. Defaults to "global".
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
            group (str, optional): Grouping variable for administrative units. Defaults to "Region".
            mhs_aggregation (str, optional): Method to aggregate multi-hazard scores. Defaults to "power_mean".
            config_file (str, optional): Path to YAML config file. Defaults to None.
            acled_file (str, optional): Path to ACLED credentials YAML. Defaults to None.
            adm_config_file (str, optional): Path to administrative config YAML. Defaults to None.
        """

        # Store basic country and CRS information
        self.iso_code = iso_code
        self.adm_level = adm_level
        self.data_dir = data_dir
        self.config_file = config_file
        self.acled_file = acled_file
        self.meter_crs = meter_crs
        self.crs = crs
        self.overwrite = overwrite

        # Store ACLED credentials and filtering options
        self.acled_key = acled_key
        self.acled_country = acled_country

        # Store conflict start date and end date
        self.conflict_start_date = self.get_start_date(
            conflict_start_date, conflict_last_n_years
        )
        self.conflict_end_date = self.get_end_date(conflict_end_date)

        # Store IOM DTM information
        self.dtm_adm_level = self.get_dtm_adm_level(dtm_adm_level)
        self.displacement_start_date = self.get_start_date(
            displacement_start_date, displacement_last_n_years
        )
        self.displacement_end_date = self.get_end_date(displacement_end_date)

        # Store Fathom flood layer parameters
        self.fathom_year = fathom_year
        self.fathom_rp = fathom_rp
        self.fathom_threshold = fathom_threshold
        self.jrc_rp = jrc_rp
        self.jrc_version = jrc_version
        self.resample_worldcover = resample_worldcover

        # Get country name from ISO Code
        self.country = self.get_country_name()

        # Locate default configuration files if not provided
        resources = importlib_resources.files("dfcv_colocation_mapping")
        if config_file is None:
            config_file = resources.joinpath("configs", "data_config.yaml")
        if acled_file is None:
            acled_file = resources.joinpath("configs", "acled_creds.yaml")
        if dtm_file is None:
            dtm_file = resources.joinpath("configs", "dtm_creds.yaml")
        if idmc_file is None:
            idmc_file = resources.joinpath("configs", "idmc_creds.yaml")
        if adm_config_file is None:
            adm_config_file = resources.joinpath("configs", "adm_config.yaml")
        if osm_config_file is None:
            osm_config_file = resources.joinpath("configs", "osm_config.yaml")

        # Load main config
        self.config = data_utils.read_config(config_file)
        self.osm_config = data_utils.read_config(osm_config_file)

        # Load ACLED credentials from file if available
        self.acled_key = acled_key
        if os.path.exists(acled_file):
            self.acled_creds = data_utils.read_config(acled_file)
            self.acled_key = self.acled_creds["access_token"]

        self.acled_filters = self.config["acled_filters"]
        self.acled_hierarchy = self.config["acled_hierarchy"]

        # Load IOM DTM credentials from file if available
        self.dtm_key = dtm_key
        if os.path.exists(dtm_file):
            self.dtm_creds = data_utils.read_config(dtm_file)
            self.dtm_key = self.dtm_creds["dtm_key"]

        self.idmc_key = idmc_key
        if os.path.exists(idmc_file):
            self.idmc_creds = data_utils.read_config(idmc_file)
            self.idmc_key = self.idmc_creds["idmc_key"]

        # Uppercase standard labels
        self.global_name = global_name.upper()
        self.fathom_name = fathom_name.upper()
        self.acled_name = acled_name.upper()
        self.ucdp_name = ucdp_name.upper()

        # Prepare directories for storing data
        self.data_dir = os.path.join(os.getcwd(), data_dir)
        self.local_dir = os.path.join(os.getcwd(), data_dir, iso_code)
        self.global_dir = os.path.join(os.getcwd(), data_dir, self.global_name)

        # Build file path for asset layer
        self.asset_files = self.get_asset_files()

        self.adm_source = adm_source
        self.admin_file = None
        self.adm_config = data_utils.read_config(adm_config_file)

        self.mhs_aggregation = mhs_aggregation

        # Download all datasets
        if download:
            self.download_all()

    def _cascade(self, category: str, values: list[str], operation: str):
        """Cascade inclusion/exclusion down the hierarchy."""
        opposite = "exclude" if operation == "include" else "include"

        self.acled_filters[operation].setdefault(category, [])
        self.acled_filters[opposite].setdefault(category, [])

        for v in values:
            # Move from opposite → operation
            if v in self.acled_filters[opposite].get(category, []):
                self.acled_filters[opposite][category].remove(v)
                self.acled_filters[operation][category].append(v)

            # Cascade down
            if category == "disorder_type" and v in self.acled_hierarchy:
                for evt, evt_dict in self.acled_hierarchy[v][
                    "event_type"
                ].items():
                    self._cascade("event_type", [evt], operation)
                    self._cascade(
                        "sub_event_type", evt_dict["sub_event_type"], operation
                    )

            elif category == "event_type":
                for disorder, disorder_dict in self.acled_hierarchy.items():
                    if v in disorder_dict["event_type"]:
                        sub_events = disorder_dict["event_type"][v][
                            "sub_event_type"
                        ]
                        self._cascade("sub_event_type", sub_events, operation)

    def get_country_name(self):
        country = pycountry.countries.get(alpha_3=self.iso_code).name
        if self.iso_code == "COD":
            country = "Democratic Republic of Congo"
        elif self.iso_code == "COG":
            country = "Republic of Congo"
        if country is None:
            raise ValueError(f"Invalid ISO code: {self.iso_code}")
        return country

    def set_acled_filters(
        self, category: str, values: list[str], operation: str = "exclude"
    ) -> dict:
        """
        Move values between include/exclude with cascading hierarchy.
        operation: "exclude" or "include"
        """
        source = "include"
        if operation != "exclude":
            source = "exclude"

        if category not in self.acled_filters[source]:
            raise ValueError(f"{category} not in filters")

        self._cascade(category, values, operation)
        return self.acled_filters

    def download_all(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.local_dir, exist_ok=True)
        os.makedirs(self.global_dir, exist_ok=True)

        # Load geoboundaries, fallback to GADM if primary source fails
        logging.info(f"Loading {self.adm_level} geoboundaries...")
        self.geoboundary = self.download_geoboundary_with_attempts()
        self.merge_columns = list(self.geoboundary.columns)

        # Load hazard layers
        logging.info("Loading asset layers...")
        self.assets = self.download_datasets("asset")

        logging.info("Loading hazard layers...")
        self.fathom = self.download_fathom()
        self.hazards = self.download_datasets("hazard")

        logging.info("Loading conflict data...")
        logging.info(f"Conflict start date: {self.conflict_start_date}")
        logging.info(f"Conflict end date: {self.conflict_end_date}")

        # Load acled conflict data
        logging.info("Loading ACLED conflict data...")
        self.acled = self.download_acled()
        self.acled_agg = self.download_acled(aggregate=True)

        # Load ucdp conflict data
        logging.info("Loading UCDP conflict data...")
        self.ucdp = self.download_ucdp()
        self.ucdp_agg = self.download_ucdp(aggregate=True)

        logging.info("Loading displacement data...")
        logging.info(
            f"Displacement start date: {self.displacement_start_date}"
        )
        logging.info(f"Displacement end date: {self.displacement_end_date}")

        logging.info("Loading IOM DTM data...")
        self.dtm = self.download_dtm(self.dtm_adm_level, filtered=False)
        self.dtm_filtered = self.download_dtm(
            self.dtm_adm_level, filtered=True
        )
        self.dtm_agg = self.download_dtm(
            self.dtm_adm_level, filtered=True, aggregate=True
        )

        logging.info("Loading IDMC displacement data...")
        self.idmc_gidd_conflict = self.download_idmc_gidd(
            cause="conflict", filtered=False
        )
        self.idmc_gidd_disaster = self.download_idmc_gidd(
            cause="disaster", filtered=False
        )
        self.idmc_gidd_conflict_agg = self.download_idmc_gidd(
            cause="conflict", aggregate=True
        )
        self.idmc_gidd_disaster_agg = self.download_idmc_gidd(
            cause="disaster", aggregate=True
        )

        # Compute multi-hazard scores
        logging.info("Calculating scores...")
        self.data = self.combine_datasets()
        self.data = self.calculate_multihazard_score(self.data)

        # Load admin config and assign grouping
        self.data = self.assign_grouping()

        logging.info("Downloading OSM...")
        self.osm = self.download_osm()

    def get_start_date(self, start_date, last_n_years):
        if start_date is None:
            start_date = (
                datetime.date.today() - relativedelta(years=last_n_years)
            ).isoformat()

        return start_date

    def get_end_date(self, end_date):
        if end_date is None:
            end_date = datetime.date.today().isoformat()

        return end_date

    def get_dtm_adm_level(self, dtm_adm_level):
        if dtm_adm_level is None:
            if int(self.adm_level[-1]) > 2:
                dtm_adm_level = "ADM2"
            else:
                dtm_adm_level = self.adm_level

        return dtm_adm_level

    def get_asset_files(self):
        asset_files = []

        for asset in self.config["asset_data"]:
            asset = asset.replace("global_", "")
            asset_file = self._build_filename(
                self.iso_code, asset, self.local_dir, ext="tif"
            )
            asset_files.append(asset_file)

        return asset_files

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

    def calculate_ahp(
        self,
        ahp_precision: int = 5,
        ahp_random_index: str = "saaty",
        cr_threshold: float = 0.10,
    ) -> gpd.GeoDataFrame:
        """Calculate hazard weights using the Analytic Hierarchy Process (AHP).

        Prompts the user to provide pairwise comparisons between hazards and computes
        the weights. If the consistency ratio exceeds the threshold, an error is raised.

        Args:
            ahp_precision (int, optional): Number of decimal places for AHP weights.
                Defaults to 5.
            ahp_random_index (str, optional): Method for calculating random index
                in consistency ratio computation. Defaults to "saaty".
            cr_threshold (float, optional): Maximum allowed consistency ratio for the
                pairwise comparisons. Defaults to 0.10.

        Returns:
            pd.DataFrame: Updated dataset with multi-hazard scores recalculated
                using the new weights.
        """

        hazard_dicts, categories = [], list(self.config["hazards"].keys())
        for category in categories:
            hazard_dicts.append(self.config["hazards"][category])

        hazards_all = reduce(lambda a, b: {**a, **b}, hazard_dicts)
        hazards = []
        for hazard in hazards_all:
            if hazard in self.data.columns:
                hazards.append(hazard)

        # Generate all unique pairwise combinations of hazards
        combinations = list(itertools.combinations(hazards, 2))

        # Initialize dictionary to store user-provided pairwise weights
        weight_dict = dict()

        # Prompt user for pairwise comparisons
        for combination in combinations:
            text = f"How much more important is {str(combination[0])} compared to {str(combination[1])}: "
            weight = input(text)
            weight_dict[combination] = weight

        # Build AHP model and calculate weights
        hazard_weights = ahpy.Compare(
            name="Hazards",
            comparisons=weight_dict,
            precision=ahp_precision,
            random_index=ahp_random_index,
        )

        # Extract consistency ratio and log it
        cr = hazard_weights.consistency_ratio
        logging.info(f"Consistency_ratio: {cr}")

        # Check if consistency ratio is acceptable
        if cr < cr_threshold:
            # Update hazard weights in config
            weights = hazard_weights.target_weights
            for hazard in weights:
                for category in categories:
                    if hazard in self.config["hazards"][category]:
                        self.config["hazards"][category][hazard] = weights[
                            hazard
                        ]
            logging.info(self.config["hazards"])

            # Recalculate multi-hazard scores in the dataset
            self.data = self.calculate_multihazard_score(self.data)
            return self.data
        else:
            # Raise error if consistency ratio exceeds threshold
            raise ValueError(
                f"Consistency ratio {cr} > 0.10. Please try again."
            )

    def assign_grouping(self):
        """Assigns a grouping to administrative units based on configuration.

        Checks if the current ISO code exists in the administrative configuration.
        If a grouping column is not already present in the dataset, it maps
        administrative units to their respective group using the configuration.

        Returns:
            pd.DataFrame: Updated dataset with a new grouping column added if applicable.
        """

        if self.iso_code in self.adm_config:
            config = self.adm_config[self.iso_code]
            group = config["group"]

            # Only add grouping column if it doesn't exist yet
            if group not in self.data.columns:
                adm_level = config["adm_level"]
                grouping = config["grouping"]

                # Map administrative level to group
                self.data[group] = self.data[adm_level].map(grouping)

        return self.data

    def combine_datasets(self) -> gpd.GeoDataFrame:
        """Combine hazard, Fathom, and ACLED datasets into a single GeoDataFrame.

        Iterates through the available datasets, replaces missing values with 0,
        and merges them based on the configured merge columns.

        Returns:
            gpd.GeoDataFrame: Combined GeoDataFrame with all datasets aligned on the merge columns.

        Raises:
            ValueError: If no datasets are available to combine.
        """

        data = []

        datasets = [self.assets, self.hazards, self.fathom]

        # Add hazards and Fathom datasets if available, replacing NaN with 0
        for dataset in datasets:
            if dataset is not None:
                dataset = dataset.mask(dataset.isna(), 0)
                data.append(dataset)

        # Add IOM DTM data if available and non-empty
        if self.dtm_agg is not None and self.dtm_adm_level == self.adm_level:
            if len(self.dtm_agg) > 0:
                data.append(self.dtm_agg)

        # Add IDMC data if available and non-empty
        if self.idmc_gidd_conflict_agg is not None:
            if len(self.idmc_gidd_conflict_agg) > 0:
                data.append(self.idmc_gidd_conflict_agg)

        if self.idmc_gidd_disaster_agg is not None:
            if len(self.idmc_gidd_disaster_agg) > 0:
                data.append(self.idmc_gidd_disaster_agg)

        # Add aggregated ACLED data if available and non-empty
        if self.acled_agg is not None:
            if len(self.acled_agg) > 0:
                data.append(self.acled_agg)

        # Add aggregated ACLED data if available and non-empty
        if self.ucdp_agg is not None:
            if len(self.ucdp_agg) > 0:
                data.append(self.ucdp_agg)

        if not data:
            raise ValueError("No datasets available to combine.")

        # Merge all datasets on configured columns
        data = data_utils._merge_data(data, columns=self.merge_columns)

        if (
            "idmc_conflict_idp_total" in data.columns
            and "idmc_disaster_idp_total" in data.columns
        ):
            data["idmc_idp_total"] = data["idmc_conflict_idp_total"].add(
                data["idmc_disaster_idp_total"], fill_value=0
            )

        return data

    def calculate_multihazard_score(
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
        """Calculate multi-hazard scores for each geographic unit.

        Computes multi-hazard scores (MHS) for each row in the dataset using
        either power mean, geometric mean, or arithmetic mean aggregation
        of all hazards. Optionally scales MHS with a conflict column.

        Args:
            data (gpd.GeoDataFrame, optional): Input GeoDataFrame containing hazard data.
            conflict_column (str, optional): Column name for conflict scaling.
                Defaults to "wbg_conflict".
            suffixes (list, optional): List of suffixes for hazard columns to aggregate.
                Defaults to ["exposure_relative", "intensity_weighted_exposure_relative"].
            aggregation (str, optional): Aggregation method: "power_mean", "geometric_mean",
                or "arithmetic_mean". Defaults to "power_mean".
            p (float, optional): Exponent for power mean aggregation. Defaults to 0.5.
            epsilon (float, optional): Small value to avoid zeros in geometric mean. Defaults to 1e-5.

        Returns:
            gpd.GeoDataFrame: Updated GeoDataFrame with new MHS columns added.
        """

        # Ensure relative exposure columns exist for all hazard columns
        for asset in self.config["asset_data"]:
            asset = asset.replace("global_", "")
            for column in data.columns:
                if "relative" not in column and asset in column:
                    colname = f"{column}_relative"
                    if "exposure" in column and colname not in data.columns:
                        data[colname] = data[column] / data[asset]

        for column in data.columns:
            if "worldcover" in column and "relative" not in column:
                data[column] = data[column] * 0.01

        # Loop through each suffix to calculate MHS
        for suffix in suffixes:
            # Prepare hazard columns and normalized weights
            hazard_dicts, categories = [], list(self.config["hazards"].keys())
            for category in categories:
                hazard_dicts.append(self.config["hazards"][category])

            all_hazards = reduce(lambda a, b: {**a, **b}, hazard_dicts)
            hazard_dicts.append(all_hazards)
            categories.append("all")

            for hazard_dict, category in zip(hazard_dicts, categories):
                for asset in self.config["asset_data"]:
                    asset = asset.replace("global_", "")
                    hazard_cols = [
                        f"{hazard}_{asset}_{suffix}" if suffix else hazard
                        for hazard in hazard_dict
                        if (f"{hazard}_{asset}_{suffix}" if suffix else hazard)
                        in data.columns
                    ]

                    total_weight = sum(list(hazard_dict.values()))
                    weights = np.array(
                        [hazard_dict[hazard] for hazard in hazard_dict]
                    )
                    if total_weight > 0:
                        weights = weights / total_weight

                    # Select only columns that exist in data
                    hazard_cols = [
                        col for col in hazard_cols if col in data.columns
                    ]
                    weights = weights[: len(hazard_cols)]

                    # Compute MHS using vectorized operations
                    if self.mhs_aggregation == "power_mean":
                        mhs = (data[hazard_cols] ** p).multiply(
                            weights, axis=1
                        ).sum(axis=1) ** (1 / p)

                    elif self.mhs_aggregation == "geometric_mean":
                        mhs = (
                            (data[hazard_cols] + epsilon)
                            .pow(weights)
                            .prod(axis=1)
                        )

                    elif self.mhs_aggregation == "arithmetic_mean":
                        mhs = (
                            data[hazard_cols]
                            .multiply(weights, axis=1)
                            .sum(axis=1)
                        )

                    # Add MHS column (scaled 0-1)
                    mhs_name = "mhs"
                    if suffix is not None:
                        mhs_name = f"{mhs_name}_{category}_{asset}_{suffix}"
                    data[mhs_name] = data_utils._minmax_scale(mhs)

                    # Optionally scale MHS by conflict
                    for conflict_column in conflict_columns:
                        mhsc_name = f"mhs_{category}_{conflict_column}"

                        if suffix is not None:
                            mhsc_name = f"{mhsc_name}_{asset}_{suffix}"

                        for csuffix in suffixes:
                            if (
                                f"{conflict_column}_{asset}_{csuffix}"
                                in data.columns
                            ):
                                conflict_scaled = data_utils._minmax_scale(
                                    data[
                                        f"{conflict_column}_{asset}_{csuffix}"
                                    ]
                                )
                                data[mhsc_name] = (
                                    data[mhs_name] * conflict_scaled
                                )

        return data

    def download_geoboundary(
        self, adm_source: str, adm_level: str, overwrite: bool = False
    ) -> gpd.GeoDataFrame:
        """Download and prepare administrative boundaries for a country.

        Downloads GADM or GeoBoundaries data for the ISO code and ADM level,
        handles renaming of columns, CRS transformations, and joins multiple levels
        if necessary.

        Args:
            adm_source (str): Source of administrative boundaries, either 'gadm' or 'geoboundary'.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing the administrative boundaries
            in the target CRS.

        Raises:
            ValueError: If `adm_source` is not recognized.
            requests.RequestException: If the download from the remote server fails.
            FileNotFoundError: If the output file cannot be found or created.
        """

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
                        self.iso_code, adm_level, self.local_dir, ext="geojson"
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
                    f"adm_source '{adm_source}' not recognized. Use 'gadm' or 'geoboundary'."
                )

            geoboundary.to_crs(self.crs).to_file(out_file)
            logging.info(
                f"Geoboundary file saved to {os.path.basename(out_file)}."
            )

        # Load final geoboundary and update attributes
        geoboundary = gpd.read_file(out_file).to_crs(self.crs)

        if adm_level == self.adm_level:
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
        try:
            osm = osm.sjoin(
                self.geoboundary, how="left", predicate="intersects"
            )
            osm = osm.drop(["index_right"], axis=1)
        except Exception as e:
            raise ValueError(f"Spatial join failed: {e}")

        osm = osm.rename(
            columns={"category": "osm_category", "amenity": "osm_amenity"}
        )
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

        if (
            self.overwrite
            or not os.path.exists(gidd_file)
            or not os.path.exists(filtered_file)
        ):
            idmc_gidd_url = self.config["urls"]["idmc_gidd_url"].format(
                self.idmc_key, self.iso_code, cause
            )
            self._download_url_progress(idmc_gidd_url, gidd_file)
            idmc_gidd = gpd.read_file(gidd_file, use_arrow=True)
            idmc_gidd = idmc_gidd.sjoin(
                self.geoboundary, how="left", predicate="intersects"
            )
            idmc_gidd = idmc_gidd.drop(["index_right"], axis=1)
            idmc_gidd.to_file(gidd_file)

            if filtered:
                if len(idmc_gidd) == 0:
                    return

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
        if filtered:
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
            return idmc_gidd_agg

        return idmc_gidd

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
                    "Network connection to https://dtm.iom.int could not be established."
                )

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
                    f"DTM download failed for {self.iso_code} ({self.country})."
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
            dataset = f"{self.global_name}_{self.ucdp_name}".lower()
            self.download_url(dataset=dataset, ext="csv")

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
            try:
                admin = self.geoboundary
                ucdp = ucdp.sjoin(admin, how="left", predicate="intersects")
                ucdp = ucdp.drop(["index_right"], axis=1)
            except Exception as e:
                raise ValueError(f"Spatial join failed: {e}")

            ucdp.to_file(local_file, driver="GeoJSON")
            logging.info(f"Saving UCDP to {local_file}")

        ucdp = gpd.read_file(local_file)

        if aggregate:
            ucdp_agg = None
            admin = self.geoboundary

            for asset, asset_file in zip(
                self.config["asset_data"], self.asset_files
            ):
                asset = asset.replace("global_", "")
                column = f"{self.ucdp_name.lower()}_{asset}_exposure"
                exposure_raster = self._build_filename(
                    self.iso_code,
                    f"{self.ucdp_name}_{asset}_exposure",
                    self.local_dir,
                    ext="tif",
                )
                exposure_vector = self._build_filename(
                    self.iso_code,
                    f"{self.ucdp_name}_{asset}_exposure_{self.adm_level}",
                    self.local_dir,
                    ext="geojson",
                )

                if self.overwrite or not os.path.exists(exposure_vector):
                    out_tif = self._calculate_custom_conflict_exposure(
                        local_file,
                        asset_file,
                        asset_name=asset,
                        conflict_src="ucdp",
                    )
                    out_tif, _ = self._calculate_exposure(
                        asset_file, out_tif, exposure_raster, threshold=1
                    )

                    data = self._calculate_zonal_stats(
                        out_tif,
                        column=column,
                        stats_agg=["sum"],
                        out_file=exposure_vector,
                    )
                    if data is None or data.empty:
                        raise ValueError(
                            "Exposure calculation failed or produced empty results."
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
            column = "conflict_count"
            event_count = self._aggregate_data(
                ucdp, agg_col=column, agg_func="count"
            )
            event_count = event_count.rename(
                columns={column: f"ucdp_{column}"}
            )
            event_count = data_utils._merge_data(
                [admin, event_count],
                columns=[f"{self.adm_level}_ID"],
                how="left",
            )
            fatalities_count = self._aggregate_data(
                ucdp, agg_col="best", agg_func="sum"
            )
            fatalities_count = fatalities_count.rename(
                columns={"best": "ucdp_fatalities"}
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
        """Download ACLED conflict data and optionally aggregate to administrative units.

        Downloads ACLED data for the given ISO country code and date range. The function
        can also aggregate conflict events to administrative units if requested.

        Args:
            population (str, optional): Type of population data to include. Defaults to "full".
            aggregate (bool, optional): Whether to aggregate conflict events to administrative units. Defaults to False.

        Returns:
            gpd.GeoDataFrame: ACLED conflict data as a GeoDataFrame. If aggregation is requested, returns aggregated data.

        Raises:
            ValueError: If ACLED API returns no data for the given query.
            requests.RequestException: If the request to ACLED API fails.
            FileNotFoundError: If the output file cannot be written or read.
        """

        # Build file paths
        raw_file = self._build_filename(
            self.iso_code, self.acled_name, self.local_dir, ext="geojson"
        )
        filtered_file = self._build_filename(
            self.iso_code,
            f"{self.acled_name}_FILTERED",
            self.local_dir,
            ext="geojson",
        )
        agg_file = self._build_filename(
            self.iso_code,
            f"{self.acled_name}_{self.adm_level}",
            self.local_dir,
            ext="geojson",
        )

        # Download ACLED data
        if self.overwrite or not os.path.exists(filtered_file):
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
                    subdata = pd.DataFrame(response.json()["data"])
                    data.append(subdata)
                    len_subdata = len(subdata)
                    params["page"] = params["page"] + 1
                except Exception as e:
                    warnings.warn("WARNING: ACLED failed to download.")
                    logging.info(e)
                    return

            # Concatenate all pages
            data = pd.concat(data)
            if len(data) == 0:
                warnings.warn(f"No ACLED data returned for {self.iso_code}")
                return

            # Convert to GeoDataFrame
            try:
                data = gpd.GeoDataFrame(
                    geometry=gpd.points_from_xy(
                        data["longitude"], data["latitude"], crs=self.crs
                    ),
                    data=data,
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to convert ACLED data to GeoDataFrame: {str(e)}"
                )

            # Clean and standardize columns
            if exposure_column in data.columns:
                data[exposure_column] = (
                    data[exposure_column]
                    .replace({"": np.nan})
                    .astype(np.float64)
                )

            # Spatial join ACLED events with admin boundaries
            try:
                data = data.sjoin(
                    self.geoboundary, how="left", predicate="intersects"
                )
                data = data.drop(["index_right"], axis=1)
            except Exception as e:
                raise ValueError(f"Spatial join failed: {e}")

            data.to_file(raw_file)
            logging.info(f"ACLED file saved to {raw_file}.")

        # Read ACLED data from file
        try:
            self.acled_raw = gpd.read_file(raw_file).to_crs(self.crs)
            acled = self._filter_acled(self.acled_raw)
            acled.to_file(filtered_file)
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to read ACLED file {raw_file}: {str(e)}"
            )

        # Aggregate to admin units if requested
        if aggregate:
            acled = self._aggregate_acled(
                acled_file=filtered_file, agg_file=agg_file
            )

        return acled

    def _filter_acled(self, acled):
        mask = (
            acled["disorder_type"].isin(
                self.acled_filters["include"].get("disorder_type", [])
            )
            & acled["event_type"].isin(
                self.acled_filters["include"].get("event_type", [])
            )
            & acled["sub_event_type"].isin(
                self.acled_filters["include"].get("sub_event_type", [])
            )
        )

        for cat in ["disorder_type", "event_type", "sub_event_type"]:
            if cat in self.acled_filters["exclude"]:
                mask &= ~acled[cat].isin(self.acled_filters["exclude"][cat])

        return acled[mask]

    def _aggregate_acled(
        self,
        acled_file: str,
        agg_file: str,
        prefix: str = "wbg",
    ):
        """Aggregate ACLED data and calculate exposure at the administrative level.

        This function performs the following steps:
        1. Reads the raw ACLED file.
        2. Aggregates ACLED events if an aggregated file does not exist.
        3. Calculates exposure raster and zonal statistics if the exposure vector does not exist.
        4. Merges aggregated ACLED data with exposure values.

        Args:
            acled_file (str): Path to the raw ACLED GeoJSON file.
            agg_file (str): Path to save or read the aggregated ACLED data.
            exposure_raster (str): Path to save or read the rasterized exposure.
            exposure_vector (str): Path to save or read the vectorized exposure results.
            prefix (str, optional): Prefix for the exposure variable. Defaults to "wbg".
            column (str, optional): Column name for the exposure calculation. Defaults to "conflict_exposure".

        Returns:
            gpd.GeoDataFrame: ACLED data aggregated and merged with exposure information.

        Raises:
            FileNotFoundError: If `acled_file` does not exist.
            ValueError: If exposure calculation fails or produces invalid results.
        """

        # Read the ACLED raw data if it exists
        if not os.path.exists(acled_file):
            raise FileNotFoundError(f"ACLED file not found: {acled_file}")
        acled = gpd.read_file(acled_file)

        # Aggregate ACLED events if the aggregated file does not exist
        if not os.path.exists(agg_file):
            self._aggregate_acled_exposure(acled, agg_file)
        agg = gpd.read_file(agg_file)

        # Calculate exposure vector if it does not exist
        full_data = [agg]
        for asset, asset_file in zip(
            self.config["asset_data"], self.asset_files
        ):
            asset = asset.replace("global_", "")
            exposure_raster = self._build_filename(
                self.iso_code,
                f"{self.acled_name}_{asset}_exposure",
                self.local_dir,
                ext="tif",
            )
            exposure_vector = self._build_filename(
                self.iso_code,
                f"{self.acled_name}_{asset}_exposure_{self.adm_level}",
                self.local_dir,
                ext="geojson",
            )

            column = f"{self.acled_name.lower()}_{asset}_exposure"
            if self.overwrite or not os.path.exists(exposure_vector):
                acled_tif = self._calculate_custom_conflict_exposure(
                    acled_file,
                    asset_file,
                    asset_name=asset,
                    conflict_src="acled",
                )
                out_tif, _ = self._calculate_exposure(
                    asset_file, acled_tif, exposure_raster, threshold=1
                )
                subdata = self._calculate_zonal_stats(
                    out_tif,
                    column=column,
                    prefix=prefix,
                    stats_agg=["sum"],
                    out_file=exposure_vector,
                )
                if subdata is None or subdata.empty:
                    raise ValueError(
                        "Exposure calculation failed or produced empty results."
                    )

            # Read exposure vector and clean zero values
            exposure_var = prefix + "_" + column
            exposure = gpd.read_file(exposure_vector)
            exposure.loc[exposure[exposure_var] == 0, exposure_var] = None
            full_data.append(exposure)

        # Merge aggregated ACLED and exposure data
        acled = data_utils._merge_data(full_data, columns=self.merge_columns)

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
        """Calculate a buffered conflict exposure raster from point events.

        This function applies event-specific buffer distances to conflict events,
        creates a temporary GeoJSON file with buffered geometries, and then
        rasterizes it to match the asset grid.

        Args:
            conflict_file (str): Path to the raw conflict GeoJSON file.
            temp_name (str, optional): Suffix for the temporary buffered GeoJSON. Defaults to "temp".
            meter_crs (str, optional): CRS to use for buffering in meters. Defaults to "EPSG:3857".

        Returns:
            str: Path to the rasterized exposure file.

        Raises:
            FileNotFoundError: If the conflict file does not exist.
            RuntimeError: If rasterization fails or GDAL command fails.
        """
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
            + f"_{asset_name.upper()}_{temp_name.upper()}.geojson"
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

                os.system(f"gdal_rasterize -burn 1 {temp_file} {out_file}")

            except Exception as e:
                raise RuntimeError(f"Error creating exposure raster: {e}")

        return out_file

    def _aggregate_acled_exposure(
        self, acled: gpd.GeoDataFrame, agg_file: str
    ) -> gpd.GeoDataFrame:
        """Aggregate ACLED event exposure by administrative units.

        Performs spatial joins between ACLED points and administrative boundaries,
        calculates population-weighted exposure and event counts, and saves the
        aggregated data to a GeoJSON file.

        Args:
            acled (gpd.GeoDataFrame): ACLED event data with population and conflict columns.
            agg_file (str): Output file path for the aggregated GeoDataFrame.

        Returns:
            gpd.GeoDataFrame: Aggregated exposure data at the administrative unit level.

        Raises:
            FileNotFoundError: If the geoboundary data required for aggregation is missing.
            ValueError: If the ACLED dataset is empty or missing required columns.
            RuntimeError: If saving the aggregated data to file fails.
        """

        # Check that geoboundary data is loaded
        if not hasattr(self, "geoboundary") or self.geoboundary is None:
            raise FileNotFoundError(
                "Geoboundary data is missing. Cannot aggregate ACLED events."
            )

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
            columns={"population_best": "acled_population_best"}
        )

        # Aggregate total conflict events
        event_count = self._aggregate_data(
            acled, agg_col="conflict_count", agg_func="count"
        )
        event_count = event_count.rename(
            columns={"conflict_count": "acled_conflict_count"}
        )

        # Aggregate total conflict events
        fatalities_count = self._aggregate_data(
            acled, agg_col="fatalities", agg_func="sum"
        )
        fatalities_count = fatalities_count.rename(
            columns={"fatalities": "acled_fatalities"}
        )

        # Aggregate conflict events where population_best is missing
        null_pop_event_count = self._aggregate_data(
            acled[acled["population_best"].isna()],
            agg_col="null_conflict_count",
            agg_func="count",
        )
        null_pop_event_count = null_pop_event_count.rename(
            columns={"null_conflict_count": "acled_null_conflict_count"}
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
        exposure_var = "acled_exposure"
        acled[exposure_var] = acled["acled_population_best"] / (
            acled["acled_conflict_count"]
            - acled["acled_null_conflict_count"].fillna(0)
        )
        acled.loc[acled[exposure_var] == 0, exposure_var] = None

        acled = self._calculate_conflict_stats(acled, source="acled")

        # Save aggregated GeoDataFrame to file
        acled.to_file(agg_file)

        return acled

    def _calculate_conflict_stats(self, data, source: str = "acled"):
        data[f"{source}_fatalities_per_conflict"] = data[
            f"{source}_fatalities"
        ].div(data[f"{source}_conflict_count"])
        data[f"{source}_fatalities_per_conflict"] = data[
            f"{source}_fatalities_per_conflict"
        ].replace([np.inf, -np.inf], np.nan)

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
        """Download a dataset from a configured URL and save it locally.

        Downloads raster or zipped datasets from a remote URL specified in the
        configuration. Supports clipping rasters to the country boundary if applicable.

        Args:
            dataset (str): Name of the dataset to download (e.g., 'gadm', 'population').
            dataset_name (str, optional): Custom dataset name for saving locally.
                Defaults to None.
            ext (str, optional): File extension for the dataset. Defaults to 'tif'.

        Returns:
            str: Path to the downloaded and optionally clipped dataset.

        Raises:
            ValueError: If the dataset URL is not found in the configuration.
            RuntimeError: If the download fails or the local file cannot be created.
        """

        # Determine the dataset name for local storage
        if dataset_name is None:
            dataset_name = dataset.replace(f"{self.global_name.lower()}_", "")

        # Build path for the global version of the dataset
        global_file = self._build_filename(
            self.global_name, dataset_name, self.global_dir, ext=ext
        )

        # Construct URL from config
        url_name = f"{dataset}_url"
        if "fluvial_flood" in dataset and self.jrc_version == "v1":
            url_name = f"{dataset}_{self.jrc_version}_url"

        if url_name in self.config["urls"]:
            if dataset == "gadm":
                url = self.config["urls"][url_name].format(
                    self.iso_code, self.adm_level[-1]
                )
            elif "wildfire" in url_name or "lightning" in url_name:
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
                try:
                    if url.endswith(".zip"):
                        self.download_zip(
                            url, dataset, out_file=global_file, ext=ext
                        )
                    elif url.endswith(".tif") or (".tif" in url):
                        self._download_url_progress(url, global_file)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to download {dataset} from {url}: {e}"
                    )

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
        """Download and extract a ZIP archive, then convert/copy the relevant dataset.

        This function downloads a ZIP file, extracts it, and processes its contents
        depending on the desired output format (`tif` or `geojson`).

        Args:
            url (str): URL to download the ZIP file from.
            dataset (str): Dataset name (used to determine storage location).
            out_file (str): Path to the final output file to save.
            ext (str, optional): Desired file extension ("tif" or "geojson").
                Defaults to "tif".

        Raises:
            RuntimeError: If the download, extraction, or file conversion fails.
            FileNotFoundError: If no suitable file is found in the extracted archive.
            ValueError: If `ext` is not supported.
        """

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

        else:
            raise ValueError(f"Unsupported extension: {ext}")

    def download_jrc(self, name: str):
        out_dir = os.path.join(self.global_dir, name)
        os.makedirs(out_dir, exist_ok=True)

        url_name = f"{name}_{self.jrc_version}_url"
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
        map_code = {
            "tree_cover": 10,
            "shrubland": 20,
            "grassland": 30,
            "cropland": 40,
            "builtup": 50,
            "bare_sparse_vegetation": 60,
            "snow_and_ice": 70,
            "permanent_water_bodies": 80,
            "herbaceous_wetland": 90,
            "mangroves": 95,
            "moss_and_lichen": 100,
        }

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
                    if not os.path.exists(worldcover_file):
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

    def download_fathom(self) -> gpd.GeoDataFrame | None:
        """Download, process, and aggregate Fathom flood data for the given country.

        This function processes local Fathom flood hazard data by building VRTs and
        converting them to GeoTIFFs, clipping the rasters to administrative boundaries,
        generating exposure rasters, and calculating zonal statistics.

        The results are merged into an aggregated dataset, which is then saved as a
        GeoJSON file for reuse.

        Returns:
            gpd.GeoDataFrame | None: The processed Fathom data as a GeoDataFrame
            in the target CRS, or `None` if the Fathom directory does not exist.

        Raises:
            RuntimeError: If GDAL commands or raster operations fail.
        """

        # Build paths
        fathom_folder = f"{self.iso_code}_{self.fathom_name}".upper()
        fathom_dir = os.path.join(self.local_dir, fathom_folder)

        full_data_file = self._build_filename(
            self.iso_code, self.fathom_name, self.local_dir, ext="geojson"
        )

        # If no Fathom directory exists, skip processing
        if not os.path.exists(fathom_dir):
            return None

        # If processed dataset doesn't exist, generate it
        if self.overwrite or not os.path.exists(full_data_file):
            full_data = None
            folders = next(os.walk(fathom_dir))[1]

            for index, folder in enumerate(folders):
                logging.info(
                    f"({index+1}/{len(folders)}) Processing {folder.lower()}..."
                )

                # File naming
                name = f"{self.iso_code}_{folder}_rp{self.fathom_rp}".upper()
                raw_tif_file = os.path.join(fathom_dir, f"{name}.tif")
                proc_tif_file = os.path.join(self.local_dir, f"{name}.tif")

                # If processed file doesn't exist, build from VRT
                if self.overwrite or not os.path.exists(proc_tif_file):
                    flood_dir = os.path.join(
                        fathom_dir,
                        folder,
                        str(self.fathom_year),
                        f"1in{self.fathom_rp}",
                    )
                    merged_file = os.path.join(fathom_dir, f"{name}.vrt")
                    self._merge_tifs(
                        f"{flood_dir}/*.tif", merged_file, raw_tif_file
                    )

                # Clip raster to admin boundary
                admin = self.geoboundary.dissolve(by="iso_code")
                nodata = self.config["nodata"][folder.lower()]
                self._clip_raster(raw_tif_file, proc_tif_file, admin, nodata)

                # Build exposure rasters
                full_data = None
                for asset, asset_file in (
                    pbar := tqdm(
                        zip(self.config["asset_data"], self.asset_files),
                        total=len(self.asset_files),
                    )
                ):

                    asset = asset.replace("global_", "")
                    pbar.set_description(f"Processing {asset}")

                    exposure_file = self._build_filename(
                        self.iso_code,
                        f"{folder}_{asset}_exposure",
                        self.local_dir,
                        ext="tif",
                    )
                    weighted_exposure_file = self._build_filename(
                        self.iso_code,
                        f"{folder}_{asset}_intensity_weighted_exposure",
                        self.local_dir,
                        ext="tif",
                    )

                    self._generate_exposure(
                        asset,
                        asset_file,
                        proc_tif_file,
                        exposure_file,
                        self.config["threshold"][folder.lower()],
                    )

                    # Custom flood metric: share of pixels > threshold
                    def custom(x):
                        return np.sum(x > self.fathom_threshold) / x.size

                    add_stats = {"custom": custom}

                    # Zonal statistics for hazard intensity
                    data = self._calculate_zonal_stats(
                        proc_tif_file,
                        column=folder.lower(),
                        add_stats=add_stats,
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

                    # Add exposure zonal statistics
                    if os.path.exists(exposure_file):
                        exposure = self._calculate_zonal_stats(
                            exposure_file,
                            column=folder.lower(),
                            suffix=f"{asset}_exposure",
                        )
                        weighted_exposure = self._calculate_zonal_stats(
                            weighted_exposure_file,
                            column=folder.lower(),
                            suffix=f"{asset}_intensity_weighted_exposure",
                        )
                        data_utils._merge_data(
                            [full_data, exposure, weighted_exposure],
                            columns=self.merge_columns,
                        )

            # Save processed Fathom dataset
            full_data.to_file(full_data_file)

        # Always load and return as GeoDataFrame in correct CRS
        full_data = gpd.read_file(full_data_file).to_crs(self.crs)

        return full_data

    def download_datasets(self, name: str = None) -> gpd.GeoDataFrame:
        if name is None:
            name = "hazard"

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
                if ("flood" in dataset) and (self.fathom is not None):
                    continue
                elif (
                    ("flood" in dataset)
                    and (self.fathom is None)
                    and (self.jrc_version == "v2")
                ):
                    local_file = self.download_jrc(dataset)
                elif "worldcover" in dataset:
                    land_cover_class = dataset.split("_")[-1]
                    local_file = self.download_worldcover(
                        land_cover_class=land_cover_class,
                        resample=self.resample_worldcover,
                    )
                else:
                    local_file = self.download_url(dataset, ext="tif")

                dataset_name = dataset.replace("global_", "")

                # File paths for derived exposures
                for asset, asset_file in (
                    pbar := tqdm(
                        zip(self.config["asset_data"], self.asset_files),
                        total=len(self.asset_files),
                    )
                ):

                    asset = asset.replace("global_", "")
                    pbar.set_description(f"Processing {asset}")

                    exposure_file = self._build_filename(
                        self.iso_code,
                        f"{dataset_name}_{asset}_exposure",
                        self.local_dir,
                        ext="tif",
                    )
                    weighted_exposure_file = self._build_filename(
                        self.iso_code,
                        f"{dataset_name}_{asset}_intensity_weighted_exposure",
                        self.local_dir,
                        ext="tif",
                    )

                    # Generate exposure rasters (skip asset rasters)
                    if dataset not in self.config["asset_data"]:
                        self._generate_exposure(
                            asset,
                            asset_file,
                            local_file,
                            exposure_file,
                            self.config["threshold"][dataset],
                        )

                    # Decide on aggregation method
                    if dataset in self.config["asset_data"]:
                        stats_agg = ["sum"]
                    else:
                        stats_agg = ["mean"]

                    # Zonal statistics for base hazard raster
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

                    # Add exposure statistics if available
                    if os.path.exists(exposure_file):
                        exposure = self._calculate_zonal_stats(
                            exposure_file,
                            column=dataset_name,
                            suffix=f"{asset}_exposure",
                        )
                        weighted_exposure = self._calculate_zonal_stats(
                            weighted_exposure_file,
                            column=dataset_name,
                            suffix=f"{asset}_intensity_weighted_exposure",
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
        """
        Generate an exposure raster by resampling the input raster and applying a threshold.

        If the exposure file does not already exist, the function first checks for a
        resampled raster. If unavailable, it resamples the input raster and then
        calculates exposure values based on the provided threshold.

        Args:
            local_file (str): Path to the input raster file (.tif).
            exposure_file (str): Path where the generated exposure raster will be saved.
            threshold (float): Threshold value used to calculate exposure.

        Raises:
            FileNotFoundError: If the input raster file does not exist.
            RuntimeError: If resampling or exposure calculation fails.
        """
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
        """
        Resample a raster file to match the resolution and bounds of the asset raster.

        This function uses GDAL to warp the input raster so that it aligns with the
        spatial resolution and bounding box of the reference asset raster.

        Args:
            in_file (str): Path to the input raster file.
            out_file (str): Path where the resampled raster will be saved.

        Returns:
            str: Path to the resampled raster file.

        Raises:
            FileNotFoundError: If the input raster or the asset file does not exist.
            RuntimeError: If GDAL fails to perform the resampling.
        """
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
        """
        Calculate exposure and intensity-weighted exposure rasters.

        This function compares a hazard raster with an asset raster to compute:
          - A binary exposure raster (1 if hazard ≥ threshold, else 0).
          - An exposure raster (asset values masked by hazard threshold).
          - An intensity-weighted exposure raster (exposure weighted by scaled hazard intensity).

        Args:
            hazard_file (str): Path to the input hazard raster file.
            exposure_file (str): Path where the exposure raster will be saved.
            threshold (float): Hazard threshold for defining exposure.

        Returns:
            tuple[str, str]: Paths to the exposure file and weighted exposure file.

        Raises:
            FileNotFoundError: If the hazard file or asset file does not exist.
        """
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
        """
        Aggregate data for a given administrative level.

        This function groups the input GeoDataFrame by the administrative level ID
        and applies an aggregation function to a specified column. If `agg_func` is
        "count", it counts the number of rows per administrative unit. Otherwise, it
        applies the specified aggregation function (e.g., sum, mean) to `agg_col`
        within each administrative unit.

        Args:
            data (gpd.GeoDataFrame): Input GeoDataFrame to aggregate.
            agg_col (str, optional): Column to aggregate. Required if `agg_func` is not "count".
            agg_func (str, optional): Aggregation function to use. Defaults to "sum".

        Returns:
            gpd.GeoDataFrame: Aggregated GeoDataFrame with administrative IDs and aggregated values.

        Raises:
            ValueError: If `agg_func` is not "count" and `agg_col` is None.
        """

        # Define administrative ID column
        if adm_level is None:
            adm_level = self.adm_level

        agg_name = f"{adm_level}_ID"
        if agg_name not in data.columns:
            agg_name = adm_level

        if agg_func != "count" and agg_col is None:
            raise ValueError(
                "agg_col must be provided when agg_func is not 'count'."
            )

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
        """
        Clips a global raster to the boundary of a given admin unit and saves it locally.

        Args:
            global_tif (str): Path to the global raster file (GeoTIFF).
            local_tif (str): Path to save the clipped raster file.
            admin (gpd.GeoDataFrame): GeoDataFrame containing the admin boundary geometry.
            nodata (list, optional): List of nodata values to mask. Defaults to [].

        Returns:
            rasterio.io.DatasetReader: The clipped raster dataset.

        Raises:
            FileNotFoundError: If the global raster file does not exist.
            ValueError: If `admin` GeoDataFrame is empty or invalid.
        """

        # Ensure the input raster exists
        if not os.path.exists(global_tif):
            raise FileNotFoundError(f"Global raster not found: {global_tif}")

        # Ensure the GeoDataFrame contains at least one geometry
        if admin.empty:
            raise ValueError(
                "Admin GeoDataFrame is empty. Cannot perform clipping."
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
        """
        Compute zonal statistics for a raster within administrative boundaries.

        This function overlays a raster file with administrative boundaries to calculate
        aggregated statistics for each administrative unit. It can compute standard
        aggregation functions (e.g., sum, mean) and custom statistics if provided. The
        results are saved to a GeoJSON file and returned as a GeoDataFrame. Column
        names can be customized with optional prefix and suffix.

        Args:
            in_file (str): Path to the input raster file.
            column (str): Base name for the output column containing zonal statistics.
            out_file (str, optional): Path to save the output GeoJSON. If None, a default
                file name is generated in the local directory.
            stats_agg (list, optional): List of aggregation functions to apply. Defaults to ["sum"].
            add_stats (list, optional): List of custom functions to apply to each zone.
            suffix (str, optional): Optional suffix to append to the output column name.
            prefix (str, optional): Optional prefix to prepend to the output column name.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame of administrative units with computed zonal statistics.

        Raises:
            FileNotFoundError: If the input raster file does not exist.
            ValueError: If the raster and administrative boundary CRS cannot be reconciled.
        """

        if not os.path.exists(in_file):
            raise FileNotFoundError(f"Raster file {in_file} does not exist.")

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

            # Save results to GeoJSON
            data.to_file(out_file)

        return gpd.read_file(out_file)

    def _build_filename(self, prefix, suffix, out_dir, ext="geojson") -> str:
        """
        Construct a standardized file path by combining a directory, prefix, suffix, and extension.

        The function generates a filename in the format `{PREFIX}_{SUFFIX}.{EXT}`
        within the specified local directory. Both the prefix and suffix are converted
        to uppercase to maintain consistency.

        Args:
            prefix (str): Prefix for the file name (typically a dataset or country code).
            suffix (str): Suffix for the file name (e.g., variable or administrative level).
            out_dir (str): Directory path where the file should be saved.
            ext (str, optional): File extension, defaults to "geojson".

        Returns:
            str: Full file path constructed from the provided components.

        Raises:
            ValueError: If any of `prefix`, `suffix`, or `out_dir` is empty.
        """
        if not prefix or not suffix or not out_dir:
            raise ValueError(
                "Prefix, suffix, and local_dir must all be provided and non-empty."
            )

        # Construct and return the full file path
        return os.path.join(
            out_dir, f"{prefix.upper()}_{suffix.upper()}.{ext}"
        )
