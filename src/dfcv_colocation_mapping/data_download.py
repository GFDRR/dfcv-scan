import os
import re
import json
import copy
import shutil
import zipfile
import logging
import datetime
import warnings
import itertools

import urllib.request
from functools import reduce
from pathlib import Path
from warnings import simplefilter

import ast
import pyrosm
import quackosm as qosm
from pyrosm.data import sources

import geojson
import importlib_resources
import numpy as np
import pandas as pd
import pycountry
import requests
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

import geopandas as gpd
import rasterio as rio
import rasterio.mask
import rasterstats
from osgeo import gdal, gdalconst
import shapely

from scipy.stats.mstats import gmean
import ahpy
import bs4
import osmnx as ox
from dtmapi import DTMApi

from concurrent.futures import ThreadPoolExecutor, as_completed
from dfcv_colocation_mapping import common_utils

pd.set_option("future.no_silent_downcasting", True)
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
ox.settings.max_query_area_size = 500000000000000
logging.basicConfig(level=logging.INFO, force=True)
io_logger = logging.getLogger("pyogrio._io")
io_logger.setLevel(logging.WARNING)

WARNING = "\033[31m"
RESET = "\033[0m"


class DownloadProgressBar(tqdm):
    """Progress bar for tracking file download progress."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Update the progress bar during a file download.

        Args:
            b (int): Number of blocks transferred so far.
            bsize (int): Size of each block in bytes.
            tsize (int, optional): Total file size in bytes, if known.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class DatasetManager:
    """
    Manager for configuring and accessing country-level datasets.

    Centralizes dataset configuration, credentials, filters, and directory structure
    for asset, conflict, displacement, and other geospatial  datasets.
    """

    def __init__(
        self,
        iso_code: str,
        adm_level: str = "ADM3",
        group: str = "region",
        adm_source: str = "geoboundaries",
        crs: str = "EPSG:4326",
        acled_username: str = None,
        acled_password: str = None,
        acled_country: str = None,
        conflict_start_date: str = None,
        conflict_end_date: str = None,
        conflict_last_n_years: int = 5,
        dtm_key: str = None,
        dtm_adm_level: str = None,
        idmc_key: str = None,
        displacement_start_year: str = None,
        displacement_end_year: str = None,
        displacement_last_n_years: int = 5,
        mhs_aggregation: str = "arithmetic_mean",
        config_file: str = None,
        dtm_cred_file: str = None,
        idmc_cred_file: str = None,
        acled_cred_file: str = None,
        adm_config_file: str = None,
        osm_config_file: str = None,
        acled_config_file: str = None,
        global_name: str = "global",
        data_dir: str = "data",
    ):
        """
        Initialize the dataset manager.

        Args:
            iso_code (str): ISO country code.
            adm_level (str): Administrative level (default: 'ADM3').
            group (str): Spatial grouping level (default: 'region').
            adm_source (str): Source of administrative boundaries (default: 'geoboundaries').
            crs (str): Coordinate reference system (default: 'EPSG:4326').
            acled_username (str, optional): ACLED API username (default: None).
            acled_password (str, optional): ACLED API password (default: None).
            acled_country (str, optional): Country name for ACLED queries (default: None).
            conflict_start_date (str, optional): Conflict start date (default: None).
            conflict_end_date (str, optional): Conflict end date (default: None).
            conflict_last_n_years (int): Lookback window for conflict data (default: 5).
            dtm_key (str, optional): DTM API key override (default: None).
            dtm_adm_level (str, optional): Admin level for DTM data (default: None).
            idmc_key (str, optional): IDMC API key override (default: None).
            displacement_start_year (str, optional): Displacement start year (default: None).
            displacement_end_year (str, optional): Displacement end year (default: None).
            displacement_last_n_years (int): Lookback window for displacement data (default: 5).
            mhs_aggregation (str): Aggregation method for MHS indicators (default: 'arithmetic_mean').
            config_file (str, optional): Main data config file (default: None).
            dtm_cred_file (str, optional): DTM credentials file (default: None).
            idmc_cred_file (str, optional): IDMC credentials file (default: None).
            acled_cred_file (str, optional): ACLED credentials file (default: None).
            adm_config_file (str, optional): Admin boundaries config file (default: None).
            osm_config_file (str, optional): OSM config file (default: None).
            acled_config_file (str, optional): ACLED config file (default: None).
            global_name (str): Name for global-level data directory (default: 'global').
            data_dir (str): Base data directory (default: 'data').
        """

        # Country and spatial configuration
        self.iso_code = iso_code
        self.adm_level = adm_level
        self.adm_source = adm_source
        self.data_dir = data_dir
        self.crs = crs

        # ACLED configuration
        self.acled_country = acled_country
        self.conflict_start_date = self._get_start_date(
            conflict_start_date, conflict_last_n_years
        )
        self.conflict_end_date = self._get_end_date(conflict_end_date)

        # Displacement configuration
        self.dtm_adm_level = self._get_dtm_adm_level(dtm_adm_level)
        self.displacement_start_year = (
            displacement_start_year
            or self._get_year(
                self._get_start_date(
                    displacement_start_year, displacement_last_n_years
                )
            )
        )
        self.displacement_end_year = displacement_end_year or self._get_year(
            self._get_end_date(displacement_end_year)
        )

        # Resolve configuration and credential file paths
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

        # Load configuration files
        self.config = common_utils.read_config(self.config_file)
        self.osm_config = common_utils.read_config(self.osm_config_file)
        self.acled_config = common_utils.read_config(self.acled_config_file)
        self.adm_config = common_utils.read_config(self.adm_config_file)
        self.country = self._get_country_name(self.iso_code)

        # Load API credentials (with optional overrides)
        self.acled_username = self._load_creds(
            self.acled_cred_file, "acled_username", acled_username
        )
        self.acled_password = self._load_creds(
            self.acled_cred_file, "acled_password", acled_password
        )
        self.dtm_key = self._load_creds(self.dtm_cred_file, "dtm_key", dtm_key)
        self.idmc_key = self._load_creds(
            self.idmc_cred_file, "idmc_key", idmc_key
        )

        # Extract config fields
        self.acled_hierarchy = self.acled_config["acled_hierarchy"]
        self.acled_selected = self.acled_config["acled_selected"]
        self.acled_drm_pillars = self.acled_config["acled_drm_pillars"]
        self.asset_categories = self.config["asset_categories"]

        # Set up data directories
        self.global_name = global_name.upper()
        self.data_dir = os.path.join(os.getcwd(), data_dir)
        self.local_dir = os.path.join(self.data_dir, iso_code)
        self.global_dir = os.path.join(self.data_dir, self.global_name)

        # Dataset selection and aggregation settings
        self.set_selected_datasets()
        self.mhs_aggregation = mhs_aggregation
        self.hazard_cols = dict()

    def download_datasets(self):
        """Download and prepare all relevant datasets for the country."""

        # Ensure data directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.local_dir, exist_ok=True)
        os.makedirs(self.global_dir, exist_ok=True)

        # Download admin boundaries
        logging.info(f"Loading {self.adm_level} geoboundaries...")
        self.geoboundary = self.download_geoboundary(
            adm_source=self.adm_source, adm_level=self.adm_level
        )

        # Download asset layers
        logging.info("Loading Asset Layers...")
        self.assets = self.download_assets()

        # Download hazard layers
        logging.info("Loading Hazard Layers...")
        self.hazards = self.download_hazards()

        # Initialize list of datasets to merge
        data = [self.geoboundary]
        for dataset in [self.assets, self.hazards]:
            if dataset is not None:
                data.append(dataset.fillna(0))

        # Download and aggregate conflict datasets
        if "acled" in self.config["conflict_selected"]:
            logging.info("Loading ACLED data...")
            self.acled, self.acled_agg = self._download_with_aggregate(
                self.download_acled
            )
            if self.acled_agg is not None and not self.acled_agg.empty:
                data.append(self.acled_agg)
            self._cleanup()

        if "ucdp" in self.config["conflict_selected"]:
            logging.info("Loading UCDP data...")
            self.ucdp, self.ucdp_agg = self._download_with_aggregate(
                self.download_ucdp
            )
            if self.ucdp_agg is not None and not self.ucdp_agg.empty:
                data.append(self.ucdp_agg)
            self._cleanup()

        # Download and aggregate displacement datasets
        if "iom_dtm" in self.config["displacement_selected"]:
            logging.info("Loading IOM DTM data...")
            self.dtm, self.dtm_agg = self._download_with_aggregate(
                self.download_dtm, self.dtm_adm_level, filtered=True
            )
            if self.dtm_agg is not None and not self.dtm_agg.empty:
                data.append(self.dtm_agg)
            self._cleanup()

        if "idmc_gidd" in self.config["displacement_selected"]:
            logging.info("Loading IDMC GIDD data...")

            # Conflict-induced displacement
            self.idmc_gidd_conflict, self.idmc_gidd_conflict_agg = (
                self._download_with_aggregate(
                    self.download_idmc_gidd, cause="conflict", filtered=True
                )
            )
            if (
                self.idmc_gidd_conflict_agg is not None
                and not self.idmc_gidd_conflict_agg.empty
            ):
                data.append(self.idmc_gidd_conflict_agg)

            # Disaster-induced displacement
            self.idmc_gidd_disaster, self.idmc_gidd_disaster_agg = (
                self._download_with_aggregate(
                    self.download_idmc_gidd, cause="disaster", filtered=True
                )
            )
            if (
                self.idmc_gidd_disaster_agg is not None
                and not self.idmc_gidd_disaster_agg.empty
            ):
                data.append(self.idmc_gidd_disaster_agg)

            self.idmc_gidd_combined = self._combine_idmc_gidd(
                self.idmc_gidd_conflict, self.idmc_gidd_disaster
            )

        # Download UNHCR displacement data
        if "unhcr" in self.config["displacement_selected"]:
            logging.info("Loading UNHCR data...")
            self.unhcr = self.download_unhcr()

        # Download and process OSM datasets
        if self.config["osm_selected"]:
            logging.info("Downloading OSM data...")
            self.osm = self.download_osm()

            # Download OSM network data (roads, waterways)
            if "networks" in self.config["osm_selected"]:
                self.osm_networks = (
                    self._process_osm_data(osm_type="networks")
                    if self.osm is not None
                    else None
                )
            # Download OSM point of interest (POI) data (hospitals, schools)
            if "pois" in self.config["osm_selected"]:
                self.osm_pois = (
                    self._process_osm_data(osm_type="pois")
                    if self.osm is not None
                    else None
                )

        # Merge datasets and compute multihazard metrics
        logging.info("Calculating multihazard scores...")
        self.data = self._merge_data(data, columns=self.merge_columns)
        self.data = self._aggregate_idmc_idp(self.data)
        self.data = self._calculate_relative_exposure(self.data)
        self.data = self.calculate_multihazard_score(self.data)

    def download_geoboundary(
        self, adm_source: str, adm_level: str, overwrite: bool = False
    ) -> gpd.GeoDataFrame:
        """
        Download and prepare administrative boundaries from either:
            - GADM (http://gadm.org/)
            - geoBoundaries (https://www.geoboundaries.org/)

        Args:
            adm_source (str): Source of administrative boundaries ('gadm' or 'geoboundaries').
            adm_level (str): Administrative level to download (e.g., 'ADM1', 'ADM2').
            overwrite (bool): Whether to overwrite existing local files (default: False).

        Returns:
            gpd.GeoDataFrame: GeoDataFrame of administrative boundaries with ISO codes and valid geometries.
        """

        # Build output filenames
        out_file = self._build_filename(
            self.iso_code,
            f"{adm_source}_{adm_level}",
            self.local_dir,
            ext="geojson",
        )

        # Download if file does not exist or overwrite requested
        if overwrite or not os.path.exists(out_file):
            if adm_source == "gadm":
                geoboundary = self.download_gadm(adm_level)
            elif adm_source == "geoboundaries":
                geoboundary = self.download_geoboundaries(adm_level)
            geoboundary.to_crs(self.crs).to_file(out_file)

        # Load geoboundary and ensure CRS + geometry validity
        geoboundary = gpd.read_file(out_file, engine="pyogrio").to_crs(
            self.crs
        )
        geoboundary["iso_code"] = self.iso_code
        geoboundary["geometry"] = geoboundary.geometry.make_valid()

        # Build filename for admin file
        admin_file = self._build_filename(
            self.iso_code,
            adm_level,
            self.local_dir,
            ext="geojson",
        )

        # Assign grouping info (e.g., region, district)
        geoboundary, group = self._assign_grouping(
            self.iso_code, geoboundary, self.adm_config
        )

        # Save standardized admin file if not exists or overwrite
        if not overwrite and not os.path.exists(admin_file):
            geoboundary.to_crs(self.crs).to_file(admin_file, engine="pyogrio")
            logging.info(f"Geoboundary file saved to {admin_file}.")

        # Reload to ensure clean, standardized file
        geoboundary = gpd.read_file(admin_file, engine="pyogrio")

        # Update class-level attributes
        if adm_level == self.adm_level:
            self.adm_source, self.admin_file, self.group = (
                adm_source,
                admin_file,
                group,
            )
            self.merge_columns = list(geoboundary.columns)

        return geoboundary

    def download_gadm(
        self, adm_level: str = "ADM3", adm_source: str = "gadm"
    ) -> gpd.GeoDataFrame:
        """
        Download and format GADM administrative boundaries.
        Source: http://gadm.org/

        Args:
            adm_level (str): Administrative level to download (default: 'ADM3').
            adm_source (str): Source name (default: 'gadm')

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with standardized column names and geometries.
        """

        # Download raw GADM file from URL
        out_file = self.download_url(
            adm_source,
            dataset_name=f"{adm_source}_{adm_level}",
            ext="geojson",
        )
        geoboundary = gpd.read_file(out_file)

        # Prepare mapping to standardize column names
        rename = dict()
        for index in range(int(adm_level[-1]) + 1):
            if index == 0:
                rename[f"GID_{index}"] = "iso_code"
            else:
                rename[f"GID_{index}"] = f"ADM{index}_ID"
                rename[f"NAME_{index}"] = f"ADM{index}"

        # Rename columns and select only relevant fields
        geoboundary = geoboundary.rename(columns=rename)
        all_columns = list(rename.values()) + ["geometry"]
        geoboundary = geoboundary[all_columns]

        # Save the formatted GeoDataFrame
        geoboundary.to_file(out_file)

        return geoboundary

    def download_geoboundaries(
        self, adm_level: str = "ADM3", adm_source: str = "geoboundaries"
    ) -> gpd.GeoDataFrame:
        """
        Download and prepare geoBoundaries administrative boundaries.
        Source: https://www.geoboundaries.org/

        Args:
            adm_level (str): Administrative level to download (default: 'ADM3').
            adm_source (str): Source name (default: 'geoboundaries')

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with standardized column names and geometries.
        """

        # URLs for downloading geoBoundaries data
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

            # Download if intermediate file does not exist
            if not os.path.exists(intermediate_file):
                url = f"{gbhumanitarian_url}{self.iso_code}/{adm_level}/"
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    download_path = response.json()["gjDownloadURL"]
                except Exception:
                    url = f"{gbopen_url}{self.iso_code}/{adm_level}/"
                    response = requests.get(url)
                    response.raise_for_status()
                    download_path = response.json()["gjDownloadURL"]

                # Save downloaded GeoJSON locally
                geoboundary = requests.get(download_path).json()
                with open(intermediate_file, "w") as file:
                    geojson.dump(geoboundary, file)

            # Load and standardize columns
            geoboundary = gpd.read_file(intermediate_file)
            geoboundary["iso_code"] = self.iso_code

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
            datasets.append(geoboundary)
            geoboundary.to_file(intermediate_file)

        # Get UTM CRS for area calculations
        meter_crs = geoboundary.estimate_utm_crs()
        geoboundary = datasets[-1].to_crs(meter_crs)
        columns = geoboundary.columns

        # Perform spatial joins with lower-level boundaries
        for index in reversed(range(level - 1)):
            current = datasets[index].to_crs(meter_crs)
            join_columns = [
                f"ADM{index+1}_ID",
                f"ADM{index+1}",
                "geometry",
            ]
            joined = geoboundary.sjoin(
                current[join_columns], predicate="intersects"
            ).drop(columns=["index_right"])
            joined = joined.to_crs(meter_crs)

            # Calculate the intersection area and percentage overlap
            adm = join_columns[0]
            joined["intersection_area"] = joined.apply(
                lambda row: row.geometry.intersection(
                    current[current[adm] == row[adm]].iloc[0].geometry
                ).area,
                axis=1,
            )
            joined["overlap_percentage"] = (
                joined["intersection_area"] / joined["geometry"].area * 100
            )

            # Filter for the desired overlap percentage
            geoboundary = joined[joined["overlap_percentage"] >= 50]
            columns = list(columns) + list(join_columns[:-1])
            geoboundary = geoboundary[columns]

        return geoboundary

    def download_osm(
        self,
        country: str = None,
        out_dir: str = None,
        iso_code: str = None,
        overwrite: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Download and process OpenStreetMap (OSM) data for a country using QuackOSM.
        Source: https://github.com/kraina-ai/quackosm

        Args:
            country (str, optional): Country name to download (default: self.country).
            out_dir (str, optional): Directory to store downloaded files (default: self.local_dir).
            iso_code (str, optional): ISO code to map to a country name (default: self.iso_code).
            overwrite (bool): Whether to overwrite existing files (default: False).

        Returns:
            gpd.GeoDataFrame: Processed OSM data with parsed tags, or None if data unavailable.
        """

        # Use default country and output directory if not provided
        country = self.country if country is None else country
        out_dir = self.local_dir if out_dir is None else out_dir

        # Flatten list of available OSM countries and normalize
        osm_countries = []
        for key in sources.available.keys():
            osm_countries.extend(sources.available[key])
        osm_countries = [osm_country.lower() for osm_country in osm_countries]

        # Determine ISO code and map to country name
        self.osm_countries = osm_countries
        iso_code = iso_code or self.iso_code

        country_codes = self.osm_config["country_map"]
        if iso_code in country_codes:
            country = country_codes[iso_code]

        # Check if OSM data exists for this country
        country_name = country.lower().replace(" ", "_")
        if country not in osm_countries:
            found = False
            for osm_country in osm_countries:
                if country_name in osm_country:
                    found = True
                    country = osm_country
                    break
            if not found:
                logging.info(
                    f"{WARNING}WARNING: OSM does not exist for {self.country}.{RESET}"
                )
                return

        # Build output filename
        osm_file = self._build_filename(
            self.iso_code, "OSM", self.local_dir, ext="gpkg"
        )

        # Download and convert OSM data
        if overwrite or not os.path.exists(osm_file):
            filename = pyrosm.get_data(country, directory=out_dir)
            osm = pyrosm.OSM(filename)

            osm = qosm.convert_pbf_to_geodataframe(filename)
            osm.to_file(osm_file)

        # Load OSM data and parse tags
        osm = gpd.read_file(osm_file)
        osm["tags"] = osm["tags"].apply(lambda x: ast.literal_eval(x))

        return osm

    def _mask_osm_tags(self, data, osm_tags) -> gpd.GeoDataFrame:
        """
        Filter OSM data based on tags configuration.

        Args:
            data (gpd.GeoDataFrame): OSM data with a 'tags' column containing dictionaries.
            osm_tags (str, list, or dict): Tags to filter by. Can be a single key,
                a list of keys, or a dictionary mapping keys to allowed values.

        Returns:
            gpd.GeoDataFrame: Subset of `data` matching the specified tags.
        """

        # Helper function to generate boolean masks for dict-type tags
        def _get_dict_mask(data, tags):
            masks = []
            for osm_key, osm_values in tags.items():
                mask = data["tags"].apply(
                    lambda x: osm_key in x and x.get(osm_key) in osm_values
                )
                masks.append(mask)
            return masks

        # Handle different types of osm_tags (list of str/dict or single dict)
        masks = []
        if isinstance(osm_tags, list):
            for tag in osm_tags:
                if isinstance(tag, str):
                    mask = data["tags"].apply(lambda x: tag in x)
                    masks.append(mask)
                elif isinstance(tag, dict):
                    masks.extend(_get_dict_mask(data, tag))
        elif isinstance(osm_tags, dict):
            masks.extend(_get_dict_mask(data, osm_tags))

        # Combine masks using logical OR and filter data
        if masks:
            combined_mask = np.logical_or.reduce(masks)
            subdata = data[combined_mask]
        else:
            # No tags specified, return all data
            subdata = data.copy()

        return subdata

    def _process_osm_data(
        self, osm_type: str, overwrite: bool = False
    ) -> dict:
        """
        Process OSM data for a specific type (networks or points of interest).

        Args:
            osm_type (str): Type of OSM data to process ('networks' or 'pois').
            overwrite (bool): Whether to overwrite existing processed files (default: False).

        Returns:
            dict: Dictionary mapping each OSM subdata type to a GeoDataFrame
                  containing processed features with administrative boundary join.
        """

        # Define which geometry types correspond to the OSM type
        data_types = {
            "networks": ["LineString", "MultiLineString"],
            "pois": ["Point", "MultiPoint"],
        }

        # Filter OSM data by geometry type
        data = self.osm[self.osm.geom_type.isin(data_types[osm_type])]
        config = self.osm_config[f"osm_{osm_type}"]

        # Process each tag/type defined in configuration
        osm_data = dict()
        for data_type in tqdm(config, total=len(config)):
            osm_subdata_file = self._build_filename(
                self.iso_code, f"OSM_{data_type}", self.local_dir, ext="gpkg"
            )

            # Extract and save subdata
            if overwrite or not os.path.exists(osm_subdata_file):
                osm_tags = config[data_type]
                subdata = self._mask_osm_tags(data, osm_tags)
                subdata.to_file(osm_subdata_file)

            # Load subdata and assign tag
            subdata = gpd.read_file(osm_subdata_file)
            subdata["tag"] = data_type

            # Spatially join with admin boundaries
            subdata = subdata.to_crs(self.geoboundary.crs).sjoin(
                self.geoboundary, how="inner", predicate="intersects"
            )
            osm_data[data_type] = subdata

        return osm_data

    def download_unhcr(self):
        params = {
            "where": f"iso3='{self.iso_code}'",
            "outFields": "*",
            "f": "geojson",
        }

        try:
            base_url = self.config["urls"]["unhcr_url"]
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            geojson = response.json()

            features = geojson["features"]
        except:
            logging.info(
                f"{WARNING}WARNING: No UNHCR data failed to download for {self.country}.{RESET}"
            )
            return gpd.GeoDataFrame()

        geoms = [shapely.geometry.shape(feat["geometry"]) for feat in features]
        properties = [feat["properties"] for feat in features]

        gdf = gpd.GeoDataFrame(properties, geometry=geoms, crs="EPSG:4326")
        return gdf

    def download_idmc_gidd(
        self,
        cause: str = "conflict",
        name: str = "idmc",
        adm_level: str = None,
        filtered: bool = True,
        aggregate: bool = False,
        idp_col: str = "total_idps",
        idmc_idp_col: str = "Total figures",
        overwrite: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Download and process IDMC Global Internal Displacement Database (GIDD) data.
        Sources:
            https://www.internal-displacement.org/database/api-documentation/
            https://helix-tools-api.idmcdb.org/external-api/?#/

        Args:
            cause (str): Cause of displacement ('conflict' or 'disaster', default 'conflict').
            name (str): Dataset name prefix (default 'idmc').
            adm_level (str, optional): Administrative level for aggregation (default: manager's level).
            filtered (bool): Whether to split data by year (default True).
            aggregate (bool): Whether to aggregate data to ADM level (default False).
            idp_col (str): Column name for processed IDPs (default 'total_idps').
            idmc_idp_col (str): Column name in raw IDMC data (default 'Total figures').
            overwrite (bool): Whether to overwrite existing files (default False).

        Returns:
            dict or gpd.GeoDataFrame:
                - If `aggregate=False`, returns a dictionary of GeoDataFrames keyed by year or 'all'.
                - If `aggregate=True`, returns a GeoDataFrame aggregated by administrative unit.
        """

        # Build filename for raw IDMC GIDD data
        gidd_file = self._build_filename(
            self.iso_code, f"{name}_{cause}", self.local_dir, ext="geojson"
        )

        # Download IDMC GIDD data
        if overwrite or not os.path.exists(gidd_file):
            try:
                idmc_gidd_url = self.config["urls"]["idmc_gidd_url"].format(
                    self.idmc_key, self.iso_code, cause
                )

                self._download_url_progress(idmc_gidd_url, gidd_file)
            except:
                logging.info(
                    f"{WARNING}WARNING: IDMC data failed to download for {self.country}.{RESET}"
                )
                logging.info(
                    f"{WARNING}Please ensure you IDMC API key is correct.{RESET}"
                )
                return

            # Load and spatially join with admin boundaries
            idmc_gidd = gpd.read_file(gidd_file, use_arrow=True)
            idmc_gidd = idmc_gidd.sjoin(
                self.geoboundary, how="left", predicate="intersects"
            )
            idmc_gidd = idmc_gidd.drop(["index_right"], axis=1)
            idmc_gidd.to_file(gidd_file)

        # Load processed file
        idmc_gidd = gpd.read_file(gidd_file, use_arrow=True)
        if len(idmc_gidd) == 0:
            logging.info(
                f"{WARNING}WARNING: No IDMC {cause.title()} data for {self.country}.{RESET}"
            )
            return

        # Calculate number of points per geometry (handles MultiPoint)
        idmc_gidd["n_points"] = idmc_gidd.geometry.apply(
            lambda geom: (
                len(geom.geoms) if geom.geom_type == "MultiPoint" else 1
            )
        )

        # Normalize IDP counts per point
        # Source: https://helix-tools-api.idmcdb.org/external-api/?#/GIDD/gidd_disaggregations_disaggregation_geojson_retrieve
        if idmc_idp_col in idmc_gidd.columns:
            idmc_gidd[idp_col] = (
                idmc_gidd[idmc_idp_col] / idmc_gidd["n_points"]
            )
        idmc_gidd = idmc_gidd.explode()

        # Filter data by year
        idmc_gidd_dict = dict()
        if filtered:
            for displacement_year in range(
                self.displacement_start_year, self.displacement_end_year + 1
            ):
                filtered_file = self._build_filename(
                    self.iso_code,
                    f"{name}_{cause}_{displacement_year}",
                    self.local_dir,
                    ext="geojson",
                )
                if overwrite or not os.path.exists(filtered_file):
                    idmc_gidd_filtered = idmc_gidd.copy()

                    event_year_col = "Year"
                    if event_year_col in idmc_gidd_filtered.columns:
                        idmc_gidd_filtered = idmc_gidd_filtered[
                            idmc_gidd_filtered[event_year_col]
                            == int(displacement_year)
                        ]
                        idmc_gidd_filtered.to_file(filtered_file)

                idmc_gidd_year = gpd.read_file(filtered_file, use_arrow=True)
                if len(idmc_gidd_year) > 0:
                    idmc_gidd_dict[displacement_year] = idmc_gidd_year
        else:
            idmc_gidd_dict["all"] = idmc_gidd

        # Aggregate data to speficied admin level
        if aggregate:
            adm_level = adm_level or self.adm_level
            agg_file = self._build_filename(
                self.iso_code,
                f"{name}_{cause}_{adm_level}",
                self.local_dir,
                ext="geojson",
            )

            if overwrite or not os.path.exists(agg_file):
                idmc_gidd_agg = self.geoboundary.copy()
                for year, idmc_gidd in idmc_gidd_dict.items():
                    agg = self._aggregate_data(
                        idmc_gidd[[f"{adm_level}_ID", idp_col]],
                        agg_col=idp_col,
                        agg_func="sum",
                        adm_level=adm_level,
                    )
                    new_col = f"idmc_{cause}_idp_total_{year}"
                    agg = agg.rename(columns={idp_col: new_col})
                    idmc_gidd_agg = self._merge_data(
                        [idmc_gidd_agg, agg],
                        columns=[f"{adm_level}_ID"],
                        how="left",
                    )
                idmc_gidd_agg.to_file(agg_file)

            idmc_gidd_agg = gpd.read_file(agg_file)
            return idmc_gidd_agg

        return idmc_gidd_dict

    def _combine_idmc_gidd(
        self, idmc_conflict_dict, idmc_disaster_dict
    ) -> dict:
        """
        Combine IDMC GIDD conflict and disaster data by year.
        Merges the conflict and disaster GeoDataFrames for each year.

        Args:
            idmc_conflict_dict (dict): Dictionary of conflict GeoDataFrames keyed by year.
            idmc_disaster_dict (dict): Dictionary of disaster GeoDataFrames keyed by year.

        Returns:
            dict: Dictionary of combined GeoDataFrames keyed by year.
        """

        # Columns to retain in combined dataset
        columns = [
            "Total figures",
            "n_points",
            "total_idps",
            "Event cause",
            "geometry",
        ]

        idmc_combined = dict()
        conflict = idmc_conflict_dict or {}
        disaster = idmc_disaster_dict or {}

        # Process only years present in both datasets
        years = conflict.keys() & disaster.keys()
        for year in years:
            # Prepare conflict data
            conflicts = gpd.GeoDataFrame(columns=["geometry"])
            if idmc_conflict_dict[year] is not None:
                conflicts = idmc_conflict_dict[year].copy()
                if not conflicts.empty:
                    conflicts = conflicts[columns]

            # Prepare disaster data
            disasters = gpd.GeoDataFrame(columns=["geometry"])
            if idmc_disaster_dict is not None:
                disasters = idmc_disaster_dict[year].copy()
                if not disasters.empty:
                    disasters = disasters[columns]

            # Combine conflict and disaster data for the year
            idmc_combined[year] = gpd.GeoDataFrame(
                pd.concat([conflicts, disasters]), geometry="geometry"
            )

        return idmc_combined

    def _aggregate_idmc_idp(
        self,
        data,
        total_col="idmc_idp_total",
        mean_col="idmc_idp_mean",
        disaster_col="idmc_disaster_idp_total",
        conflict_col="idmc_conflict_idp_total",
    ) -> gpd.GeoDataFrame:
        """
        Aggregate IDMC IDP data across years.

        Computes yearly totals by summing conflict and disaster IDPs for each year,
        and calculates the mean across all available years.

        Args:
            data (gpd.GeoDataFrame or pd.DataFrame): Data containing IDMC IDP columns.
            total_col (str): Base name for total IDP columns (default 'idmc_idp_total').
            mean_col (str): Column name for the mean of yearly totals (default 'idmc_idp_mean').
            disaster_col (str): Base name for disaster IDP columns (default 'idmc_disaster_idp_total').
            conflict_col (str): Base name for conflict IDP columns (default 'idmc_conflict_idp_total').

        Returns:
            gpd.GeoDataFrame or pd.DataFrame: Input data with new total and mean IDP columns.
        """

        yearly_totals = []

        # Compute yearly totals by summing conflict and disaster IDPs
        for year in range(
            self.displacement_start_year, self.displacement_end_year + 1
        ):
            conflict_col_year = f"{conflict_col}_{year}"
            disaster_col_year = f"{disaster_col}_{year}"
            total_col_year = f"{total_col}_{year}"

            idp_cols = [
                col
                for col in [conflict_col_year, disaster_col_year]
                if col in data.columns
            ]

            data[total_col_year] = data[idp_cols].sum(axis=1)
            yearly_totals.append(total_col_year)

        # Compute mean across all yearly totals
        if yearly_totals:
            data[mean_col] = data[yearly_totals].mean(axis=1)

        return data

    def download_dtm(
        self,
        dtm_adm_level: str = "ADM2",
        idp_column: str = "numPresentIdpInd",
        year: int = None,
        filtered: bool = False,
        aggregate: bool = False,
        overwrite: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Download and process DTM IDP data for a country.

        Args:
            dtm_adm_level (str): Admin level to download ('ADM1' or 'ADM2', default 'ADM2').
            idp_column (str): Column name for IDP counts (default 'numPresentIdpInd').
            year (int, optional): Year to filter data for (default None, uses latest available).
            filtered (bool): Whether to filter data by year and reporting round (default False).
            aggregate (bool): Whether to aggregate data to admin boundaries (default False).
            overwrite (bool): Whether to overwrite existing downloaded or processed files (default False).

        Returns:
            gpd.GeoDataFrame or pd.DataFrame: Filtered or aggregated DTM data, or raw DataFrame if not aggregated.
        """

        # Build file paths
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
        geojson_file = self._build_filename(
            self.iso_code,
            f"DTM_{dtm_adm_level}",
            self.local_dir,
            ext="geojson",
        )

        # Load existing files
        data = None
        if os.path.exists(filtered_file):
            if not aggregate:
                data = pd.read_csv(filtered_file)
            elif os.path.exists(geojson_file):
                data = gpd.read_file(geojson_file)

        # Exit if no API key
        if self.dtm_key is None:
            return data
        else:
            # Initialize DTM API
            try:
                api = DTMApi(subscription_key=self.dtm_key)
                self.dtm_countries = api.get_all_countries()
            except:
                logging.info(
                    f"{WARNING}WARNING: Network connection to dtm.iom.int could not be established.{RESET}"
                )
                return data

        # Download DTM IDP data
        if (
            overwrite
            or not os.path.exists(raw_file)
            or not os.path.exists(filtered_file)
        ):
            try:
                country_name = self.dtm_countries[
                    self.dtm_countries["admin0Pcode"] == self.iso_code
                ]["admin0Name"].values[0]
            except:
                logging.info(
                    f"{WARNING}WARNING: No DTM data available for {self.country}.{RESET}"
                )
                return

            # Ensure administrative boundaries are available
            try:
                adm = self.download_geoboundary(
                    adm_source="geoboundaries",
                    adm_level=dtm_adm_level,
                    overwrite=False,
                )
            except Exception as e:
                logging.info(e)
                adm = self.download_geoboundary(
                    adm_source="gadm", adm_level=dtm_adm_level
                )

            # Define reporting period
            dtm = None
            displacement_start_date = f"{self.displacement_start_year}-01-01"
            displacement_end_date = f"{self.displacement_end_year}-12-31"

            if dtm_adm_level == "ADM1":
                dtm = api.get_idp_admin1_data(
                    CountryName=country_name,
                    FromReportingDate=displacement_start_date,
                    ToReportingDate=displacement_end_date,
                )

            elif dtm_adm_level == "ADM2":
                dtm = api.get_idp_admin2_data(
                    CountryName=country_name,
                    FromReportingDate=displacement_start_date,
                    ToReportingDate=displacement_end_date,
                )

            if len(dtm) == 0:
                logging.info(
                    f"{WARNING}WARNING: No DTM data available for {self.country}.{RESET}"
                )
                return

            dtm.to_csv(raw_file)

            # Filter by year and latest reporting round
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

        # Load raw or filtered data
        dtm = pd.read_csv(raw_file)
        if filtered:
            dtm = pd.read_csv(filtered_file)

        # Aggregate to admin level
        if aggregate:
            dtm_adm_level_num = dtm_adm_level[-1]
            column = f"admin{dtm_adm_level_num}Name"
            dtm_agg = self._aggregate_data(
                dtm[[column, idp_column]],
                agg_col=idp_column,
                agg_func="sum",
                adm_level=column,
            )
            dtm_agg = dtm_agg.rename(columns={idp_column: "dtm_idp_total"})

            # Merge with geoboundary and save
            adm = self.geoboundary
            dtm_agg = adm.merge(
                dtm_agg, left_on=dtm_adm_level, right_on=column, how="left"
            )
            dtm_agg.to_crs(self.crs).to_file(geojson_file)
            dtm_agg = gpd.read_file(geojson_file)

            return dtm_agg

        return dtm

    def download_ucdp(
        self,
        aggregate: bool = False,
        ucdp_name: str = "ucdp",
        overwrite: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Download and process UCDP conflict data for a country.

        Args:
            aggregate (bool): Whether to aggregate data by administrative boundaries (default False).
            ucdp_name (str): Name of the UCDP dataset (default 'ucdp').
            overwrite (bool): Whether to overwrite existing local or global files (default False).

        Returns:
            gpd.GeoDataFrame: Filtered and optionally aggregated UCDP data.
        """

        # Build file paths
        local_file = self._build_filename(
            self.iso_code, ucdp_name, self.local_dir, ext="geojson"
        )
        global_file = self._build_filename(
            self.global_name, ucdp_name, self.global_dir, ext="csv"
        )

        # Download global UCDP data
        if overwrite or not os.path.exists(global_file):
            try:
                dataset = f"{self.global_name}_{ucdp_name}".lower()
                self.download_url(dataset=dataset, ext="csv")
            except:
                logging.info(
                    f"{WARNING}WARNING: UCDP Data failed to download for {self.country}.{RESET}"
                )
                return

        # Process local data
        if overwrite or not os.path.exists(local_file):
            ucdp = pd.read_csv(global_file, low_memory=False)
            ucdp["country"] = ucdp["country"].apply(
                lambda x: re.sub(r"\s*\([^)]*\)", "", x).strip()
            )

            # Handle country name special cases
            country = self.country
            if self.iso_code == "COD":
                country = "DR Congo"

            # Filter by country and conflict dates
            ucdp = ucdp[ucdp["country"] == country]
            ucdp["date_start"] = pd.to_datetime(ucdp["date_start"])
            ucdp = ucdp[ucdp["date_start"] >= self.conflict_start_date]
            ucdp = ucdp[ucdp["date_start"] <= self.conflict_end_date]

            # Map type_of_violence codes to descriptive strings
            type_of_violence_map = {
                1: "State-based conflict",
                2: "Non-state conflict",
                3: "One-sided violence",
            }
            ucdp["type_of_violence"] = ucdp["type_of_violence"].replace(
                type_of_violence_map
            )

            if len(ucdp) == 0:
                logging.info(
                    f"{WARNING}WARNING: No UCDP data found for {self.iso_code}.{RESET}"
                )
                return

            ucdp = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(
                    ucdp["longitude"], ucdp["latitude"], crs=self.crs
                ),
                data=ucdp,
            )

            # Spatial join with administrative boundaries
            ucdp = ucdp.sjoin(
                self.geoboundary, how="left", predicate="intersects"
            )
            ucdp = ucdp.drop(["index_right"], axis=1)

            # Save to local GeoJSON
            ucdp.to_file(local_file, driver="GeoJSON")
            logging.info(f"UCDP file saved to {local_file}")

        # Load processed data
        ucdp = gpd.read_file(local_file)

        # Aggregate if requested
        if aggregate:
            ucdp = self._aggregate_ucdp(ucdp, local_file)

        return ucdp

    def _aggregate_ucdp(
        self,
        ucdp,
        local_file: str,
        ucdp_name: str = "ucdp",
        overwrite: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Aggregate UCDP conflict data with assets and administrative boundaries.

        Args:
            ucdp (gpd.GeoDataFrame): UCDP conflict events.
            local_file (str): Path to local UCDP GeoJSON file.
            ucdp_name (str): Dataset name (default 'ucdp').
            overwrite (bool): Whether to overwrite existing files (default False).

        Returns:
            gpd.GeoDataFrame: Aggregated UCDP conflict data with exposures and statistics.
        """

        ucdp_agg = None
        admin = self.geoboundary

        # Loop through asset layers and calculate exposure
        for asset_name, asset_file in (
            pbar := tqdm(
                zip(self.asset_names, self.asset_files),
                total=len(self.asset_names),
            )
        ):
            pbar.set_description(f"Processing {asset_name}")
            column = f"{ucdp_name}_{asset_name}_exposure"
            exposure_raster = self._build_filename(
                self.iso_code,
                f"{ucdp_name}_{asset_name}_exposure",
                self.local_dir,
                ext="tif",
            )

            # Build filenames for raster and vector outputs
            exposure_vector = self._build_filename(
                self.iso_code,
                f"{ucdp_name}_{asset_name}_exposure_{self.adm_level}",
                self.local_dir,
                ext="geojson",
            )

            # Compute exposure
            if overwrite or not os.path.exists(exposure_vector):
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
                    overwrite=overwrite,
                )

            # Read exposure vector and clean zero values
            ucdp_agg_sub = gpd.read_file(exposure_vector)
            ucdp_agg_sub.loc[ucdp_agg_sub[column] == 0, column] = None

            # Merge with cumulative exposure
            ucdp_agg = (
                ucdp_agg_sub
                if ucdp_agg is None
                else self._merge_data(
                    [ucdp_agg, ucdp_agg_sub], columns=self.merge_columns
                )
            )

        # Final exposure output path
        final_exposure_vector = self._build_filename(
            self.iso_code,
            f"{ucdp_name}_exposure_{self.adm_level}",
            self.local_dir,
            ext="geojson",
        )

        # Aggregate total conflict events per admin unit
        column = "total_conflict_count"
        event_count = self._aggregate_data(
            ucdp, agg_col=column, agg_func="count"
        )
        event_count = event_count.rename(columns={column: f"ucdp_{column}"})
        event_count = self._merge_data(
            [admin, event_count],
            columns=[f"{self.adm_level}_ID"],
            how="left",
        )

        # Aggregate total fatalities
        fatalities_count = self._aggregate_data(
            ucdp, agg_col="best", agg_func="sum"
        )
        fatalities_count = fatalities_count.rename(
            columns={"best": "ucdp_total_fatalities"}
        )
        fatalities_count = self._merge_data(
            [admin, fatalities_count],
            columns=[f"{self.adm_level}_ID"],
            how="left",
        )

        # Merge all aggregated data
        ucdp = self._merge_data(
            [event_count, fatalities_count, ucdp_agg],
            columns=self.merge_columns,
        )

        # Compute additional conflict statistics
        self._calculate_conflict_stats(ucdp, source="ucdp")

        # Save final aggregated exposure
        ucdp.to_file(final_exposure_vector)

        return ucdp

    def download_acled(
        self,
        population: str = "full",
        aggregate: bool = False,
        exposure_column: str = "population_best",
        acled_name: str = "acled",
        overwrite: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Download and process ACLED conflict event data for a country.

        Args:
            population (str): Population field to use (default 'full').
            aggregate (bool): Whether to aggregate exposure by asset/admin (default False).
            exposure_column (str): Column in ACLED for exposure values (default 'population_best').
            acled_name (str): Dataset name (default 'acled').
            overwrite (bool): Whether to overwrite existing local files (default False).

        Returns:
            gpd.GeoDataFrame or dict:
                Aggregated GeoDataFrame if aggregate=True,
                or dictionary of GeoDataFrames per asset otherwise.
        """

        # Build file paths
        acled_dict = dict()
        raw_file = self._build_filename(
            self.iso_code, acled_name, self.local_dir, ext="geojson"
        )
        acled_agg_file = self._build_filename(
            self.iso_code,
            f"{acled_name}_{self.adm_level}",
            self.local_dir,
            ext="geojson",
        )

        # Download ACLED data
        if overwrite or not os.path.exists(raw_file):
            # Get an access token
            acled_token = self._get_acled_access_token(
                username=self.acled_username,
                password=self.acled_password,
                token_url="https://acleddata.com/oauth/token",
            )
            if acled_token is None:
                return

            params = dict(
                country=self.country,
                event_date=f"{self.conflict_start_date}|{self.conflict_end_date}",
                event_date_where="BETWEEN",
                page=1,
            )
            headers = {
                "Authorization": f"Bearer {acled_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            # Paginate through ACLED API results
            acled_url = self.config["urls"]["acled_url"]
            len_subdata = -1
            data = []

            global_file = self._build_filename(
                self.global_name,
                acled_name,
                self.global_dir,
                ext="csv",
            )

            if os.path.exists(global_file):
                data = pd.read_csv(global_file, engine="pyarrow")
                data = data[data.country == self.country]

                data["event_date"] = pd.to_datetime(data["event_date"])
                data = data[data["event_date"] >= self.conflict_start_date]
                data = data[data["event_date"] <= self.conflict_end_date]

            else:
                # Paginate through ACLED API results
                while len_subdata != 0:
                    logging.info(f"Reading ACLED page {params['page']}...")
                    response = requests.get(
                        acled_url, params=params, headers=headers
                    )
                    response.raise_for_status()

                    subdata = pd.DataFrame(response.json()["data"])
                    data.append(subdata)
                    len_subdata = len(subdata)
                    params["page"] = params["page"] + 1

                data = pd.concat(data)

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

        if len(self.acled_raw) == 0:
            logging.info(
                f"{WARNING}WARNING: No ACLED data returned for {self.country}.{RESET}"
            )
            return

        # Return aggregated data if already exists
        if aggregate and os.path.exists(acled_agg_file):
            return gpd.read_file(acled_agg_file)

        # Process each asset layer
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
                f"{acled_name}_{asset_name}_FILTERED",
                self.local_dir,
                ext="geojson",
            )

            # Filter ACLED events relevant to this asset
            asset_category = self.get_asset_category(asset_name)
            acled = self._filter_acled(
                self.acled_raw,
                self.acled_selected[asset_category],
                filtered_file,
            )

            # Aggregate if requested
            if aggregate:
                agg_file = self._build_filename(
                    self.iso_code,
                    f"{acled_name}_{asset_name}_{self.adm_level}",
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

        # Merge all aggregated data if any
        if len(full_data) > 0:
            if overwrite or not os.path.exists(acled_agg_file):
                acled = self._merge_data(full_data, columns=self.merge_columns)
                acled.to_file(acled_agg_file)
            acled = gpd.read_file(acled_agg_file)
            return acled

        return acled_dict

    def get_asset_category(self, asset_name: str):
        """
        Return the ACLED category given the asset name.

        Args:
            asset_name(str): Name of the asset

        Returns:
            str: Asset category (e.g. infrastructure, agriculture, etc.)
        """
        # Identify asset category
        asset_category = [
            key
            for key, values in self.asset_categories.items()
            if asset_name in values
        ]
        if not asset_category:
            logging.info(
                f"{WARNING}WARNING: {asset_name} not in any ACLED asset category list.{RESET}"
            )
            return
        return asset_category[0]

    def _get_acled_access_token(
        self, username: str, password: str, token_url: str
    ):
        """
        Obtain an access token for the ACLED API using user credentials.

        Args:
            username (str): ACLED account username.
            password (str): ACLED account password.
            token_url (str): URL to request the access token from ACLED.

        Returns:
            str: ACLED API access token.

        Raises:
            Exception: If the access token request fails.
        """

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "username": username,
            "password": password,
            "grant_type": "password",
            "client_id": "acled",
        }

        response = requests.post(token_url, headers=headers, data=data)

        if response.status_code == 200:
            token_data = response.json()
            return token_data["access_token"]
        else:
            logging.info(
                f"{WARNING}WARNING: Failed to get access token: {response.status_code} {response.text} {RESET}"
            )
            return None

    def _filter_acled(
        self,
        data: pd.DataFrame,
        hierarchy: dict = None,
        out_file: str = None,
        overwrite: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Filter ACLED events based on a hierarchy of disorder and event types.

        Args:
            data (pd.DataFrame): Raw ACLED event data.
            hierarchy (dict, optional): Hierarchy of disorder, event, and sub-event types. (default: self.acled_hierarchy)
            out_file (str, optional): Path to save filtered GeoDataFrame. If None, data is not saved.
            overwrite (bool): Whether to overwrite existing filtered file (default: False).

        Returns:
            gpd.GeoDataFrame: Filtered ACLED events as a GeoDataFrame.
        """

        valid_rows = []

        # Only filter if overwriting or output file does not exist
        if overwrite or not os.path.exists(out_file):
            hierarchy = hierarchy or self.acled_hierarchy

            # Build list of all valid (disorder_type, event_type, sub_event_type) tuples
            for disorder_type, event_dict in hierarchy.items():
                for event_type, sub_events in event_dict.items():
                    for sub_event in sub_events:
                        valid_rows.append(
                            (disorder_type, event_type, sub_event)
                        )

            # Create a DataFrame of valid events
            columns = ["disorder_type", "event_type", "sub_event_type"]
            valid_df = pd.DataFrame(valid_rows, columns=columns)

            # Merge with raw data to keep only valid events
            filtered = data.merge(
                valid_df,
                on=columns,
                how="inner",
            )

            # Save filtered file
            filtered.to_file(out_file)

        # Read the filtered GeoDataFrame
        filtered = gpd.read_file(out_file)

        return filtered

    def _aggregate_acled(
        self,
        acled_file: str,
        agg_file: str,
        asset_name: str,
        asset_file: str,
        prefix: str = "wbg",
        acled_name: str = "acled",
        overwrite: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Aggregate ACLED event data for a specific asset and calculate exposure.

        Args:
            acled_file (str): Path to filtered ACLED event GeoDataFrame.
            agg_file (str): Path to save aggregated ACLED exposure GeoDataFrame.
            asset_name (str): Name of the asset being analyzed.
            asset_file (str): Path to asset data (raster or vector) for exposure calculation.
            prefix (str): Prefix for exposure column names (default 'wbg').
            acled_name (str): Base name for ACLED files (default 'acled').
            overwrite (bool): Whether to overwrite existing output files (default False).

        Returns:
            gpd.GeoDataFrame: Aggregated ACLED exposure data for the asset.
        """

        # Load ACLED events
        acled = gpd.read_file(acled_file)

        if overwrite or not os.path.exists(agg_file):
            # Aggregate exposure for the asset
            agg = self._aggregate_acled_exposure(acled, asset_name)
            full_data = [agg]

            # Build file paths for raster and vector exposure outputs
            exposure_raster = self._build_filename(
                self.iso_code,
                f"{acled_name}_{asset_name}_exposure",
                self.local_dir,
                ext="tif",
            )
            exposure_vector = self._build_filename(
                self.iso_code,
                f"{acled_name}_{asset_name}_exposure_{self.adm_level}",
                self.local_dir,
                ext="geojson",
            )

            # Calculate exposure if file doesn't exist or overwrite is True
            column = f"{acled_name}_{asset_name}_exposure"
            if overwrite or not os.path.exists(exposure_vector):
                acled_tif = self._calculate_custom_conflict_exposure(
                    acled_file,
                    asset_file,
                    asset_name=asset_name,
                    conflict_src="acled",
                )
                out_tif, _ = self._calculate_exposure(
                    asset_file, acled_tif, exposure_raster, threshold=1
                )

                # Compute zonal statistics for exposure
                self._calculate_zonal_stats(
                    out_tif,
                    column=column,
                    prefix=prefix,
                    stats_agg=["sum"],
                    out_file=exposure_vector,
                    overwrite=overwrite,
                )

            # Read exposure vector and clean zero values
            exposure_var = prefix + "_" + column
            exposure = gpd.read_file(exposure_vector)
            exposure.loc[exposure[exposure_var] == 0, exposure_var] = None
            full_data.append(exposure)

            # Merge aggregated exposure data with administrative boundaries
            acled = self._merge_data(full_data, columns=self.merge_columns)
            acled.to_file(agg_file)

        # Read aggregated ACLED data
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
        overwrite: bool = False,
    ) -> str:
        """
        Creates a buffered raster representation of conflict events given a specific asset.

        Args:
            conflict_file (str): Path to the conflict event GeoDataFrame (GeoJSON).
            asset_file (str): Path to asset raster file used as a template for rasterization.
            asset_name (str): Name of the asset being analyzed.
            conflict_src (str): Source of conflict data ('acled' or 'ucdp').
            temp_name (str): Temporary filename identifier (default 'temp').
            buffer_size (int): Default buffer size in meters (used for UCDP) (default: 3000).
            overwrite (bool): Whether to overwrite existing files (default False).

        Returns:
            str: Path to the rasterized conflict exposure GeoTIFF.
        """

        # Check that the conflict file exists
        if not os.path.exists(conflict_file):
            raise FileNotFoundError(
                f"conflict file not found: {conflict_file}"
            )

        # Helper function to determine buffer size based on event type and fatalities
        def _get_buffer_size(event, fatality):
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
        if overwrite or not os.path.exists(temp_file):
            data = gpd.read_file(conflict_file)
            data["values"] = 1

            # Get buffer size depending on conflict data source
            if conflict_src == "acled":
                data["buffer_size"] = data.apply(
                    lambda x: _get_buffer_size(x.event_type, x.fatalities),
                    axis=1,
                )
            elif conflict_src == "ucdp":
                data["buffer_size"] = buffer_size

            # Apply buffer using meter CRS
            meter_crs = data.estimate_utm_crs()
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
        if overwrite or not os.path.exists(out_file):
            # Create empty raster based on asset template
            with rio.open(asset_file) as src:
                out_image = src.read(1)
                out_image = np.zeros(out_image.shape)

                out_meta = src.meta.copy()
                with rio.open(out_file, "w", **out_meta) as dest:
                    dest.write(out_image, 1)

            os.system(f"gdal_rasterize -at -burn 1 {temp_file} {out_file}")

        return out_file

    def _aggregate_acled_exposure(
        self, acled: gpd.GeoDataFrame, asset: str
    ) -> gpd.GeoDataFrame:
        """
        Aggregate ACLED conflict events and compute exposure for a given asset.

        Args:
            acled (gpd.GeoDataFrame): ACLED conflict events with geometry and relevant attributes.
            asset (str): Name of the asset to compute exposure for.

        Returns:
            gpd.GeoDataFrame: Aggregated conflict exposure data joined with admin boundaries.
        """

        # Helper function to sum while ignoring all-NaN arrays
        def _nansumwrapper(a, **kwargs):
            if np.isnan(a).all():
                return np.nan
            else:
                return np.nansum(a, **kwargs)

        admin = self.geoboundary
        data = [admin]

        # Aggregate total conflict events
        event_count = self._aggregate_data(
            acled, agg_col="conflict_count", agg_func="count"
        )
        event_count = event_count.rename(
            columns={"conflict_count": f"acled_{asset}_conflict_count"}
        )
        data.append(event_count)

        # Aggregate total conflict events
        fatalities_count = self._aggregate_data(
            acled, agg_col="fatalities", agg_func="sum"
        )
        fatalities_count = fatalities_count.rename(
            columns={"fatalities": f"acled_{asset}_fatalities"}
        )
        data.append(fatalities_count)

        if "population_best" in acled.columns:
            # Aggregate population sum
            pop_sum = self._aggregate_data(
                acled,
                agg_col="population_best",
                agg_func=lambda x: _nansumwrapper(x),
            )
            pop_sum = pop_sum.rename(
                columns={"population_best": f"acled_{asset}_population_best"}
            )
            data.append(pop_sum)

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
            data.append(null_pop_event_count)

        # Merge all aggregated data with admin boundaries
        acled = self._merge_data(
            data,
            columns=[f"{self.adm_level}_ID"],
            how="left",
        )

        # Calculate population-weighted conflict exposure
        if "population_best" in acled.columns:
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

    def _calculate_conflict_stats(
        self, data, source, asset: str = "total"
    ) -> gpd.GeoDataFrame:
        """
        Calculate conflict statistics: fatalities per conflict.

        Args:
            data (gpd.GeoDataFrame): GeoDataFrame containing conflict data.
            source (str): Source of the conflict data (e.g., "acled" or "ucdp").
            asset (str, optional): Specific asset or category to calculate stats for. (default: "total")

        Returns:
            gpd.GeoDataFrame: Updated GeoDataFrame with additional statistics columns.
        """
        col_base = f"{source}_{asset}"
        per_conflict = f"{col_base}_fatalities_per_conflict"

        data[per_conflict] = (
            data[f"{col_base}_fatalities"] / data[f"{col_base}_conflict_count"]
        ).replace([np.inf, -np.inf], np.nan)

        return data

    def _download_url_progress(self, url: str, output_path: str):
        """
        Download a file from a URL with a progress bar.

        Args:
            url (str): URL of the file to download.
            output_path (str): Local path where the downloaded file will be saved.
        """

        desc = os.path.basename(output_path)

        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=desc
        ) as t:
            urllib.request.urlretrieve(
                url, filename=output_path, reporthook=t.update_to
            )

    def download_url(
        self,
        dataset: str,
        dataset_name: str = None,
        ext: str = "tif",
        overwrite: bool = False,
    ) -> str:
        """Download a dataset from a URL.

        This method handles ZIP, NetCDF, and GeoTIFF files.
        Global datasets are clipped to the country boundary after download.

        Args:
            dataset (str): Key identifying the dataset in the configuration.
            dataset_name (str, optional): Name to use for saving the dataset locally (default: None)
            ext (str, optional): File extension for the dataset. (default: tif)
            overwrite (bool, optional): Whether to overwrite existing files. (default: False)

        Returns:
            str: Path to the local dataset file.
        """

        if dataset_name is None:
            dataset_name = dataset.replace(f"{self.global_name.lower()}_", "")

        global_file = self._build_filename(
            self.global_name, dataset_name, self.global_dir, ext=ext
        )

        url_name = f"{dataset}_url"
        if url_name in self.config["urls"]:
            url = self.config["urls"][url_name]
            if dataset == "gadm":
                url = self.config["urls"][url_name].format(
                    self.iso_code, self.adm_level[-1]
                )
            elif "gwis" in url_name:
                date_today = datetime.date.today().strftime("%Y-%m-%d")
                url = self.config["urls"][url_name].format(date_today)
            elif "geos5" not in url_name:
                url = self.config["urls"][url_name].format(
                    self.iso_code, self.iso_code.lower()
                )

        # Check if the dataset is global
        if self.global_name.lower() in dataset:
            # Download if not already present
            if overwrite or not os.path.exists(global_file):
                logging.info(f"Downloading {url}...")
                if url.endswith(".zip"):
                    self.download_zip(
                        url, dataset, out_file=global_file, ext=ext
                    )
                elif url.endswith(".nc"):
                    self.download_netcdf(
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
            if overwrite or not os.path.exists(local_file):
                if url.endswith(".zip"):
                    self.download_zip(
                        url, dataset, out_file=local_file, ext=ext
                    )
                elif url.endswith(".tif") or (".tif" in url):
                    self._download_url_progress(url, local_file)

        return local_file

    def download_netcdf(
        self, url: str, dataset: str, out_file: str, ext: str = "tif"
    ) -> str:
        """Download a NetCDF dataset, convert monthly files to GeoTIFFs, and merge into a single raster.

        Args:
            url (str): URL template for the NetCDF files, with a placeholder for month.
            dataset (str): Name of the dataset being downloaded.
            out_file (str): Path to the final merged output GeoTIFF.
            ext (str, optional): File extension for the output file. (default: "tif")

        Returns:
            str: Path to the final merged GeoTIFF file.
        """

        out_dir = (
            self.global_dir
            if self.global_name.lower() in dataset
            else self.local_dir
        )
        nc_dir = os.path.join(out_dir, dataset)
        os.makedirs(nc_dir, exist_ok=True)

        files = []
        for index in range(1, 13):
            num = "0" + str(index) if len(str(index)) == 1 else str(index)
            nc_file = os.path.join(nc_dir, f"{dataset.upper()}_{num}.nc")
            tif_file = nc_file.replace(".nc", ".tif")
            if not os.path.exists(tif_file):
                self._download_url_progress(url.format(num), nc_file)
                os.system(
                    f'gdal_translate -a_srs EPSG:4326 NETCDF:"{nc_file}":GEOS-5_FWI {tif_file}'
                )
            files.append(tif_file)

        with rio.open(files[0]) as src0:
            profile = src0.profile
            max_arr = src0.read(1).astype(np.float32)

            for f in files:
                with rio.open(f) as src:
                    arr = src.read(1)
                    np.maximum(max_arr, arr, out=max_arr)

        profile.update(dtype="float32")

        with rio.open(out_file, "w", **profile) as dst:
            dst.write(max_arr.astype(np.float32), 1)

        return out_file

    def download_zip(
        self, url: str, dataset: str, out_file: str, ext: str = "tif"
    ) -> None:
        """Download a ZIP file, extract its contents, and convert to the desired format.

        Args:
            url (str): URL to download the ZIP file.
            dataset (str): Name of the dataset.
            out_file (str): Path to save the final output file.
            ext (str, optional): Desired output file format. Options: "tif", "geojson", "csv". (default: "tif")

        Returns:
            None: The function writes the processed file to `out_file`.
        """

        # Decide output directory (global vs local)
        out_dir = (
            self.global_dir
            if self.global_name.lower() in dataset
            else self.local_dir
        )
        zip_file = os.path.join(out_dir, f"{dataset.upper()}.zip")
        zip_dir = os.path.join(out_dir, dataset)

        # Download and extract ZIP if not already done
        if not os.path.exists(zip_file) and not os.path.exists(zip_dir):
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
            elif len(tif_files) > 1:
                tif_file = zip_file.replace(".zip", ".tif")
                vrt_file = tif_file.replace(".tif", ".vrt")
                self._merge_tifs(f"{zip_dir}/*.tif", vrt_file, tif_file)
            else:
                tif_file = tif_files[0]

            os.system(
                f"gdal_translate -a_srs EPSG:4326 -co TILED=YES -co COMPRESS=LZW -co BIGTIFF=YES -co NUM_THREADS=ALL_CPUS --config GDAL_CACHEMAX 512 {os.path.join(zip_dir, tif_file)} {out_file}"
            )
            shutil.rmtree(zip_dir)

        elif ext == "geojson":
            geojson_files = [
                file
                for file in os.listdir(zip_dir)
                if file.endswith(".geojson")
            ]

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

            else:
                geojson_file = geojson_files[0]
                shutil.copyfile(os.path.join(zip_dir, geojson_file), out_file)

            shutil.rmtree(zip_dir)

        elif ext == "csv":
            csv_files = [
                file for file in os.listdir(zip_dir) if file.endswith(".csv")
            ]
            shutil.copyfile(os.path.join(zip_dir, csv_files[0]), out_file)

    def download_fathom(
        self,
        name: str,
        ext: str = "tif",
        fathom_year: int = 2020,
        fathom_rp: int = 100,
        fathom_name: str = "fathom",
        overwrite: bool = False,
    ) -> str:
        """Download and process Fathom flood hazard data for a specific country.

        Args:
            name (str): Name of the Fathom dataset (e.g., "flood_depth").
            ext (str, optional): File extension/format for the output raster. (default: "tif")
            fathom_year (int, optional): Year of the Fathom data. (default: 2020)
            fathom_rp (int, optional): Return period of the flood event (e.g., 100-year flood). (default: 100).
            fathom_name (str, optional): Base directory name for Fathom data. (default: "fathom").
            overwrite (bool, optional): Whether to overwrite existing processed files. (default: False)

        Returns:
            str: Path to the processed and clipped Fathom raster file for the country.
        """
        fathom_dir = os.path.join(self.local_dir, fathom_name)

        # If processed dataset doesn't exist, generate it
        raw_file = os.path.join(fathom_dir, f"{name.upper()}.{ext}")
        local_file = os.path.join(
            self.local_dir, f"{self.iso_code.upper()}_{name.upper()}.{ext}"
        )

        # If processed file doesn't exist, build from VRT
        if overwrite or not os.path.exists(local_file):
            flood_dir = os.path.join(
                fathom_dir,
                name.replace("_" + fathom_name, "").upper(),
                str(fathom_year),
                f"1in{fathom_rp}",
            )
            merged_file = os.path.join(fathom_dir, f"{name.upper()}.vrt")
            self._merge_tifs(f"{flood_dir}/*.{ext}", merged_file, raw_file)

            # Clip raster to admin boundary
            admin = self.geoboundary.dissolve(by="iso_code")
            nodata = self.config["nodata"][name.lower()]
            self._clip_raster(raw_file, local_file, admin, nodata)

        return local_file

    def download_jrc(
        self,
        name: str,
        jrc_rp: int = 100,
    ) -> str:
        """Download and process JRC global flood hazard data for a specific country.

        Args:
            name (str): Name of the JRC dataset (e.g., "flood_depth").
            jrc_rp (int, optional): Return period of the flood event (e.g., 100-year flood). (default: 100)

        Returns:
            str: Path to the processed and clipped JRC raster file for the country.
        """
        out_dir = os.path.join(self.global_dir, name)
        os.makedirs(out_dir, exist_ok=True)

        url_name = f"{name}_url"
        url = self.config["urls"][url_name].format(jrc_rp)
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
            name.replace(f"{self.global_name.lower()}_", ""),
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
        """Download and process ESA WorldCover land cover data for a country.

        Args:
            land_cover_class (str): The land cover class to extract (e.g., "forest", "urban").
            name (str, optional): Dataset name (default: "worldcover")
            year (int, optional): Year of the WorldCover dataset (2020 or 2021) (default: 2021)
            resample (bool, optional): Whether to resample the raster to match another dataset (default: True)
            overwrite (bool, optional): Whether to overwrite existing files (default: False)

        Returns:
            str: Path to the processed raster file for the specified land cover class and country.
        """
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
                        response = requests.get(url, allow_redirects=True)
                        response.raise_for_status()
                        with open(filename, "wb") as f:
                            f.write(response.content)

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
        """
        Create a binary mask of a raster based on a specified class code.

        Args:
            raster_file (str): Path to the input raster file.
            out_file (str): Path to save the output binary raster.
            code (int): The class value to retain (set to 1); all other values become 0.

        Returns:
            str: Path to the saved binary raster file.
        """

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

    def download_assets(
        self,
        name: str = "assets",
        resample: bool = True,
        overwrite: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Download, process, and merge asset datasets for a given country.

        Args:
            name (str, optional): Name of the asset dataset. (default: "assets")
            resample (bool, optional): Whether to resample datasets (e.g., WorldCover)
                                       to match reference resolution. (default: True)
            overwrite (bool, optional): Whether to overwrite existing merged data. (default: False)

        Returns:
            gpd.GeoDataFrame: Merged asset dataset in the CRS of the object.
        """
        # Output file path for merged hazard data
        full_data_file = self._build_filename(
            self.iso_code,
            f"{name}_{self.adm_level}",
            self.local_dir,
            ext="geojson",
        )

        # If overwrite is set or file does not exist, regenerate it
        if overwrite or not os.path.exists(full_data_file):
            full_data = None

            global_assets = self.config[f"{name}_selected"]
            for index, asset in enumerate(global_assets):
                logging.info(
                    f"({index+1}/{len(global_assets)}) Processing {asset}..."
                )

                # Download raster asset (GeoTIFF)
                if "worldcover" in asset:
                    land_cover_class = asset.split("_")[-1]
                    self.download_worldcover(
                        land_cover_class=land_cover_class,
                        resample=resample,
                    )
                else:
                    self.download_url(asset, ext="tif")

            for asset_name, asset_file in zip(
                self.asset_names, self.asset_files
            ):
                # Zonal statistics for base hazard raster
                stats_agg = ["sum"]
                data = self._calculate_zonal_stats(
                    asset_file,
                    column=asset_name,
                    stats_agg=stats_agg,
                    overwrite=overwrite,
                )

                # Merge into cumulative asset
                if full_data is None:
                    full_data = data.copy()
                if not set(data.columns).issubset(set(full_data.columns)):
                    full_data = self._merge_data(
                        [full_data, data], columns=self.merge_columns
                    )
                self._cleanup()

            # Save merged hazard asset
            full_data.to_file(full_data_file)
            logging.info(f"Data saved to {full_data_file}.")

        # Always load and return data in correct CRS
        full_data = gpd.read_file(full_data_file).to_crs(self.crs)

        return full_data

    def download_hazards(
        self, name: str = "hazards", overwrite: bool = False
    ) -> gpd.GeoDataFrame:
        """
        Download, process, and merge hazard datasets for a given country.

        Args:
            name (str, optional): Name of the hazard dataset. (default: "hazards")
            overwrite (bool, optional): Whether to overwrite existing merged data (default: False)

        Returns:
            gpd.GeoDataFrame: Merged hazard dataset with exposure statistics
                              for all assets, in the CRS of the object.
        """

        # Output file path for merged hazard data
        full_data_file = self._build_filename(
            self.iso_code,
            f"{name}_{self.adm_level}",
            self.local_dir,
            ext="geojson",
        )

        if overwrite or not os.path.exists(full_data_file):
            full_data = None

            global_hazards = self.config[f"{name}_selected"]
            for index, hazard in enumerate(global_hazards):
                logging.info(
                    f"({index+1}/{len(global_hazards)}) Downloading {hazard}..."
                )
                if "fathom" in hazard:
                    self.download_fathom(hazard)
                elif "jrc_glofas" in hazard:
                    self.download_jrc(hazard)
                else:
                    self.download_url(hazard, ext="tif")

            for index, (hazard_name, hazard_file) in enumerate(
                zip(self.hazard_names, self.hazard_files)
            ):
                logging.info(
                    f"({index+1}/{len(self.hazard_names)}) Processing {hazard_name}..."
                )

                for asset_name, asset_file in (
                    pbar := tqdm(
                        zip(self.asset_names, self.asset_files),
                        total=len(self.asset_names),
                    )
                ):
                    pbar.set_description(f"Processing {asset_name}")

                    exposure_file = self._build_filename(
                        self.iso_code,
                        f"{hazard_name}_{asset_name}_exposure",
                        self.local_dir,
                        ext="tif",
                    )
                    weighted_exposure_file = self._build_filename(
                        self.iso_code,
                        f"{hazard_name}_{asset_name}_intensity_weighted_exposure",
                        self.local_dir,
                        ext="tif",
                    )

                    # Generate exposure rasters
                    self._generate_exposure(
                        asset_name,
                        asset_file,
                        hazard_file,
                        exposure_file,
                        self.config["threshold"][hazard_name],
                        overwrite=overwrite,
                    )

                    # Zonal statistics for base hazard raster
                    stats_agg = ["mean"]
                    data = self._calculate_zonal_stats(
                        hazard_file,
                        column=hazard_name,
                        stats_agg=stats_agg,
                        overwrite=overwrite,
                    )

                    # Merge into cumulative hazard
                    if full_data is None:
                        full_data = data.copy()

                    elif not set(data.columns).issubset(
                        set(full_data.columns)
                    ):
                        full_data = self._merge_data(
                            [full_data, data], columns=self.merge_columns
                        )

                    exposure = self._calculate_zonal_stats(
                        exposure_file,
                        column=hazard_name,
                        suffix=f"{asset_name}_exposure",
                        overwrite=overwrite,
                    )
                    weighted_exposure = self._calculate_zonal_stats(
                        weighted_exposure_file,
                        column=hazard_name,
                        suffix=f"{asset_name}_intensity_weighted_exposure",
                        overwrite=overwrite,
                    )
                    full_data = self._merge_data(
                        [full_data, exposure, weighted_exposure],
                        columns=self.merge_columns,
                    )

                self._cleanup()

            # Save merged hazard hazard
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
        overwrite: bool = False,
    ) -> None:
        """
        Generate exposure raster for a given asset based on a threshold.

        This function resamples the asset raster to match the target resolution,
        then calculates exposure using the resampled raster and a threshold value.

        Args:
            asset (str): Name of the asset.
            asset_file (str): Path to the original asset raster file.
            local_file (str): Path to the local raster file used as reference for resampling.
            exposure_file (str): Path to the output exposure raster file.
            threshold (float): Threshold value for exposure calculation.
            overwrite (bool): Whether to overwrite existing exposure files. (default: False)

        Returns:
            None
        """
        # Only generate exposure if it hasn't already been computed
        if overwrite or not os.path.exists(exposure_file):
            resampled_file = local_file.replace(
                ".tif", f"_{asset.upper()}_RESAMPLED.tif"
            )

            # Resample raster if resampled version does not already exist
            if overwrite or not os.path.exists(resampled_file):
                self._resample_raster(asset_file, local_file, resampled_file)

            self._calculate_exposure(
                asset_file, resampled_file, exposure_file, threshold
            )

    def _resample_raster(
        self, asset_file: str, in_file: str, out_file: str
    ) -> str:
        # Open the asset raster (reference for resolution and bounds)
        asset = gdal.Open(asset_file, 0)

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
        Resample an input raster to match the resolution and bounds of a reference asset raster.

        This function uses GDAL to warp the input raster (`in_file`) so that it aligns with
        the resolution and bounding box of the reference raster (`asset_file`). The resampled
        raster is saved to `out_file`.

        Args:
            asset_file (str): Path to the reference asset raster file used to determine resolution and bounds.
            in_file (str): Path to the input raster file to be resampled.
            out_file (str): Path to the output resampled raster file.

        Returns:
            str: Path to the resampled raster file (`out_file`).
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
            asset_binary, hazard = self._match_shape(asset_binary, hazard)
            hazard_scaled = self._minmax_scale(hazard * asset_binary)

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
        Aggregate a GeoDataFrame by administrative level.

        Args:
            data (gpd.GeoDataFrame): The input GeoDataFrame to aggregate.
            agg_col (str, optional): Column to aggregate. (default: None)
            agg_func (str, optional): Aggregation function to apply (e.g., "sum", "count"). (default: "sum")
            adm_level (str, optional): Administrative level column to group by. (default: self.adm_level)

        Returns:
            gpd.GeoDataFrame: Aggregated GeoDataFrame with columns [adm_level, agg_col].
        """

        # Define administrative ID column
        adm_level = adm_level or self.adm_level
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

    def _merge_tifs(self, in_files, vrt_file, tif_file) -> None:
        """
        Merge multiple TIFF files into a single GeoTIFF via a VRT.

        Args:
            in_files (str): Input file pattern or list of TIFF files to merge.
            vrt_file (str): Temporary VRT file to create during the merge.
            tif_file (str): Output merged GeoTIFF file.

        Returns:
            None
        """
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
        Clip a global raster to the boundary of a given administrative area.

        Args:
            global_tif (str): Path to the input global raster file.
            local_tif (str): Path to save the clipped raster locally.
            admin (gpd.GeoDataFrame): GeoDataFrame containing the administrative boundary.
            nodata (list, optional): List of values to treat as nodata and replace with 0. (default: [])

        Returns:
            rio.io.DatasetReader: Rasterio dataset of the clipped raster.
        """
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

    def calculate_multihazard_score(
        self,
        data: gpd.GeoDataFrame,
        conflict_columns: list = None,
        suffixes: list = None,
        aggregation: str = "arithmetic_mean",
        p: float = 0.5,
        epsilon: float = 0.00001,
        scaled: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Calculate multi-hazard scores (MHS) and conflict-adjusted MHS for each asset.

        Args:
            data (gpd.GeoDataFrame): GeoDataFrame containing hazard and asset data.
            conflict_columns (list, optional): List of conflict-related columns to adjust MHS by.
                (default: self.config["conflict_columns"])
            suffixes (list, optional): List of suffixes used for hazard columns (default: self.config["suffixes"])
            aggregation (str, optional): Method to aggregate multiple hazard layers. Options include
                'arithmetic_mean', 'geometric_mean', etc. (default: "arithmetic_mean")
            p (float, optional): Power parameter. (default: 0.5)
            epsilon (float, optional): Small constant to avoid division by zero in calculations. (default: 0.00001)
            scaled (bool, optional): Whether to scale the multi-hazard score using min-max scaling. (default: True)

        Returns:
            gpd.GeoDataFrame: Input GeoDataFrame with added columns for MHS and conflict-adjusted MHS.
                Columns follow the naming scheme:
                - mhs_{category}_{asset}_{suffix}: Multi-hazard score per asset
                - mhs_{category}_{conflict_column}_{asset}_{suffix}: Conflict-adjusted MHS
        """

        conflict_columns = conflict_columns or self.config["conflict_columns"]
        suffixes = suffixes or self.config["suffixes"]

        hazard_dicts = {**self.config["hazards_all"], "all": self.hazard_names}

        for suffix in suffixes:
            self.hazard_cols[suffix] = dict()
            for category, hazards in hazard_dicts.items():
                self.hazard_cols[suffix][category] = dict()
                hazards = [hazard.replace("global_", "") for hazard in hazards]
                for asset in self.asset_names:
                    hazard_cols = [
                        f"{hazard}_{asset}_{suffix}"
                        for hazard in hazards
                        if f"{hazard}_{asset}_{suffix}" in data.columns
                        and not (data[f"{hazard}_{asset}_{suffix}"] == 0).all()
                    ]
                    self.hazard_cols[suffix][category][asset] = hazard_cols

                    if len(hazard_cols) == 0:
                        continue

                    weights = np.ones(len(hazard_cols))
                    mhs = self._calculate_mhs(
                        data[hazard_cols], weights, aggregation
                    )

                    mhs_name = f"mhs_{category}_{asset}_{suffix}"
                    data[mhs_name] = self._minmax_scale(mhs) if scaled else mhs

                    for conflict in conflict_columns:
                        conflict_col = f"{conflict}_{asset}_{suffix}"

                        if conflict_col not in data.columns:
                            continue

                        conflict_val = data[conflict_col]
                        condition = scaled or "absolute" in conflict_col
                        conflict_scaled = (
                            self._minmax_scale(conflict_val)
                            if condition
                            else conflict_val
                        )

                        mhsc_name = f"mhs_{category}_{conflict_col}"
                        data[mhsc_name] = data[mhs_name] * conflict_scaled

        return data

    def _calculate_zonal_stats(
        self,
        in_file: str,
        column: str,
        out_file: str = None,
        stats_agg: list = ["sum"],
        add_stats: list = None,
        suffix: str = None,
        prefix: str = None,
        overwrite: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Compute zonal statistics for a raster file over administrative boundaries.

        Args:
            in_file (str): Path to the input raster file.
            column (str): Name of the column to store zonal statistics.
            out_file (str, optional): Path to save output GeoJSON. (default: None, auto-generated)
            stats_agg (list, optional): List of statistics to aggregate, e.g., ["sum", "mean"]. (default: ["sum"])
            add_stats (list, optional): Additional statistics to compute. (default: None)
            suffix (str, optional): Suffix to append to the output column name. (default: None)
            prefix (str, optional): Prefix to prepend to the output column name. (default: None)
            overwrite (bool, optional): Whether to overwrite existing output files. (default: False)

        Returns:
            gpd.GeoDataFrame: GeoDataFrame of administrative boundaries with zonal statistics added as a column.
        """

        # Extract base name from raster file
        name = os.path.basename(in_file).split(".")[0]

        # Generate default output path if not provided
        if out_file is None:
            out_file = os.path.join(
                self.local_dir, f"{name}_{self.adm_level}.geojson"
            )

        if overwrite or not os.path.exists(out_file):
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

    def _calculate_relative_exposure(self, data: gpd.GeoDataFrame):
        """
        Compute relative exposure for asset and hazard columns in a GeoDataFrame.

        Args:
            data (gpd.GeoDataFrame): GeoDataFrame containing asset and hazard exposure columns.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with additional adjusted and relative exposure columns.

        Notes:
            Columns containing "worldcover" are scaled by 0.01 and saved with "_adjusted" suffix.
            Relative exposure is calculated per asset as column/asset total.
        """

        for column in data.columns:
            if "worldcover" in column and "_adjusted" not in column:
                data[column + "_adjusted"] = data[column] * 0.01

            if "exposure" in column and "_absolute" not in column:
                column_absolute = column + "_absolute"
                data = data.rename(columns={column: column_absolute})

        for asset_name in self.asset_names:
            for column in data.columns:
                if asset_name in column and "absolute" in column:
                    column_relative = column.replace("absolute", "relative")
                    data[column] = data[column].astype(float).fillna(np.nan)
                    denominator = data[asset_name].replace(0, np.nan)
                    data[column_relative] = data[column] / denominator

        return data

    def _get_dataset_names(self, name: str):
        """
        Generate dataset names and corresponding local file paths for a given dataset type.

        Args:
            name (str): The dataset category (e.g., "assets", "hazards") to retrieve.

        Returns:
            tuple[list, list]: A tuple containing:
                - dataset_names (list): Cleaned dataset names.
                - dataset_files (list): Corresponding local file paths for each dataset.

        Notes:
            - Removes the global name prefix from each dataset.
            - Builds local file paths in the self.local_dir with ".tif" extension.
        """
        dataset_names, dataset_files = [], []

        for dataset in self.config[f"{name}_selected"]:
            dataset = dataset.replace(f"{self.global_name.lower()}_", "")
            dataset_file = self._build_filename(
                self.iso_code, dataset, self.local_dir, ext="tif"
            )
            dataset_names.append(dataset)
            dataset_files.append(dataset_file)

        return dataset_names, dataset_files

    def _cleanup(self, local_dir: str = None) -> None:
        """
        Remove intermediate or temporary files in the specified directory.
        Deletes files containing keywords indicating temporary processing:
              "resampled", "binary", "filtered", "temp" (case-insensitive).

        Args:
            local_dir (str, optional): Directory to clean up. Defaults to self.local_dir.
        """
        local_dir = local_dir or self.local_dir
        keywords = ["resampled", "binary", "filtered", "temp"]
        for file in os.listdir(local_dir):
            filepath = os.path.join(local_dir, file)
            for keyword in keywords:
                if os.path.isfile(filepath) and keyword.upper() in file:
                    os.remove(filepath)

    def _download_with_aggregate(self, func, *args, **kwargs):
        """
        Download a dataset and optionally its aggregated version.

        Args:
            func (callable): Function to download the dataset. Should accept `aggregate` as a keyword argument.
            *args: Positional arguments to pass to `func`.
            **kwargs: Keyword arguments to pass to `func`.

        Returns:
            tuple:
                - base (object): The base dataset returned by `func`.
                - agg (object or None): The aggregated dataset if available; otherwise None.
        """
        base = func(*args, **kwargs)
        agg = (
            func(*args, **kwargs, aggregate=True) if base is not None else None
        )
        return base, agg

    def _resolve_config_path(self, provided_path, filename):
        """
        Resolve the path to a configuration file.

        Args:
            provided_path (str, optional): Custom path to the configuration file.
                If None, defaults to the package resources path.
            filename (str): Name of the configuration file.

        Returns:
            str: The resolved path to the configuration file.
        """
        resources = importlib_resources.files("dfcv_colocation_mapping")
        return provided_path or resources.joinpath("configs", filename)

    def _get_country_name(self, iso_code: str):
        """
        Get the full country name from an ISO-3 code, applying any overrides in the config.

        Args:
            iso_code (str): ISO-3 country code (e.g., 'PHL').

        Returns:
            str: The full country name, possibly overridden by the configuration.
        """
        country = pycountry.countries.get(alpha_3=iso_code).name
        for config_iso_code in self.config["country_map_code"]:
            if iso_code == config_iso_code:
                return self.config["country_map_code"][iso_code]
        return country

    def _load_creds(self, file_path: str, key_field: str, default: str = None):
        """
        Load a credential from a configuration file.

        Args:
            file_path (str): Path to the credential/configuration file.
            key_field (str): Key to look up in the file.
            default (str, optional): Value to return if the key is not found or file does not exist. (default: None)

        Returns:
            str: The value corresponding to the key_field, or the default.
        """
        if os.path.exists(file_path):
            creds = common_utils.read_config(file_path)
            return creds.get(key_field)
        return default

    def _get_year(self, date: str):
        """
        Extract the year from a date string.

        Args:
            date (str): Date string in the format 'YYYY-MM-DD'.

        Returns:
            int: The year as an integer.
        """
        return datetime.datetime.strptime(date, "%Y-%m-%d").year

    def _get_start_date(self, start_date, last_n_years: int):
        """
        Get the start date for a time range, defaulting to N years ago if not provided.

        Args:
            start_date (str, optional): The start date in 'YYYY-MM-DD' format. (default: None)
            last_n_years (int, optional): Number of years to go back if start_date is None. (default: 5)

        Returns:
            str: Start date in 'YYYY-MM-DD' format.
        """
        return (
            start_date
            or (
                datetime.date.today() - relativedelta(years=last_n_years)
            ).isoformat()
        )

    def _get_end_date(self, end_date):
        """
        Get the end date for a time range, defaulting to today if not provided.

        Args:
            end_date (str, optional): End date in 'YYYY-MM-DD' format. (default: None)

        Returns:
            str: End date in 'YYYY-MM-DD' format.
        """
        return end_date or datetime.date.today().isoformat()

    def _get_dtm_adm_level(self, dtm_adm_level):
        """
        Determine the DTM administrative level based on the object's adm_level if not provided.

        Args:
            dtm_adm_level (str, optional): Desired DTM administrative level. (default: None)

        Returns:
            str: The administrative level to use.
        """
        if dtm_adm_level is None:
            if int(self.adm_level[-1]) > 2:
                dtm_adm_level = "ADM2"
            else:
                dtm_adm_level = self.adm_level

        return dtm_adm_level

    def collate_osm_tags(self, osm_data: dict, tags: list) -> gpd.GeoDataFrame:
        """
        Collate and concatenate OSM features across multiple tags.

        Args:
            osm_data (dict): Dictionary mapping OSM tag names to
                GeoDataFrames.
            tags (list): List of OSM tag names to collate.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing features from
            all specified tags, ordered by decreasing feature count.
        """

        # Collect GeoDataFrames and record their sizes
        gdfs_with_counts = []
        for tag in tags:
            gdf = osm_data[tag].copy()
            gdfs_with_counts.append((len(gdf), gdf))

        # Sort by feature count
        gdfs_with_counts.sort(key=lambda x: x[0], reverse=True)

        # Concatenate — denser ones first, sparser last
        ordered_gdfs = [gdf for _, gdf in gdfs_with_counts]
        combined = gpd.GeoDataFrame(pd.concat(ordered_gdfs, ignore_index=True))

        return combined

    def update_acled_selected(self, drm_pillar: str = None):
        """
        Update the ACLED selection based on a DRM pillar.

        Filters the existing ACLED configuration so that only categories,
        subcategories, and values compatible with the specified DRM pillar
        are retained. If no pillar is provided, the selection is returned
        unchanged.

        Args:
            drm_pillar (str, optional): DRM pillar used to filter ACLED
                selections. If None, no filtering is applied.

        Returns:
            dict: Updated ACLED selection dictionary.
        """

        self.acled_selected = copy.deepcopy(
            self.acled_config["acled_selected"]
        )
        if drm_pillar is None:
            return self.acled_selected

        d2 = self.acled_drm_pillars[drm_pillar]
        for sector, d1 in self.acled_selected.items():
            self.acled_selected[sector] = {
                cat: {
                    key: list(set(values) & set(d2[cat][key]))
                    for key, values in subcats.items()
                    if key in d2[cat] and set(values) & set(d2[cat][key])
                }
                for cat, subcats in d1.items()
                if cat in d2
            }

        return self.acled_selected

    def _assign_grouping(self, iso_code, data, config):
        """
        Assign grouping to a GeoDataFrame based on ISO code and configuration.

        Args:
            iso_code (str): ISO country code.
            data (gpd.GeoDataFrame): GeoDataFrame to assign grouping to.
            config (dict): Configuration dictionary mapping ISO codes to grouping rules.

        Returns:
            tuple[gpd.GeoDataFrame, str]: Updated GeoDataFrame and the group name.
        """
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

    def _merge_data(
        self,
        full_data: gpd.GeoDataFrame,
        columns: list = [],
        how: str = "left",
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

    def _build_filename(self, prefix, suffix, out_dir, ext="geojson") -> str:
        """
        Construct a standardized filename using prefix, suffix, and extension.

        Args:
            prefix (str): Prefix for the filename (e.g., country code).
            suffix (str): Suffix for the filename (e.g., dataset name).
            out_dir (str): Directory where the file will be saved.
            ext (str, optional): File extension. (default: "geojson")

        Returns:
            str: Full file path.
        """
        return os.path.join(
            out_dir, f"{prefix.upper()}_{suffix.upper()}.{ext}"
        )

    def _minmax_scale(self, data: pd.Series) -> pd.Series:
        """
        Performs min-max scaling to [0, 1].

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

    def _match_shape(self, src1: np.ndarray, src2: np.ndarray) -> np.ndarray:
        """
        Match the shape of two 2D arrays by cropping and padding the second array.

        The function ensures that `src2` has the same shape as `src1` by:
        1. Cropping `src2` to the minimum overlapping rows and columns.
        2. Padding `src2` with zeros (bottom and right) if it is still smaller.

        `src1` is returned unchanged.

        Args:
            src1 (np.ndarray): Reference 2D array whose shape is preserved.
            src2 (np.ndarray): 2D array to be cropped and/or padded.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - src1: The original reference array.
                - src2: The adjusted array with the same shape as src1.
        """
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

    def _calculate_mhs(
        self,
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

    def set_selected_datasets(self):
        """
        Load and set the selected asset and hazard datasets for the object.

        Populates:
            self.asset_names (list): List of selected asset dataset names.
            self.asset_files (list): Corresponding local file paths for assets.
            self.hazard_names (list): List of selected hazard dataset names.
            self.hazard_files (list): Corresponding local file paths for hazards.
        """
        self.asset_names, self.asset_files = self._get_dataset_names("assets")
        self.hazard_names, self.hazard_files = self._get_dataset_names(
            "hazards"
        )
