import os
import re
import json
import shutil
import requests
import logging
import warnings
from tqdm import tqdm
import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np

import urllib.request
import zipfile

import geojson
from osgeo import gdal
import geopandas as gpd
import rasterio as rio
import rasterio.mask
import rasterstats
import pycountry

import importlib_resources
import itertools
import ahpy
import bs4
import requests
from functools import reduce

from scipy.stats.mstats import gmean
from dfcv_colocation_mapping import data_utils

logging.basicConfig(level=logging.INFO, force=True)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class DatasetManager:
    def __init__(
        self,
        iso_code: str,
        adm_source: str = 'geoboundary',
        acled_key: str = None,
        acled_email: str = None,
        acled_limit: str = None,
        acled_exclude: str = None,
        acled_country: str = None,
        acled_name: str = "acled",
        ucdp_name: str = "ucdp",
        conflict_start_date: str = None,
        conflict_end_date: str = None,
        jrc_rp: int = 100,
        jrc_version: str = "v1",
        fathom_year: int = 2020,
        fathom_rp: int = 50,
        fathom_threshold: int = 50,
        fathom_name: str = "fathom",
        adm_level: str = "ADM3",
        datasets: list = None,
        data_dir: str = "data",
        meter_crs: str = "EPSG:3857",
        crs: str = "EPSG:4326",
        asset: str = "worldpop",
        global_name: str = "global",
        overwrite: bool = False,
        group: str = "Region",
        mhs_aggregation: str = "power_mean",
        config_file: str = None,
        acled_file: str = None,
        adm_config_file: str = None
    ):
        """Initialize a DatasetManager object for a given country.

        Args:
            iso_code (str): ISO3 country code.
            adm_source (str, optional): Administrative boundary source. Defaults to 'geoboundary'.
            acled_key (str, optional): ACLED API key. Defaults to None.
            acled_email (str, optional): ACLED account email. Defaults to None.
            conflict_start_date (str, optional): Conflict start date. This is set to 10 years ago from today if none. Defaults to None.
            conflict_end_date (str, optional): Conflict end date. This is set to the current date if none. Defaults to None.
            acled_limit (str, optional): Dictionary specifying filter conditions on columns and sub-columns. Defaults to None.
                Examples of limit dictionary value:
                limit = {"disorder_type": {"Strategic developments": {"sub_event_type": ["Looting/property destruction", "Arrests"]}}}
                limit = {"disorder_type": ["Demonstrations", "Political violence"]}
            acled_exclude (str, optional): Dictionary specifying exclusion conditions on columns. Defaults to None.
                Rows matching the specified values will be removed. 
                Example of exclude dictionary value:
                exclude = {"disorder_type": ["Strategic developments"]}
                exclude = {"sub_event_type": ["Peaceful protest"]}
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
            asset (str, optional): Default asset layer name. Defaults to "worldpop".
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
        self.asset = asset
        self.overwrite = overwrite

        # Store ACLED credentials and filtering options
        self.acled_key = acled_key
        self.acled_email = acled_email
        self.acled_limit = acled_limit
        self.acled_exclude = acled_exclude
        self.acled_country = acled_country

        # Store conflict start date and end date
        self.conflict_start_date = conflict_start_date
        self.conflict_end_date = conflict_end_date

        if self.conflict_start_date is None:
            self.conflict_start_date = (datetime.date.today() - relativedelta(years=10)).isoformat()
        if self.conflict_end_date is None:
            self.conflict_end_date = datetime.date.today().isoformat()

        # Store Fathom flood layer parameters
        self.fathom_year = fathom_year
        self.fathom_rp = fathom_rp
        self.fathom_threshold = fathom_threshold
        self.jrc_rp = jrc_rp
        self.jrc_version = jrc_version

        # Get country name from ISO Code
        self.country = pycountry.countries.get(alpha_3=self.iso_code).name
        if self.iso_code == "COD": # Edge case
            self.country = "Democratic Republic of Congo"
        if self.country is None:
            raise ValueError(f"Invalid ISO code: {self.iso_code}")

        # Locate default configuration files if not provided
        resources = importlib_resources.files("dfcv_colocation_mapping")
        if config_file is None:
            config_file = resources.joinpath("configs", "data_config.yaml")
        if acled_file is None:
            acled_file = resources.joinpath("configs", "acled_creds.yaml")
        if adm_config_file is None:
            adm_config_file = resources.joinpath("configs", "adm_config.yaml")

        # Load main config
        self.config = data_utils.read_config(config_file)

        # Load ACLED credentials from file if available
        if os.path.exists(acled_file):
            self.acled_creds = data_utils.read_config(acled_file)
            self.acled_key = self.acled_creds["acled_key"]
            self.acled_email = self.acled_creds["acled_email"]

        # Uppercase standard labels
        self.global_name = global_name.upper()
        self.fathom_name = fathom_name.upper()
        self.acled_name = acled_name.upper()
        self.ucdp_name = ucdp_name.upper()

        # Prepare directories for storing data
        self.data_dir = os.path.join(os.getcwd(), data_dir)
        self.local_dir = os.path.join(os.getcwd(), data_dir, iso_code)
        self.global_dir = os.path.join(os.getcwd(), data_dir, self.global_name)
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.local_dir, exist_ok=True)
        os.makedirs(self.global_dir, exist_ok=True)

        # Build file path for asset layer
        self.asset_file = self._build_filename(iso_code, asset, self.local_dir, ext="tif")
        self.admin_file = None

        # Load geoboundaries, fallback to GADM if primary source fails
        logging.info("Loading geoboundary...")
        self.adm_source = adm_source
        self.geoboundary = self.download_geoboundary_with_attempts()
        self.merge_columns = list(self.geoboundary.columns)

        # Load hazard layers
        logging.info("Loading asset and hazard layers...")
        self.assets = self.download_datasets("asset")
        self.fathom = self.download_fathom()
        self.hazards = self.download_datasets("hazard")
        
        logging.info(f"Downloading conflict data from {self.conflict_start_date} to {self.conflict_end_date}")

        # Load acled conflict data
        logging.info("Loading ACLED conflict data...")
        self.acled = self.download_acled()
        self.acled_agg = self.download_acled(aggregate=True)

        # Load ucdp conflict data
        logging.info("Loading UCDP conflict data...")
        self.ucdp = self.download_ucdp()
        self.ucdp_agg = self.download_ucdp(aggregate=True)

        # Compute multi-hazard scores
        logging.info("Calculating scores...")
        self.mhs_aggregation = mhs_aggregation
        self.data = self.combine_datasets()
        self.data = self.calculate_multihazard_score(self.data)

        # Load admin config and assign grouping
        self.adm_config = data_utils.read_config(adm_config_file)
        self.data = self.assign_grouping()


    def download_geoboundary_with_attempts(self, attempts: int = 3):
        for i in range(attempts):
            try:
                return self.download_geoboundary(self.adm_source)
            except Exception as err:
                logging.info(err)
                logging.info(f"Loading geoboundaries failed. Trying again {i}/{attempts}")
                pass
                
        logging.info(f"Loading geoboundaries failed. Trying with GADM...")
        return self.download_geoboundary("gadm")


    def calculate_ahp(
        self, 
        ahp_precision: int = 5, 
        ahp_random_index: str = "saaty", 
        cr_threshold: float = 0.10
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
        
        # Get list of hazards from config
        #hazards_all = self.config["hazards"].keys()

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
            name='Hazards', 
            comparisons=weight_dict, 
            precision=ahp_precision, 
            random_index=ahp_random_index
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
                        self.config["hazards"][category][hazard] = weights[hazard]
            logging.info(self.config["hazards"])

            # Recalculate multi-hazard scores in the dataset
            self.data = self.calculate_multihazard_score(self.data)
            return self.data
        else:
            # Raise error if consistency ratio exceeds threshold
            raise ValueError(f'Consistency ratio {cr} > 0.10. Please try again.')
                     

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

        # Add hazards and Fathom datasets if available, replacing NaN with 0
        for dataset in [self.assets, self.hazards, self.fathom]:
            if dataset is not None:
                dataset = dataset.mask(dataset.isna(), 0)
                data.append(dataset)

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
        
        return data

    def calculate_multihazard_score(
        self,
        data: gpd.GeoDataFrame,
        conflict_columns: list = [
            "wbg_acled",
            "ucdp"
        ],
        suffixes: list = [
            "exposure_relative", 
            "intensity_weighted_exposure_relative"
        ],
        aggregation: str = "power_mean",
        p: float = 0.5,
        epsilon: float = 0.00001
    ) -> gpd.GeoDataFrame:
        """Calculate multi-hazard scores for each geographic unit.
    
        Computes multi-hazard scores (MHS) for each row in the dataset using
        either power mean, geometric mean, or arithmetic mean aggregation
        of all hazards. Optionally scales MHS with a conflict column.
    
        Args:
            data (gpd.GeoDataFrame): Input GeoDataFrame containing hazard data.
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
        for column in data.columns:
            if "relative" not in column:
                colname = f"{column}_relative"
                if "exposure" in column and colname not in data.columns:
                    data[colname] = data[column] / data[self.asset]

        # Loop through each suffix to calculate MHS
        for suffix in suffixes:
            # Prepare hazard columns and normalized weights
            hazard_dicts, categories = [], list(self.config["hazards"].keys())
            for category in categories:
                hazard_dicts.append(self.config["hazards"][category])

            from functools import reduce
            all_hazards = reduce(lambda a, b: {**a, **b}, hazard_dicts)
            hazard_dicts.append(all_hazards)
            categories.append("all")

            for hazard_dict, category in zip(hazard_dicts, categories):
                hazard_cols = [
                    f"{hazard}_{self.asset}_{suffix}" if suffix else hazard
                    for hazard in hazard_dict
                    if (f"{hazard}_{self.asset}_{suffix}" if suffix else hazard) in data.columns
                ]
                
                total_weight = sum(list(hazard_dict.values()))
                weights = np.array([
                    hazard_dict[hazard] 
                    for hazard in hazard_dict
                ])
                if total_weight > 0:
                    weights = weights / total_weight
                
                # Select only columns that exist in data
                hazard_cols = [col for col in hazard_cols if col in data.columns]
                weights = weights[:len(hazard_cols)]  
                
                # Compute MHS using vectorized operations
                if self.mhs_aggregation == "power_mean":
                    mhs = (data[hazard_cols] ** p).multiply(weights, axis=1).sum(axis=1) ** (1 / p)
                
                elif self.mhs_aggregation == "geometric_mean":
                    mhs = (data[hazard_cols] + epsilon).pow(weights).prod(axis=1)
                
                elif self.mhs_aggregation == "arithmetic_mean":
                    mhs = data[hazard_cols].multiply(weights, axis=1).sum(axis=1)
    
                # Add MHS column (scaled 0-1)
                mhs_name = "mhs"
                if suffix is not None:
                    mhs_name = f"{mhs_name}_{category}_{self.asset}_{suffix}"
                data[mhs_name] = data_utils._minmax_scale(mhs)
    
                # Optionally scale MHS by conflict
                for conflict_column in conflict_columns:
                    mhsc_name = f"mhs_{category}_{conflict_column}"
                    
                    if suffix is not None:
                        mhsc_name = f"{mhsc_name}_{self.asset}_{suffix}"
    
                    for csuffix in suffixes:
                        if f"{conflict_column}_{self.asset}_{csuffix}" in data.columns:
                            conflict_scaled = data_utils._minmax_scale(data[f"{conflict_column}_{self.asset}_{csuffix}"])
                            data[mhsc_name] = data[mhs_name] * conflict_scaled
    
        return data

    
    def download_geoboundary(self, adm_source: str) -> gpd.GeoDataFrame:
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
            self.iso_code, f"{adm_source}_{self.adm_level}", self.local_dir, ext="geojson"
        )    
        gadm_file = self._build_filename(
            self.iso_code, f"gadm_{self.adm_level}", self.local_dir, ext="geojson"
        ) 

        # Prefer GADM file if exists
        if os.path.exists(gadm_file) and not self.overwrite:
            adm_source = "gadm"
            out_file = gadm_file

        # Download only if file doesn't exist or overwrite is True
        elif self.overwrite or not os.path.exists(out_file):
            logging.info(f"Downloading geoboundary for {self.iso_code}...")

            # Download GADM dataset
            if adm_source == 'gadm':
                try:
                    self.download_url(
                        adm_source, 
                        dataset_name=f"{adm_source}_{self.adm_level}", 
                        ext="geojson"
                    )
                    geoboundary = gpd.read_file(out_file)
                except Exception as e:
                    raise FileNotFoundError(f"Failed to download or read GADM data: {str(e)}")

                # Rename columns to standard format
                rename = dict()
                for index in range(int(self.adm_level[-1])+1):
                    if index == 0:
                        rename[f'GID_{index}'] = 'iso_code'
                    else:
                        rename[f'GID_{index}'] = f'ADM{index}_ID'
                        rename[f'NAME_{index}'] = f'ADM{index}'
    
                geoboundary = geoboundary.rename(columns=rename)
                all_columns = list(rename.values()) + ['geometry']
                geoboundary = geoboundary[all_columns]

                geoboundary.to_file(out_file)
                logging.info(f"Geoboundary file saved to {out_file}.")
                
            elif adm_source == "geoboundary":
                # Download GeoBoundaries dataset
                gbhumanitarian_url = self.config["urls"]["gbhumanitarian_url"]
                gbopen_url = self.config["urls"]["gbopen_url"]
                level = int(self.adm_level[-1])

                # Download each administrative level
                datasets = []
                for index in range(1, level+1):
                    adm_level = f"ADM{index}"
                    intermediate_file = self._build_filename(
                        self.iso_code, adm_level, self.local_dir, ext="geojson"
                    )

                    # Try GBHumanitarian URL first
                    url = f"{gbhumanitarian_url}{self.iso_code}/{adm_level}/"
                    try:
                        r = requests.get(url)
                        download_path = r.json()["gjDownloadURL"]
                    except Exception:
                        # Fallback to GBOpen URL if GBHumanitarian URL fails
                        try:
                            url = f"{gbopen_url}{self.iso_code}/{adm_level}/"
                            r = requests.get(url)
                            download_path = r.json()["gjDownloadURL"]
                        except Exception as e:
                            raise requests.RequestException(f"Failed to download {adm_level} boundaries: {str(e)}")
            
                    # Download and save the GeoJSON data
                    try:
                        geoboundary = requests.get(download_path).json()
                        with open(intermediate_file, "w") as file:
                            geojson.dump(geoboundary, file)
                    except Exception as e:
                        raise FileNotFoundError(f"Failed to save GeoJSON file {intermediate_file}: {str(e)}")

                    try:
                        # Read the downloaded GeoJSON into a GeoDataFrame
                        geoboundary = gpd.read_file(intermediate_file)
                        geoboundary["iso_code"] = self.iso_code
                
                        # Select relevant columns and rename them
                        geoboundary = geoboundary[["iso_code", "shapeName", "shapeID", "geometry"]]
                        geoboundary.columns = ["iso_code", adm_level, f"{adm_level}_ID", "geometry"]

                        # Save geoboundary with renamed columns
                        datasets.append(geoboundary)
                        geoboundary.to_file(intermediate_file)
                        logging.info(f"Geoboundary file saved to {intermediate_file}.")
                        
                    except Exception as e:
                        raise FileNotFoundError(f"Failed to read GeoJSON file {intermediate_file}: {str(e)}")

                # Merge multiple levels
                geoboundary = datasets[-1].to_crs(self.meter_crs)
                columns = geoboundary.columns

                # Iterate through the remaining DataFrames and perform joins
                for index in reversed(range(level-1)):
                    current = datasets[index].to_crs(self.meter_crs)
                    join_columns = [f"ADM{index+1}_ID", f"ADM{index+1}", 'geometry']
                    joined = geoboundary.sjoin(current[join_columns], predicate="intersects").drop(columns=["index_right"])
                    joined = joined.to_crs(self.meter_crs)
                                        
                    # Calculate the intersection area and percentage overlap
                    adm = join_columns[0]
                    joined['intersection_area'] = joined.apply(
                        lambda row: row.geometry.intersection(current[current[adm] == row[adm]].iloc[0].geometry).area, axis=1
                    )
                    joined['overlap_percentage'] = joined['intersection_area'] / joined['geometry'].area * 100 
                    
                    # Filter for the desired overlap percentage
                    geoboundary = joined[joined['overlap_percentage'] >= 50]
                    columns = list(columns) + list(join_columns[:-1])
                    geoboundary = geoboundary[columns]
            else:
                raise ValueError(f"adm_source '{adm_source}' not recognized. Use 'gadm' or 'geoboundary'.")
                
            geoboundary.to_crs(self.crs).to_file(out_file)
            logging.info(f"Geoboundary file saved to {out_file}.")

        # Load final geoboundary and update attributes
        self.admin_file = out_file
        self.adm_source = adm_source
        geoboundary = gpd.read_file(out_file).to_crs(self.crs)
        self.merge_columns = list(geoboundary.columns)
        
        return geoboundary


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
            ucdp["country"] = ucdp["country"].apply(lambda x: re.sub(r'\s*\([^)]*\)', '', x))
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
                3: "One-sided violence"
            }
            ucdp['type_of_violence'] = ucdp['type_of_violence'].replace(type_of_violence_map)

            if len(ucdp) == 0:
                logging.info(f"No UCDP data found for {self.iso_code}.")
                return 

            ucdp = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(ucdp["longitude"], ucdp["latitude"], crs=self.crs),
                data=ucdp
            )            
            ucdp.to_file(local_file, driver="GeoJSON")            
            logging.info(f"Saving UCDP to {local_file}")

        ucdp = gpd.read_file(local_file)

        if aggregate:
            exposure_raster = self._build_filename(
                self.iso_code, f"{self.ucdp_name}_{self.asset}_exposure", self.local_dir, ext="tif"
            )
            exposure_vector = self._build_filename(
                self.iso_code, f"{self.ucdp_name}_{self.asset}_exposure_{self.adm_level}", self.local_dir, ext="geojson"
            )

            if self.overwrite or not os.path.exists(exposure_vector):
                out_tif = self._calculate_custom_conflict_exposure(local_file, conflict_src="ucdp")
                out_tif, _ = self._calculate_exposure(out_tif, exposure_raster, threshold=1)

                column = f"{self.ucdp_name.lower()}_{self.asset}_exposure"
                data = self._calculate_zonal_stats(
                    out_tif,
                    column=column,
                    stats_agg=["sum"],
                    out_file=exposure_vector
                )
                if data is None or data.empty:
                    raise ValueError("Exposure calculation failed or produced empty results.")

                # Read exposure vector and clean zero values
                ucdp_agg = gpd.read_file(exposure_vector)
                ucdp_agg.loc[ucdp_agg[column] == 0, column] = None
    
                # Spatial join UCDP events with admin boundaries
                try:
                    admin = self.geoboundary
                    ucdp = ucdp.sjoin(admin, how="left", predicate="intersects")
                    ucdp = ucdp.drop(["index_right"], axis=1)
                except Exception as e:
                    raise ValueError(f"Spatial join failed: {e}")
    
                # Aggregate total conflict events
                column = "conflict_count"
                event_count = self._aggregate_data(
                    ucdp, agg_col=column, agg_func="count"
                )
                event_count = event_count.rename(
                    columns={column: f"ucdp_{column}"}
                )
                event_count = data_utils._merge_data(
                    [admin, event_count], columns=[f"{self.adm_level}_ID"], how="left"
                )

                fatalities_count = self._aggregate_data(
                    ucdp, agg_col="best", agg_func="sum"
                )
                fatalities_count = fatalities_count.rename(
                    columns={"best": f"ucdp_fatalities"}
                )
                fatalities_count = data_utils._merge_data(
                    [admin, fatalities_count], columns=[f"{self.adm_level}_ID"], how="left"
                )
                
                ucdp = data_utils._merge_data(
                    [event_count, fatalities_count, ucdp_agg], columns=self.merge_columns
                )

                agg = self._calculate_conflict_stats(ucdp, source="ucdp")
                
                ucdp.to_file(exposure_vector)

            ucdp = gpd.read_file(exposure_vector)
    
        return ucdp       

    
    def download_acled(
        self, 
        population: str = "full", 
        aggregate: bool = False
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
        out_file = self._build_filename(self.iso_code, self.acled_name, self.local_dir, ext="geojson")
        agg_file = self._build_filename(self.iso_code, f"{self.acled_name}_{self.adm_level}", self.local_dir, ext="geojson")
        
        exposure_raster = self._build_filename(
            self.iso_code, f"{self.acled_name}_{self.asset}_exposure", self.local_dir, ext="tif"
        )
        exposure_vector = self._build_filename(
            self.iso_code, f"{self.acled_name}_{self.asset}_exposure_{self.adm_level}", self.local_dir, ext="geojson"
        )

        if self.acled_key is None or self.acled_email is None:
            warnings.warn("WARNING: ACLED key or email is invalid.")
            return

        # Download ACLED data if needed
        if self.overwrite or not os.path.exists(out_file):    
            logging.info(f"Downloading ACLED data for {self.iso_code}...")
    
            params = dict(
                key=self.acled_key,
                email=self.acled_email,
                country=self.country,
                event_date=f"{self.conflict_start_date}|{self.conflict_end_date}",
                event_date_where="BETWEEN",
                population=population,
                page=1,
            )

            # Paginate through ACLED API results
            acled_url = self.config["urls"]["acled_url"]
            len_subdata = -1
            data = []
            while len_subdata != 0:
                try:
                    logging.info(f"Reading ACLED page {params['page']}...")
                    response = requests.get(acled_url, params=params)
                    subdata = pd.DataFrame(response.json()["data"])
                    data.append(subdata)
                    len_subdata = len(subdata)
                    params["page"] = params["page"] + 1  
                except:
                    warnings.warn("WARNING: ACLED failed to download.")
                    return

            # Concatenate all pages
            data = pd.concat(data)
            if len(data) == 0:
                warnings.warn(f"No ACLED data returned for {self.iso_code}")
                return

            # Convert to GeoDataFrame
            try:
                data = gpd.GeoDataFrame(
                    geometry=gpd.points_from_xy(data["longitude"], data["latitude"], crs=self.crs),
                    data=data,
                )
            except Exception as e:
                raise ValueError(f"Failed to convert ACLED data to GeoDataFrame: {str(e)}")

            # Clean and standardize columns
            data["population_best"] = (
                data["population_best"].replace({"": np.nan}).astype(np.float64)
            )
            data["disorder_type"] = data["disorder_type"].replace(
                {
                    "Political violence; Demonstrations": "Demonstrations",
                }
            )

            data["type_of_violence"] = None
            data["civilian_targeting"] = data["civilian_targeting"].replace("", None)
            data.loc[data["civilian_targeting"] == "Civilian targeting", "type_of_violence"] = "One-sided violence"
            data.loc[data["inter1"].str.contains("Civilians") | data["inter2"].str.contains("Civilians"), "type_of_violence"] = "One-sided violence"
            data.loc[data["inter1"].str.contains("State") | data["inter2"].str.contains("State"), "type_of_violence"] = "State-based conflict"
            data["type_of_violence"] = data["type_of_violence"].fillna("Non-state conflict")
            data[["type_of_violence", "civilian_targeting", "inter1", "inter2"]][data["type_of_violence"] == "State-based conflict"]

            # Save to file
            data.to_file(out_file)
            logging.info(f"ACLED file saved to {out_file}.")

        # Read ACLED data from file
        try:
            acled = gpd.read_file(out_file).to_crs(self.crs)
        except Exception as e:
            raise FileNotFoundError(f"Failed to read ACLED file {out_file}: {str(e)}")

        # Apply optional filters
        if self.acled_limit is not None:
            acled = self._limit_filter(acled, self.acled_limit)
        if self.acled_exclude is not None:
            acled = self._exclude_filter(acled, self.acled_exclude)

        # Aggregate to admin units if requested
        if aggregate:
            acled = self._aggregate_acled(
                acled_file=out_file,
                agg_file=agg_file,
                exposure_raster=exposure_raster,
                exposure_vector=exposure_vector
            )
    
        return acled


    def _limit_filter(self, data: gpd.GeoDataFrame, limit: dict)  -> gpd.GeoDataFrame:
        """Filter a GeoDataFrame based on specified column or nested column conditions.

        The `limit` dictionary allows filtering on columns directly or on nested subtypes.
    
        Examples of `limit`:
            limit = {
                "disorder_type": {
                    "Strategic developments": {
                        "sub_event_type": ["Looting/property destruction", "Arrests"]
                    }
                }
            }
            limit = {"disorder_type": ["Demonstrations", "Political violence"]}
    
        Args:
            data (gpd.GeoDataFrame): Input GeoDataFrame to filter.
            limit (dict): Dictionary specifying filter conditions on columns and sub-columns.
    
        Returns:
            gpd.GeoDataFrame: Filtered GeoDataFrame.
    
        Raises:
            ValueError: If the `limit` argument is not a dictionary.
            KeyError: If a column or nested subtype specified in `limit` does not exist in `data`.
        """
        
        if limit is None:
            return data

        if not isinstance(limit, dict):
            raise ValueError("`limit` must be a dictionary specifying filter conditions.")
            
        for column in limit:
            if column not in data.columns:
                raise KeyError(f"Column '{column}' specified in limit does not exist in the data.")

            # Handle nested filtering: column -> subtype -> subsubtype -> allowed values
            if isinstance(limit[column], dict):
                for subtype in limit[column]:
                    if subtype not in data[column].unique():
                        raise KeyError(f"Subtype '{subtype}' not found in column '{column}'.")

                    # Split data into rows that match subtype and rows that don't
                    temp1 = data[data[column] != subtype]
                    temp2 = data[data[column] == subtype]

                    # Iterate over subsubtypes and filter rows that match allowed values
                    for subsubtype in limit[column][subtype]:
                        subset = temp2[
                            temp2[subsubtype].isin(limit[column][subtype][subsubtype])
                        ]

                # Combine filtered and unfiltered parts
                data = gpd.GeoDataFrame(pd.concat([temp1, subset]))

            # Handle direct list filtering
            elif isinstance(limit[column], list):
                data = data[data[column].isin(limit[column])]

            else:
                raise ValueError(f"Limit for column '{column}' must be a dict or list.")
                
        return data


    def _exclude_filter(self, data: gpd.GeoDataFrame, exclude: dict):
        """Exclude rows from a GeoDataFrame based on specified column values.

        Each key in the `exclude` dictionary specifies a column, and the corresponding 
        value is a list of values to exclude from that column.
    
        Args:
            data (gpd.GeoDataFrame): Input GeoDataFrame to filter.
            exclude (dict): Dictionary specifying values to exclude per column.
    
        Returns:
            gpd.GeoDataFrame: Filtered GeoDataFrame with specified values removed.
    
        Raises:
            ValueError: If `exclude` is not a dictionary.
            KeyError: If a column specified in `exclude` does not exist in the data.
        """
        
        if exclude is None:
            return data
    
        if not isinstance(exclude, dict):
            raise ValueError("`exclude` must be a dictionary specifying values to exclude.")
            
        for column in exclude:
            if column not in data.columns:
                raise KeyError(f"Column '{column}' specified in exclude does not exist in the data.")

            # Remove rows where the column value matches any value in the exclude list
            data = data[~data[column].isin(exclude[column])]
            
        return data
    

    def _aggregate_acled(
        self,
        acled_file: str,
        agg_file: str,
        exposure_raster: str,
        exposure_vector: str,
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
        column = f"{self.acled_name.lower()}_{self.asset}_exposure"
        if self.overwrite or not os.path.exists(exposure_vector):
            acled_tif = self._calculate_custom_conflict_exposure(acled_file, conflict_src="acled")
            out_tif, _ = self._calculate_exposure(acled_tif, exposure_raster, threshold=1)

            data = self._calculate_zonal_stats(
                out_tif,
                column=column,
                prefix=prefix,
                stats_agg=["sum"],
                out_file=exposure_vector
            )
            if data is None or data.empty:
                raise ValueError("Exposure calculation failed or produced empty results.")

        # Read exposure vector and clean zero values
        exposure_var = prefix + "_" + column
        exposure = gpd.read_file(exposure_vector)
        exposure.loc[exposure[exposure_var] == 0, exposure_var] = None

        # Merge aggregated ACLED and exposure data
        acled = data_utils._merge_data([agg, exposure], columns=self.merge_columns)
    
        return acled


    def _calculate_custom_conflict_exposure(
        self,
        conflict_file: str,
        conflict_src: str = "acled",
        temp_name: str = "temp",
        buffer_size: int = 3000,
        meter_crs: str = "EPSG:3857"
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
            raise FileNotFoundError(f"conflict file not found: {conflict_file}")

        # Helper function to determine buffer size based on event type and fatalities
        def get_buffer_size(event, fatality):
            if (event != "Strategic developments"):
                if (
                    (event == "Riots")
                    | ((event == "Violence against civilians") & (fatality == 0))
                ):
                    return 2000
                return 5000
            return 0

        # Create temporary buffered GeoJSON filename
        filename = os.path.basename(conflict_file).split(".")[0] + f"_{temp_name.upper()}.geojson"
        temp_file = os.path.join(self.local_dir, filename)

        #Create temporary raster file for buffered data
        if self.overwrite or not os.path.exists(temp_file):
            data = gpd.read_file(conflict_file)
            data["values"] = 1

            # Get buffer size depending on conflict data source
            if conflict_src == "acled":
                data["buffer_size"] = data.apply(
                    lambda x: get_buffer_size(x.event_type, x.fatalities), axis=1
                )
            elif conflict_src == "ucdp":
                data["buffer_size"] = buffer_size

            # Apply buffer using meter CRS
            data["geometry"] = data.to_crs(meter_crs).apply(
                lambda x: x.geometry.buffer(x.buffer_size), axis=1
            )
            data = data.set_crs(meter_crs, allow_override=True).to_crs(self.crs)
            data.to_file(temp_file)
            logging.info(f"Temporary file saved to {temp_file}.")

        # Define output raster path
        out_file = os.path.join(self.local_dir, conflict_file.replace(".geojson", ".tif"))

        # Rasterize if raster does not exist
        if self.overwrite or not os.path.exists(out_file):
            try:
                # Create empty raster based on asset template
                with rio.open(self.asset_file) as src:
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
        self,
        acled: gpd.GeoDataFrame,
        agg_file: str
    )  -> gpd.GeoDataFrame:
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
            raise FileNotFoundError("Geoboundary data is missing. Cannot aggregate ACLED events.")

        # Helper function to sum while ignoring all-NaN arrays
        def _nansumwrapper(a, **kwargs):
            if np.isnan(a).all():
                return np.nan
            else:
                return np.nansum(a, **kwargs)

        # Spatial join ACLED events with admin boundaries
        try:
            admin = self.geoboundary
            agg = acled.sjoin(admin, how="left", predicate="intersects")
            agg = agg.drop(["index_right"], axis=1)
        except Exception as e:
            raise ValueError(f"Spatial join failed: {e}")
        
        # Aggregate population sum
        pop_sum = self._aggregate_data(
            agg,
            agg_col="population_best",
            agg_func=lambda x: _nansumwrapper(x),
        )
        pop_sum = pop_sum.rename(
            columns={"population_best": "acled_population_best"}
        )

        # Aggregate total conflict events
        event_count = self._aggregate_data(
            agg, agg_col="conflict_count", agg_func="count"
        )
        event_count = event_count.rename(
            columns={"conflict_count": "acled_conflict_count"}
        )

        # Aggregate total conflict events
        fatalities_count = self._aggregate_data(
            agg, agg_col="fatalities", agg_func="sum"
        )
        fatalities_count = fatalities_count.rename(
            columns={"fatalities": "acled_fatalities"}
        )

        # Aggregate conflict events where population_best is missing
        null_pop_event_count = self._aggregate_data(
            agg[agg["population_best"].isna()],
            agg_col="null_conflict_count",
            agg_func="count",
        )
        null_pop_event_count = null_pop_event_count.rename(
            columns={"null_conflict_count": "acled_null_conflict_count"}
        )

        # Merge all aggregated data with admin boundaries
        agg = data_utils._merge_data(
            [admin, pop_sum, event_count, fatalities_count, null_pop_event_count],
            columns=[f"{self.adm_level}_ID"],
            how="left",
        )

        # Calculate population-weighted conflict exposure
        exposure_var = "acled_exposure"
        agg[exposure_var] = agg["acled_population_best"] / (
            agg["acled_conflict_count"] - agg["acled_null_conflict_count"].fillna(0)
        )
        agg.loc[agg[exposure_var] == 0, exposure_var] = None

        agg = self._calculate_conflict_stats(agg, source="acled")
        

        # Save aggregated GeoDataFrame to file
        agg.to_file(agg_file)
            
        return agg


    def _calculate_conflict_stats(self, data, source: str = "acled"):
        data[f"{source}_fatalities_per_conflict"] = data[f"{source}_fatalities"].div(data[f"{source}_conflict_count"])
        data[f"{source}_fatalities_per_conflict"] = data[f"{source}_fatalities_per_conflict"].replace([np.inf, -np.inf], np.nan)

        return data

        
    
    def _download_url_progress(self, url, output_path):
        with DownloadProgressBar(
            unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]
        ) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

            
    def download_url(self, dataset: str, dataset_name: str = None, ext: str = "tif") -> str:
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
            if dataset == 'gadm':
                url = self.config["urls"][url_name].format(self.iso_code, self.adm_level[-1])
            else:
                url = self.config["urls"][url_name].format(self.iso_code, self.iso_code.lower())
                
        # Check if the dataset is global
        if self.global_name.lower() in dataset:
            # Download if not already present
            if self.overwrite or not os.path.exists(global_file):
                logging.info(f"Downloading {url}...")
                try:
                    if url.endswith(".zip"):
                        self.download_zip(url, dataset, out_file=global_file, ext=ext)
                    elif url.endswith(".tif"):
                        #urllib.request.urlretrieve(url, global_file)
                        self._download_url_progress(url, global_file)
                except Exception as e:
                    raise RuntimeError(f"Failed to download {dataset} from {url}: {e}")

            # Clip raster to country boundary if applicable
            local_file = self._build_filename(self.iso_code, dataset_name, self.local_dir, ext=ext)
            if ext == "tif":
                nodata = self.config.get("nodata", {}).get(dataset, [])
                admin = self.geoboundary.dissolve(by="iso_code")
                self._clip_raster(global_file, local_file, admin, nodata)
    
        else:
            # For non-global datasets, just download locally
            local_file = self._build_filename(self.iso_code, dataset_name, self.local_dir, ext)
            if self.overwrite or not os.path.exists(local_file):
                if url.endswith(".zip"):
                    self.download_zip(url, dataset, out_file=local_file, ext=ext)
                if url.endswith(".tif"):
                    #urllib.request.urlretrieve(url, local_file)
                    self._download_url_progress(url, local_file)
    
        return local_file

        
    def download_zip(self, url: str, dataset: str, out_file: str, ext: str = "tif") -> None:
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
        out_dir = self.global_dir  if 'global' in dataset else self.local_dir
        zip_file = os.path.join(out_dir, f"{dataset}.zip")
        zip_dir = os.path.join(out_dir, dataset)

        # Download and extract ZIP if not already done
        if not os.path.exists(zip_file) and not os.path.exists(zip_dir):
            #urllib.request.urlretrieve(url, zip_file)
            self._download_url_progress(url, zip_file)
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(zip_dir)
            os.remove(zip_file)

        if ext == "tif":
            # Look for GeoTIFF files
            tif_files = [file for file in os.listdir(zip_dir) if file.endswith(".tif")]
            
            if len(tif_files) == 0:
                # If no .tif, convert .grd file to GeoTIFF using GDAL
                grd_files = [file for file in os.listdir(zip_dir) if file.endswith(".grd")]
                if len(grd_files) > 0:
                    grd_file = grd_files[0]
                    tif_file = os.path.join(zip_dir, grd_file.replace(".grd", ".tif"))
                    os.system(f"gdal_translate -a_srs EPSG:4326 {os.path.join(zip_dir, grd_file)} {tif_file}")
            else:
                tif_file = tif_files[0]
        
            #shutil.copyfile(os.path.join(zip_dir, tif_file), out_file)
            os.system(f"gdal_translate -a_srs EPSG:4326 {os.path.join(zip_dir, tif_file)} {out_file}")
            shutil.rmtree(zip_dir)
            
        elif ext == "geojson":
            # Look for GeoJSON files
            geojson_files = [file for file in os.listdir(zip_dir) if file.endswith(".geojson")]

            # If no .geojson, convert from .json feature collection
            if len(geojson_files) == 0:
                json_file = [file for file in os.listdir(zip_dir) if file.endswith(".json")][0]
                json_file = os.path.join(zip_dir, json_file)
                with open(json_file) as data:
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
            csv_files = [file for file in os.listdir(zip_dir) if file.endswith(".csv")]
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
        links = [link["href"] for link in data.find_all("a") if "depth.tif" in link["href"]]
        for link in tqdm(links, total=len(links)):          
            out_file = os.path.join(out_dir, link)
            if not os.path.exists(out_file):
                #urllib.request.urlretrieve(url+link, out_file)
                self._download_url_progress(url+link, out_file)
        
        vrt_file = os.path.join(self.global_dir, f"{name.upper()}.vrt")
        global_file = os.path.join(self.global_dir, f"{name.upper()}.tif")
        
        if not os.path.exists(global_file):
            logging.info(f"Generating flood map. Hang tight, this might take a while...")
            self._merge_tifs(f"{out_dir}/*.tif", vrt_file, global_file)   
            logging.info(f"Flood map saved to {global_file}.")

        local_file = self._build_filename(
            self.iso_code, name.replace(f"global_", ""), self.local_dir, ext="tif"
        )
        if not os.path.exists(local_file):
            admin = self.geoboundary.dissolve(by="iso_code")
            self._clip_raster(global_file, local_file, admin)

        return local_file

    
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
                logging.info(f"({index+1}/{len(folders)}) Processing {folder.lower()}...")

                # File naming
                name = f"{self.iso_code}_{folder}_rp{self.fathom_rp}".upper()
                raw_tif_file = os.path.join(fathom_dir, f"{name}.tif")
                proc_tif_file = os.path.join(self.local_dir, f"{name}.tif")

                # If processed file doesn't exist, build from VRT
                if self.overwrite or not os.path.exists(proc_tif_file):
                    flood_dir = os.path.join(fathom_dir, folder, str(self.fathom_year), f"1in{self.fathom_rp}")
                    merged_file = os.path.join(fathom_dir, f"{name}.vrt")
                    self._merge_tifs(f"{flood_dir}/*.tif", merged_file, raw_tif_file)

                # Clip raster to admin boundary
                admin = self.geoboundary.dissolve(by="iso_code")
                nodata = self.config["nodata"][folder.lower()]
                self._clip_raster(raw_tif_file, proc_tif_file, admin, nodata)

                # Build exposure rasters
                exposure_file = self._build_filename(
                    self.iso_code, f"{folder}_{self.asset}_exposure", self.local_dir, ext="tif"
                )
                weighted_exposure_file = self._build_filename(
                    self.iso_code, f"{folder}_{self.asset}_intensity_weighted_exposure", self.local_dir, ext="tif"
                )
    
                self._generate_exposure(
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
                
                full_data = (
                    data
                    if full_data is None
                    else data_utils._merge_data([full_data, data], columns=self.merge_columns)
                )

                # Add exposure zonal statistics
                if os.path.exists(exposure_file):
                    exposure = self._calculate_zonal_stats(
                        exposure_file,
                        column=folder.lower(),
                        suffix=f"{self.asset}_exposure",
                    )
                    weighted_exposure = self._calculate_zonal_stats(
                        weighted_exposure_file,
                        column=folder.lower(),
                        suffix=f"{self.asset}_intensity_weighted_exposure",
                    )
                    full_data = data_utils._merge_data(
                        [full_data, exposure, weighted_exposure], columns=self.merge_columns
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
            self.iso_code, f"{name}_{self.adm_level}", self.local_dir, ext="geojson"
        )

        # If overwrite is set or file does not exist, regenerate it    
        if self.overwrite or not os.path.exists(full_data_file):
            full_data = None
            
            for index, dataset in enumerate(datasets):
                logging.info(f"({index+1}/{len(datasets)}) Processing {dataset}...")

                # Download raster dataset (GeoTIFF)
                if ('flood' in dataset) and (self.fathom is None) and (self.jrc_version == "v2"):
                    local_file = self.download_jrc(dataset) 
                elif ('flood' in dataset) and (self.fathom is not None):
                    continue
                else:
                    local_file = self.download_url(dataset, ext='tif')  
                
                dataset_name = dataset.replace(f"global_", "")

                # File paths for derived exposures
                exposure_file = self._build_filename(
                    self.iso_code, f"{dataset_name}_{self.asset}_exposure", self.local_dir, ext="tif"
                )
                weighted_exposure_file = self._build_filename(
                    self.iso_code, f"{dataset_name}_{self.asset}_intensity_weighted_exposure", self.local_dir, ext="tif"
                )

                # Generate exposure rasters (skip asset rasters)
                if dataset not in self.config["asset_data"]:
                    self._generate_exposure(
                        local_file, exposure_file, self.config["threshold"][dataset]
                    )

                # Decide on aggregation method
                stats_agg = ["sum"] if dataset == "worldpop" else ["mean"]

                # Zonal statistics for base hazard raster
                data = self._calculate_zonal_stats(
                    local_file,
                    column=dataset_name,
                    stats_agg=stats_agg,
                )

                # Merge into cumulative dataset
                full_data = (
                    data
                    if full_data is None
                    else data_utils._merge_data([full_data, data], columns=self.merge_columns)
                )

                # Add exposure statistics if available
                if os.path.exists(exposure_file):
                    exposure = self._calculate_zonal_stats(
                        exposure_file,
                        column=dataset_name,
                        suffix=f"{self.asset}_exposure",
                    )
                    weighted_exposure = self._calculate_zonal_stats(
                        weighted_exposure_file,
                        column=dataset_name,
                        suffix=f"{self.asset}_intensity_weighted_exposure",
                    )
                    full_data = data_utils._merge_data(
                        [full_data, exposure, weighted_exposure], columns=self.merge_columns
                    )

            # Save merged hazard dataset
            full_data.to_file(full_data_file)
            logging.info(f"Data saved to {full_data_file}.")

        # Always load and return data in correct CRS
        full_data = gpd.read_file(full_data_file).to_crs(self.crs)
        
        return full_data

    
    def _generate_exposure(
        self,
        local_file: str, 
        exposure_file: str, 
        threshold: float
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
            raise FileNotFoundError(f"Input raster file not found: {local_file}")

        # Define the path for the resampled version of the input raster
        resampled_file = local_file.replace(".tif", "_RESAMPLED.tif")

        # Only generate exposure if it hasn't already been computed
        if self.overwrite or not os.path.exists(exposure_file):
            # Resample raster if resampled version does not already exist
            if self.overwrite or not os.path.exists(resampled_file):
                try:
                    self._resample_raster(local_file, resampled_file)
                except Exception as e:
                    raise RuntimeError(f"Failed to resample raster: {e}")

            try:
                self._calculate_exposure(resampled_file, exposure_file, threshold)
            except Exception as e:
                raise RuntimeError(f"Failed to calculate exposure: {e}")

    
    def _resample_raster(self, in_file: str, out_file: str) -> str:
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
        if not os.path.exists(self.asset_file):
            raise FileNotFoundError(f"Asset file not found: {self.asset_file}")

        # Open the asset raster (reference for resolution and bounds)
        asset = gdal.Open(self.asset_file, 0)
        if asset is None:
            raise RuntimeError(f"Failed to open asset raster: {self.asset_file}")

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
        self, hazard_file: str, exposure_file: str, threshold: float
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
            raise FileNotFoundError(f"Hazard raster file not found: {hazard_file}")
        if not os.path.exists(self.asset_file):
            raise FileNotFoundError(f"Asset raster file not found: {self.asset_file}")

        # Open both asset and hazard rasters
        with rio.open(self.asset_file, "r") as src1, rio.open(hazard_file, "r") as src2:
            # Asset raster values
            asset = src1.read(1)
            asset[asset < 0] = 0

            # Hazard raster values
            hazard = src2.read(1)
            if (
                'drought' not in hazard_file.lower() 
                and 'heat_stress' not in hazard_file.lower() 
            ):
                hazard[hazard < 0] = 0

            if 'heat_stress' in hazard_file.lower():
                hazard = hazard / 100

            # Scale hazard values to [0, 1] for weighting
            asset_binary = asset.copy()
            asset_binary[asset_binary > 0] = 1
            hazard_scaled = data_utils._minmax_scale(hazard * asset_binary)

            # Binary raster: hazard above threshold = 1, else 0
            if 'drought' in hazard_file.lower():
                binary = (hazard < threshold).astype(int)
            else:
                binary = (hazard >= threshold).astype(int)

            # Exposure: asset presence masked by hazard exceedance
            exposure = asset * binary

            # Weighted exposure: exposure scaled by hazard intensity
            weighted_exposure = exposure * hazard_scaled          

            # Copy metadata from asset raster to preserve georeferencing
            out_meta = src1.meta.copy()

        # Save binary exposure raster
        binary_file = exposure_file.replace("EXPOSURE", "BINARY")
        with rio.open(binary_file, "w", **out_meta) as dst:
            dst.write(binary, 1)

        # Save intensity-weighted exposure raster
        weighted_exposure_file = exposure_file.replace("EXPOSURE", "INTENSITY_WEIGHTED_EXPOSURE")
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
        agg_name = f"{self.adm_level}_ID"

        if agg_func != "count" and agg_col is None:
            raise ValueError("agg_col must be provided when agg_func is not 'count'.")

        # Perform aggregation
        if agg_func == "count":
            # Count number of rows per admin unit
            agg = data.groupby([agg_name], dropna=False).size().reset_index()
        else:
            # Apply specified aggregation function to agg_col
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
        os.system(f"gdal_translate -co TILED=YES -co COMPRESS=LZW -co BIGTIFF=YES -co NUM_THREADS=ALL_CPUS --config GDAL_CACHEMAX 512 {vrt_file} {tif_file}")

        
    def _clip_raster(
        self,
        global_tif: str, 
        local_tif: str, 
        admin: gpd.GeoDataFrame, 
        nodata: list = []
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
            raise ValueError("Admin GeoDataFrame is empty. Cannot perform clipping.")
    
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
            out_file = os.path.join(self.local_dir, f"{name}_{self.adm_level}.geojson")
    
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
            logging.info(f"Zonal stats saved to {out_file}.")
    
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
            raise ValueError("Prefix, suffix, and local_dir must all be provided and non-empty.")
    
        # Construct and return the full file path
        return os.path.join(out_dir, f"{prefix.upper()}_{suffix.upper()}.{ext}")