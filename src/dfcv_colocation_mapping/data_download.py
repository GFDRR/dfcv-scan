import os
import json
import shutil
import requests
import logging
import datetime

import pandas as pd
import numpy as np

import subprocess
import urllib.request
import zipfile

import geojson
from osgeo import gdal
import geopandas as gpd
import rasterio as rio
import rasterstats
import pycountry

import itertools
import ahpy

import importlib_resources
from dfcv_colocation_mapping import data_utils

logging.basicConfig(level=logging.INFO, force=True)

resources = importlib_resources.files("dfcv_colocation_mapping")
_config_file = resources.joinpath("configs", "config.yaml")
_acled_file = resources.joinpath("configs", "acled_creds.yaml")
_adm_config_file = resources.joinpath("configs", "adm_config.yaml")


class DatasetManager:
    def __init__(
        self,
        iso_code: str,
        adm_source: str = 'geoboundary',
        acled_key: str = None,
        acled_email: str = None,
        acled_start_date: str = None,
        acled_end_date: str = None,
        acled_limit: str = None,
        acled_exclude: str = None,
        acled_country: str = None,
        acled_name: str = "acled",
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
        self.iso_code = iso_code
        self.adm_level = adm_level
        self.data_dir = data_dir
        self.config_file = config_file
        self.acled_file = acled_file
        self.meter_crs = meter_crs
        self.crs = crs
        self.asset = asset
        self.overwrite = overwrite

        self.acled_key = acled_key
        self.acled_email = acled_email
        self.acled_start_date = acled_start_date
        self.acled_end_date = acled_end_date
        self.acled_limit = acled_limit
        self.acled_exclude = acled_exclude
        self.acled_country = acled_country
        
        self.fathom_year = fathom_year
        self.fathom_rp = fathom_rp
        self.fathom_threshold = fathom_threshold

        if config_file is None:
            config_file = _config_file
        if acled_file is None:
            acled_file = _acled_file
        if adm_config_file is None:
            adm_config_file = _adm_config_file
            
        self.config = data_utils.read_config(config_file)

        if os.path.exists(acled_file):
            self.acled_creds = data_utils.read_config(acled_file)
            self.acled_key = self.acled_creds["acled_key"]
            self.acled_email = self.acled_creds["acled_email"]
        
        self.global_name = global_name.upper()
        self.fathom_name = fathom_name.upper()
        self.acled_name = acled_name.upper()

        self.data_dir = os.path.join(os.getcwd(), data_dir)
        self.local_dir = os.path.join(os.getcwd(), data_dir, iso_code)
        self.global_dir = os.path.join(os.getcwd(), data_dir, self.global_name)
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.local_dir, exist_ok=True)
        os.makedirs(self.global_dir, exist_ok=True)

        self.asset_file = self._build_filename(iso_code, asset, self.local_dir, ext="tif")
        self.admin_file = None

        logging.info("Loading geoboundary...")
        try:
            self.geoboundary = self.download_geoboundary(adm_source)
        except:
            logging.info("Loading geoboundaries failed. Trying with GADM...")
            self.geoboundary = self.download_geoboundary("gadm")
        self.merge_columns = list(self.geoboundary.columns)
        
        logging.info("Loading hazard layers...")
        self.hazards = self.download_hazards()
        self.fathom = self.download_fathom()
        
        logging.info("Loading conflict data...")
        self.acled = self.download_acled()
        self.acled_agg = self.download_acled(aggregate=True)

        logging.info("Calculating scores...")
        self.mhs_aggregation = mhs_aggregation
        
        self.data = self.combine_datasets()
        self.data = self.calculate_multihazard_score(self.data)

        self.adm_config = data_utils.read_config(adm_config_file)
        self.data = self.assign_grouping()


    def calculate_ahp(self):
        hazards = self.config["hazards"].keys()
        combinations = list(itertools.combinations(hazards, 2))
        
        weight_dict = dict()
        for combination in combinations:
            weight = input(f"How much more important is {str(combination[0])} compared to {str(combination[1])}: ")
            weight_dict[combination] = weight
        
        hazard_weights = ahpy.Compare(name='Hazards', comparisons=weight_dict, precision=5, random_index='saaty')
        cr = hazard_weights.consistency_ratio
        logging.info(f"Consistency_ratio: {cr}")

        if cr < 0.10:
            self.config["hazards"] = hazard_weights.target_weights
            logging.info(self.config["hazards"])
            self.data = self.calculate_multihazard_score(self.data)
            return self.data
        else:
            raise ValueError(f'Consistency ratio {cr} > 0.10. Please try again.')
                     

    def assign_grouping(self):
        if self.iso_code in self.adm_config:
            config = self.adm_config[self.iso_code]
            group = config["group"]
            if group not in self.data.columns:
                adm_level = config["adm_level"]
                grouping = config["grouping"]
                self.data[group] = self.data[adm_level].map(grouping)
        return self.data

    
    def combine_datasets(self) -> gpd.GeoDataFrame:
        data = []
        for dataset in [self.hazards, self.fathom]:
            if dataset is not None:
                dataset = dataset.mask(dataset.isna(), 0)
                data.append(dataset)

        if self.acled_agg is not None:
            if len(self.acled_agg) > 0:
                data.append(self.acled_agg)

        data = data_utils._merge_data(data, columns=self.merge_columns)    
        return data


    def calculate_multihazard_score(
        self,
        data: gpd.GeoDataFrame,
        conflict_column: str = "wbg_conflict",
        suffixes = ["exposure_relative", "intensity_weighted_exposure_relative"],
        aggregation: str = "power_mean",
        p: float = 0.5,
        epsilon: float = 0.00001
    ):
        for column in data.columns:
            if "relative" not in column:
                colname = f"{column}_relative"
                if "exposure" in column and colname not in data.columns:
                    data[colname] = data[column] / data[self.asset]
    
        for suffix in suffixes:
            total_weight = sum(list(self.config["hazards"].values()))

            if self.mhs_aggregation == "power_mean":
                mhs = 0
                for hazard, weight in self.config["hazards"].items():
                    if total_weight > 0:
                        weight = weight / total_weight
                    
                    if suffix is not None:
                        hazard = f"{hazard}_{suffix}"
    
                    if hazard in data.columns:
                        mhs = mhs + (weight * (data[hazard]) ** p)
                
                mhs = mhs ** (1 / p)
                        
            elif self.mhs_aggregation == "geometric_mean":
                mhs = 1
                for hazard, weight in self.config["hazards"].items():
                    if total_weight > 0:
                        weight = weight / total_weight
                    
                    if suffix is not None:
                        hazard = f"{hazard}_{suffix}"
    
                    if hazard in data.columns:
                        mhs = mhs * ((data[hazard] + epsilon) ** weight)

            elif self.mhs_aggregation == "arithmetic_mean":
                mhs = 0
                for hazard, weight in self.config["hazards"].items():
                    if total_weight > 0:
                        weight = weight / total_weight
                    
                    if suffix is not None:
                        hazard = f"{hazard}_{suffix}"
    
                    if hazard in data.columns:
                        mhs = mhs + ((data[hazard]) * weight)       
    
            mhs_name = "mhs"
            if suffix is not None:
                mhs_name = f"{mhs_name}_{suffix}"
            data[mhs_name] = data_utils._minmax_scale(mhs)
    
            mhsc_name = f"mhs_{conflict_column}"
            if suffix is not None:
                mhsc_name = f"{mhsc_name}_{suffix}"

            if f"{conflict_column}_{suffix}" in data.columns:
                conflict_scaled = data_utils._minmax_scale(data[f"{conflict_column}_{suffix}"])
                data[mhsc_name] = data[mhs_name] * conflict_scaled
    
        return data

    
    def download_geoboundary(self, adm_source: str) -> gpd.GeoDataFrame:
        out_file = self._build_filename(
            self.iso_code, f"{adm_source}_{self.adm_level}", self.local_dir, ext="geojson"
        )    

        gadm_file = self._build_filename(
            self.iso_code, f"gadm_{self.adm_level}", self.local_dir, ext="geojson"
        ) 
        if os.path.exists(gadm_file):
            adm_source = "gadm"
            out_file = gadm_file

        if self.overwrite or not os.path.exists(out_file):
            logging.info(f"Downloading geoboundary for {self.iso_code}...")

            if adm_source == 'gadm':
                self.download_url(adm_source, dataset_name=f"{adm_source}_{self.adm_level}", ext="geojson")
                geoboundary = gpd.read_file(out_file)
                
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
                
            elif adm_source == "geoboundary":
                gbhumanitarian_url = self.config["urls"]["gbhumanitarian_url"]
                gbopen_url = self.config["urls"]["gbopen_url"]
                level = int(self.adm_level[-1])

                datasets = []
                for index in range(1, level+1):
                    adm_level = f"ADM{index}"
                    intermediate_file = self._build_filename(
                        self.iso_code, adm_level, self.local_dir, ext="geojson"
                    )
                    url = f"{gbhumanitarian_url}{self.iso_code}/{adm_level}/"
                    try:
                        r = requests.get(url)
                        download_path = r.json()["gjDownloadURL"]
                    except Exception:
                        # Fallback to GBOpen URL if GBHumanitarian URL fails
                        url = f"{gbopen_url}{self.iso_code}/{adm_level}/"
                        r = requests.get(url)
                        download_path = r.json()["gjDownloadURL"]
            
                    # Download and save the GeoJSON data
                    logging.info(f"Downloading from {download_path}...")
                    geoboundary = requests.get(download_path).json()
                    with open(intermediate_file, "w") as file:
                        geojson.dump(geoboundary, file)
            
                    # Read the downloaded GeoJSON into a GeoDataFrame
                    geoboundary = gpd.read_file(intermediate_file)
                    geoboundary["iso_code"] = self.iso_code
            
                    # Select relevant columns and rename them
                    geoboundary = geoboundary[["iso_code", "shapeName", "shapeID", "geometry"]]
                    geoboundary.columns = ["iso_code", adm_level, f"{adm_level}_ID", "geometry"]
                    
                    datasets.append(geoboundary)
                    geoboundary.to_file(intermediate_file)
                    logging.info(f"Geoboundary file saved to {intermediate_file}.")

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
                
            geoboundary.to_crs(self.crs).to_file(out_file)
            logging.info(f"Geoboundary file saved to {out_file}.")

        self.admin_file = out_file
        self.adm_source = adm_source
        geoboundary = gpd.read_file(out_file).to_crs(self.crs)
        self.merge_columns = list(geoboundary.columns)
        return geoboundary

    
    def download_acled(self, population: str = "full", aggregate: bool = False,) -> gpd.GeoDataFrame:
        country = pycountry.countries.get(alpha_3=self.iso_code)

        out_file = self._build_filename(self.iso_code, self.acled_name, self.local_dir, ext="geojson")
        agg_file = self._build_filename(self.iso_code, f"{self.acled_name}_{self.adm_level}", self.local_dir, ext="geojson")
        exposure_raster = self._build_filename(self.iso_code, f"{self.acled_name}_exposure", self.local_dir, ext="tif")
        exposure_vector = self._build_filename(
            self.iso_code, f"{self.acled_name}_exposure_{self.adm_level}", self.local_dir, ext="geojson"
        )
    
        if self.overwrite or not os.path.exists(out_file):
            if self.acled_end_date is None:
                self.acled_end_date = datetime.date.today().isoformat()
    
            logging.info(f"Downloading ACLED data for {self.iso_code}...")
    
            params = dict(
                key=self.acled_key,
                email=self.acled_email,
                country=country.name,
                event_date=f"{self.acled_start_date}|{self.acled_end_date}",
                event_date_where="BETWEEN",
                population=population,
                page=1,
            )
            acled_url = self.config["urls"]["acled_url"]
            len_subdata = -1
            data = []
            while len_subdata != 0:
                logging.info(f"Reading ACLED page {params['page']}...")
                response = requests.get(acled_url, params=params)
                subdata = pd.DataFrame(response.json()["data"])
                data.append(subdata)
                len_subdata = len(subdata)
                params["page"] = params["page"] + 1
    
            data = pd.concat(data)
            if len(data) == 0:
                return
                
            data = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(data["longitude"], data["latitude"], crs=self.crs),
                data=data,
            )
            data["population_best"] = (
                data["population_best"].replace({"": np.nan}).astype(np.float64)
            )
            data["disorder_type"] = data["disorder_type"].replace(
                {
                    "Political violence; Demonstrations": "Demonstrations",
                }
            )
            logging.info(f"ACLED file saved to {out_file}.")
            data.to_file(out_file)
    
        acled = gpd.read_file(out_file).to_crs(self.crs)
    
        if self.acled_limit is not None:
            acled = self._limit_filter(acled, self.acled_limit)
    
        if self.acled_exclude is not None:
            acled = self._exclude_filter(acled, self.acled_exclude)
    
        if aggregate:
            admin = self.geoboundary
            acled = self._aggregate_acled(
                acled_file=out_file,
                agg_file=agg_file,
                exposure_raster=exposure_raster,
                exposure_vector=exposure_vector
            )
    
        return acled


    def _limit_filter(self, data: gpd.GeoDataFrame, limit: dict):
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


    def _exclude_filter(self, data: gpd.GeoDataFrame, exclude: dict):
        for column in exclude:
            data = data[~data[column].isin(exclude[column])]
        return data
    

    def _aggregate_acled(
        self,
        acled_file: str,
        agg_file: str,
        exposure_raster: str,
        exposure_vector: str,
        prefix: str = "wbg",
        column: str = "conflict_exposure"
    ):
        acled = gpd.read_file(acled_file)
        if not os.path.exists(agg_file):
            self._aggregate_acled_exposure(acled, agg_file)
        agg = gpd.read_file(agg_file)

        if not os.path.exists(exposure_vector):
            acled_tif = self._calculate_custom_acled_exposure(acled_file)
            out_tif, _ = self._calculate_exposure(acled_tif, exposure_raster, threshold=1)
            data = self._calculate_zonal_stats(
                out_tif,
                column=column,
                prefix=prefix,
                stats_agg=["sum"],
                out_file=exposure_vector
            )
        exposure_var = prefix + "_" + column
        exposure = gpd.read_file(exposure_vector)
        exposure.loc[exposure[exposure_var] == 0, exposure_var] = None
        acled = data_utils._merge_data([agg, exposure], columns=self.merge_columns)
    
        return acled


    def _calculate_custom_acled_exposure(
        self,
        acled_file: str,
        temp_name: str = "temp",
        meter_crs: str = "EPSG:3857"
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
    
        filename = os.path.basename(acled_file).split(".")[0] + f"_{temp_name.upper()}.geojson"
        temp_file = os.path.join(self.local_dir, filename)
    
        if not os.path.exists(temp_file):
            data = gpd.read_file(acled_file)
            data["values"] = 1
            data["buffer_size"] = data.apply(
                lambda x: get_buffer_size(x.event_type, x.fatalities), axis=1
            )
            data["geometry"] = data.to_crs(meter_crs).apply(
                lambda x: x.geometry.buffer(x.buffer_size), axis=1
            )
            data = data.set_crs(meter_crs, allow_override=True).to_crs(self.crs)
            data.to_file(temp_file)
    
        out_file = os.path.join(self.local_dir, acled_file.replace(".geojson", ".tif"))
        if not os.path.exists(out_file):
            with rio.open(self.asset_file) as src:
                out_image = src.read(1)
                out_image = np.zeros(out_image.shape)
    
                out_meta = src.meta.copy()
                with rio.open(out_file, "w", **out_meta) as dest:
                    dest.write(out_image, 1)
    
            os.system(f"gdal_rasterize -burn 1 {temp_file} {out_file}")
        return out_file


    def _aggregate_acled_exposure(
        self,
        acled: gpd.GeoDataFrame,
        agg_file: str
    ):
        def _nansumwrapper(a, **kwargs):
            if np.isnan(a).all():
                return np.nan
            else:
                return np.nansum(a, **kwargs)

        admin = self.geoboundary
        agg = acled.sjoin(admin, how="left", predicate="intersects")
        agg = agg.drop(["index_right"], axis=1)
    
        pop_sum = self._aggregate_data(
            agg,
            agg_col="population_best",
            agg_func=lambda x: _nansumwrapper(x),
        )
        event_count = self._aggregate_data(
            agg, agg_col="conflict_count", agg_func="count"
        )
        null_pop_event_count = self._aggregate_data(
            agg[agg["population_best"].isna()],
            agg_col="null_conflict_count",
            agg_func="count",
        )
        agg = data_utils._merge_data(
            [admin, pop_sum, event_count, null_pop_event_count],
            columns=[f"{self.adm_level}_ID"],
            how="left",
        )
        exposure_var = "acled_conflict_exposure"
        agg[exposure_var] = agg["population_best"] / (
            agg["conflict_count"] - agg["null_conflict_count"].fillna(0)
        )
        agg.loc[agg[exposure_var] == 0, exposure_var] = None
        agg.to_file(agg_file)
        return agg

    
    def download_url(self, dataset: str, dataset_name: str = None, ext: str = "tif"):  
        if dataset_name is None:
            dataset_name = dataset.replace(f"{self.global_name.lower()}_", "")
        global_file = self._build_filename(
            self.global_name, dataset_name, self.global_dir, ext="tif"
        )
        url_name = f"{dataset}_url"
        if url_name in self.config["urls"]:
            if dataset == 'gadm':
                url = self.config["urls"][url_name].format(self.iso_code, self.adm_level[-1])
            else:
                url = self.config["urls"][url_name].format(self.iso_code, self.iso_code.lower())
                
            logging.info(f"Downloading {url}...")
    
        if self.global_name.lower() in dataset:
            if not os.path.exists(global_file):
                if url.endswith(".zip"):
                    self.download_zip(url, dataset, out_file=global_file, ext=ext)
                elif url.endswith(".tif"):
                    urllib.request.urlretrieve(url, global_file)
    
            local_file = self._build_filename(self.iso_code, dataset_name, self.local_dir, ext="tif")
    
            nodata = []
            if dataset in self.config["nodata"]:
                nodata = self.config["nodata"][dataset]
            admin = self.geoboundary.dissolve(by="iso_code")
            data_utils._clip_raster(global_file, local_file, admin, nodata)
    
        else:
            local_file = self._build_filename(self.iso_code, dataset_name, self.local_dir, ext)
            if not os.path.exists(local_file):
                if url.endswith(".zip"):
                    self.download_zip(url, dataset, out_file=local_file, ext=ext)
                if url.endswith(".tif"):
                    urllib.request.urlretrieve(url, local_file)
    
        return local_file

        
    def download_zip(self, url: str, dataset: str, out_file: str, ext: str = "tif"):
        out_dir = self.global_dir  if 'global' in dataset else self.local_dir
        zip_file = os.path.join(out_dir, f"{dataset}.zip")
        zip_dir = os.path.join(out_dir, dataset)
    
        if not os.path.exists(zip_file):
            urllib.request.urlretrieve(url, zip_file)
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(zip_dir)
            os.remove(zip_file)

        if ext == "tif":
            tif_files = [file for file in os.listdir(zip_dir) if file.endswith(".tif")]
            if len(tif_files) == 0:
                grd_file = [file for file in os.listdir(zip_dir) if file.endswith(".grd")][0]
                tif_file = os.path.join(zip_dir, grd_file.replace(".grd", ".tif"))
                subprocess.run(
                    [
                        "gdal_translate",
                        "-a_srs",
                        "EPSG:4326",
                        os.path.join(zip_dir, grd_file),
                        tif_file,
                    ]
                )
            else:
                tif_file = tif_files[0]
        
            shutil.copyfile(os.path.join(zip_dir, tif_file), out_file)
            shutil.rmtree(zip_dir)
            
        elif ext == "geojson":
            geojson_files = [file for file in os.listdir(zip_dir) if file.endswith(".geojson")]
            if len(geojson_files) == 0:
                json_file = [file for file in os.listdir(zip_dir) if file.endswith(".json")][0]
                json_file = os.path.join(zip_dir, json_file)
                with open(json_file) as data:
                    features = json.load(data)["features"]
                geojson = gpd.GeoDataFrame.from_features(features)
                geojson = geojson.set_crs(self.crs)
                geojson.to_file(out_file)
            else:
                geojson_file = geojson_files[0]
                shutil.copyfile(os.path.join(zip_dir, geojson_file), out_file)
            shutil.rmtree(zip_dir)

    
    def download_fathom(self):
        fathom_folder = f"{self.iso_code}_{self.fathom_name}".upper()
        fathom_dir = os.path.join(self.local_dir, fathom_folder)
    
        full_data_file = self._build_filename(
            self.iso_code, self.fathom_name, self.local_dir, ext="geojson"
        )
        os.path.join(self.local_dir, f"{self.iso_code}_{self.fathom_name}_{self.adm_level}.geojson")

        if not os.path.exists(fathom_dir):
            return None
    
        if not os.path.exists(full_data_file):
            full_data = None
            folders = next(os.walk(fathom_dir))[1]
            for index, folder in enumerate(folders):
                logging.info(f"({index+1}/{len(folders)}) Processing {folder.lower()}...")
                name = f"{self.iso_code}_{folder}_rp{self.fathom_rp}".upper()
                raw_tif_file = os.path.join(fathom_dir, f"{name}.tif")
                proc_tif_file = os.path.join(self.local_dir, f"{name}.tif")
    
                if not os.path.exists(proc_tif_file):
                    flood_dir = os.path.join(fathom_dir, folder, str(self.fathom_year), f"1in{self.fathom_rp}")
                    merged_file = os.path.join(fathom_dir, f"{name}.vrt")
    
                    subprocess.call(
                        ["gdalbuildvrt", merged_file, f"{flood_dir}/*.tif"], shell=True
                    )
                    subprocess.call(
                        ["gdal_translate", "-co", "TILED=YES", merged_file, raw_tif_file],
                        shell=True,
                    )
    
                admin = self.geoboundary.dissolve(by="iso_code")
                nodata = self.config["nodata"][folder.lower()]
                data_utils._clip_raster(raw_tif_file, proc_tif_file, admin, nodata)
    
                exposure_file = self._build_filename(
                    self.iso_code, f"{folder}_exposure", self.local_dir, ext="tif"
                )
                weighted_exposure_file = self._build_filename(
                    self.iso_code, f"{folder}_intensity_weighted_exposure", self.local_dir, ext="tif"
                )
    
                self._generate_exposure(
                    proc_tif_file,
                    exposure_file,
                    self.config["threshold"][folder.lower()],
                )
    
                def custom(x):
                    return np.sum(x > self.fathom_threshold) / x.size
    
                add_stats = {"custom": custom}
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
                if os.path.exists(exposure_file):
                    exposure = self._calculate_zonal_stats(
                        exposure_file,
                        column=folder.lower(),
                        suffix="exposure",
                    )
                    weighted_exposure = self._calculate_zonal_stats(
                        weighted_exposure_file,
                        column=folder.lower(),
                        suffix="intensity_weighted_exposure",
                    )
                    full_data = data_utils._merge_data(
                        [full_data, exposure, weighted_exposure], columns=self.merge_columns
                    )
    
            full_data.to_file(full_data_file)
    
        full_data = gpd.read_file(full_data_file).to_crs(self.crs)
        return full_data


    def download_hazards(self, datasets: list = None):
        if datasets is None:
            datasets = self.config["datasets"]
    
        full_data_file = self._build_filename(
            self.iso_code, f"hazards_{self.adm_level}", self.local_dir, ext="geojson"
        )
    
        if self.overwrite or not os.path.exists(full_data_file):
            full_data = None
            for index, dataset in enumerate(datasets):
                logging.info(f"({index+1}/{len(datasets)}) Downloading {dataset}...")
                local_file = self.download_url(dataset, ext='tif')
                dataset_name = dataset.replace(f"global_", "")
                exposure_file = self._build_filename(
                    self.iso_code, f"{dataset_name}_exposure", self.local_dir, ext="tif"
                )
                weighted_exposure_file = self._build_filename(
                    self.iso_code, f"{dataset_name}_intensity_weighted_exposure", self.local_dir, ext="tif"
                )
    
                if dataset != self.asset:
                    self._generate_exposure(
                        local_file, exposure_file, self.config["threshold"][dataset]
                    )
    
                stats_agg = ["sum"] if dataset == "worldpop" else ["mean"]
                data = self._calculate_zonal_stats(
                    local_file,
                    column=dataset_name,
                    stats_agg=stats_agg,
                )
    
                full_data = (
                    data
                    if full_data is None
                    else data_utils._merge_data([full_data, data], columns=self.merge_columns)
                )
                if os.path.exists(exposure_file):
                    exposure = self._calculate_zonal_stats(
                        exposure_file,
                        column=dataset_name,
                        suffix="exposure",
                    )
                    weighted_exposure = self._calculate_zonal_stats(
                        weighted_exposure_file,
                        column=dataset_name,
                        suffix="intensity_weighted_exposure",
                    )
                    full_data = data_utils._merge_data(
                        [full_data, exposure, weighted_exposure], columns=self.merge_columns
                    )
            full_data.to_file(full_data_file)
    
        full_data = gpd.read_file(full_data_file).to_crs(self.crs)
        return full_data

    
    def _generate_exposure(
        self,
        local_file: str, 
        exposure_file: str, 
        threshold: float
    ) -> None:
        resampled_file = f"{local_file.split('.')[0]}_RESAMPLED.tif"
    
        if not os.path.exists(exposure_file):
            if not os.path.exists(resampled_file):
                self._resample_raster(local_file, resampled_file)
    
            self._calculate_exposure(resampled_file, exposure_file, threshold)

    
    def _resample_raster(self, in_file: str, out_file: str) -> str:
        asset = gdal.Open(self.asset_file, 0)
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

    
    def _calculate_exposure(
        self, hazard_file: str, exposure_file: str, threshold: float
    ) -> str:
        with rio.open(self.asset_file, "r") as src1, rio.open(hazard_file, "r") as src2:
            asset = src1.read(1)
            asset[asset < 0] = 0
            
            hazard = src2.read(1)
            hazard[hazard < 0] = 0
            
            hazard_scaled = data_utils._minmax_scale(hazard)
    
            binary = (hazard >= threshold).astype(int)
            exposure = asset * binary
            
            weighted_exposure = exposure * hazard_scaled          
            
            out_meta = src1.meta.copy()
    
        binary_file = exposure_file.replace("EXPOSURE", "BINARY")
        with rio.open(binary_file, "w", **out_meta) as dst:
            dst.write(binary, 1)

        weighted_exposure_file = exposure_file.replace("EXPOSURE", "INTENSITY_WEIGHTED_EXPOSURE")
        with rio.open(weighted_exposure_file, "w", **out_meta) as dst:
            dst.write(weighted_exposure, 1)
    
        with rio.open(exposure_file, "w", **out_meta) as dst:
            dst.write(exposure, 1)

        return exposure_file, weighted_exposure_file


    def _aggregate_data(
        self,
        data: gpd.GeoDataFrame,
        agg_col: str = None,
        agg_func: str = "sum",
    ) -> gpd.GeoDataFrame:
    
        agg_name = f"{self.adm_level}_ID"
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
    
        name = os.path.basename(in_file).split(".")[0]
        
        if out_file is None:
            out_file = os.path.join(self.local_dir, f"{name}_{self.adm_level}.geojson")
    
        if not os.path.exists(out_file):
            admin_file = self.admin_file
            admin = self.geoboundary
            original_crs = admin.crs
    
            with rio.open(in_file) as src:
                if admin.crs != src.crs:
                    admin = admin.to_crs(src.crs)
                    admin.to_file(admin_file)
    
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


    def _build_filename(self, prefix, suffix, local_dir, ext="geojson"):
        return os.path.join(local_dir, f"{prefix.upper()}_{suffix.upper()}.{ext}")
    