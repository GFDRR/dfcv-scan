import os
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
import geopandas as gpd
import pycountry

from utils import data_utils
from utils import mhs_utils

logging.basicConfig(level=logging.INFO, force=True)


class DatasetManager:
    def __init__(
        self,
        iso_code: str,
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
        config_file: str = None,
        crs: str = "EPSG:4326",
        asset: str = "worldpop",
        global_name: str = "global",
        overwrite: bool = False,
    ):
        self.iso_code = iso_code
        self.adm_level = adm_level
        self.data_dir = data_dir
        self.config_file = config_file
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
        self.config = data_utils.read_config(config_file)
        
        self.global_name = global_name.upper()
        self.fathom_name = fathom_name.upper()
        self.acled_name = acled_name.upper()

        self.data_dir = os.path.join(os.getcwd(), data_dir)
        self.out_dir = os.path.join(os.getcwd(), data_dir, iso_code)
        self.global_dir = os.path.join(os.getcwd(), data_dir, global_name)
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.global_dir, exist_ok=True)

        self.asset_file = data_utils._build_filename(iso_code, asset, self.out_dir, ext="tif")
        self.merge_columns = ["iso_code", adm_level, f"{adm_level}_ID", "geometry"]

        self.hazards = None
        self.fathom = None
        self.acled = None
        self.acled_agg = None


    def generate_datasets(self) -> gpd.GeoDataFrame:
        self.geoboundary = self.download_geoboundary()
    
        data = []
        self.hazards = self.download_hazards()
        self.fathom = self.download_fathom()

        for dataset in [self.hazards, self.fathom]:
            if dataset is not None:
                dataset = dataset.mask(dataset.isna(), 0)
                data.append(dataset)

        self.acled = self.download_acled()
        self.acled_agg = self.download_acled(aggregate=True)
        if self.acled_agg is not None:
            data.append(self.acled_agg)
    
        data = data_utils._merge_data(data, columns=self.merge_columns)
        for column in data.columns:
            if "exposure" in column:
                data[f"{column}_relative"] = data[column] / data[self.asset]
    
        mhs_utils.calculate_multihazard_score(data)
    
        return data

    
    def download_geoboundary(self) -> gpd.GeoDataFrame:
        out_file = data_utils._build_filename(
            self.iso_code, self.adm_level, self.out_dir, ext="geojson"
        )
    
        gbhumanitarian_url = self.config["urls"]["gbhumanitarian_url"]
        gbopen_url = self.config["urls"]["gbopen_url"]
    
        if self.overwrite or not os.path.exists(out_file):
            logging.info(f"Downloading geoboundary for {self.iso_code}...")
            url = f"{gbhumanitarian_url}{self.iso_code}/{self.adm_level}/"
            try:
                r = requests.get(url)
                download_path = r.json()["gjDownloadURL"]
            except Exception:
                # Fallback to GBOpen URL if GBHumanitarian URL fails
                url = f"{gbopen_url}{self.iso_code}/{self.adm_level}/"
                r = requests.get(url)
                download_path = r.json()["gjDownloadURL"]
    
            # Download and save the GeoJSON data
            geoboundary = requests.get(download_path).json()
            with open(out_file, "w") as file:
                geojson.dump(geoboundary, file)
    
            # Read the downloaded GeoJSON into a GeoDataFrame
            geoboundary = gpd.read_file(out_file)
            geoboundary["iso_code"] = self.iso_code
    
            # Select relevant columns and rename them
            geoboundary = geoboundary[["iso_code", "shapeName", "shapeID", "geometry"]]
            geoboundary.columns = ["iso_code", self.adm_level, f"{self.adm_level}_ID", "geometry"]
            logging.info(f"Geoboundary file saved to {out_file}.")
            geoboundary.to_file(out_file, engine="fiona")
    
        geoboundary = gpd.read_file(out_file).to_crs(self.crs)
        return geoboundary

    
    def download_acled(self, population: str = "full", aggregate: bool = False,) -> gpd.GeoDataFrame:
        country = pycountry.countries.get(alpha_3=self.iso_code)

        out_file = data_utils._build_filename(self.iso_code, self.acled_name, self.out_dir, ext="geojson")
        agg_file = data_utils._build_filename(
            self.iso_code, f"{self.acled_name}_{self.adm_level}", self.out_dir, ext="geojson"
        )
        exposure_raster = data_utils._build_filename(
            self.iso_code, f"{self.acled_name}_exposure", self.out_dir, ext="tif"
        )
        exposure_vector = data_utils._build_filename(
            self.iso_code, f"{self.acled_name}_exposure_{self.adm_level}", self.out_dir, ext="geojson"
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
            acled = data_utils._limit_filter(acled, self.acled_limit)
    
        if self.acled_exclude is not None:
            acled = data_utils._exclude_filter(acled, self.acled_exclude)
    
        if aggregate:
            admin = self.download_geoboundary()
            acled = data_utils._aggregate_acled(
                admin=admin,
                acled_file=out_file,
                asset_file=self.asset_file,
                agg_file=agg_file,
                exposure_raster=exposure_raster,
                exposure_vector=exposure_vector,
                adm_level=self.adm_level,
            )
    
        return acled

    
    def download_url(self, dataset: str):    
        dataset_name = dataset.replace(f"{self.global_name.lower()}_", "")
        global_file = data_utils._build_filename(
            self.global_name, dataset_name, self.global_dir, ext="tif"
        )
    
        url_name = f"{dataset}_url"
        if url_name in self.config["urls"]:
            url = self.config["urls"][url_name].format(self.iso_code, self.iso_code.lower())
    
        if self.global_name.lower() in dataset:
            if not os.path.exists(global_file):
                if url.endswith(".zip"):
                    self.download_zip(url, dataset, out_file=global_file)
                elif url.endswith(".tif"):
                    urllib.request.urlretrieve(url, global_file)
    
            local_file = data_utils._build_filename(self.iso_code, dataset_name, self.out_dir, ext="tif")
    
            nodata = []
            if dataset in self.config["nodata"]:
                nodata = self.config["nodata"][dataset]
            admin = self.download_geoboundary().dissolve(by="iso_code")
            data_utils._clip_raster(global_file, local_file, admin, nodata)
    
        else:
            local_file = data_utils._build_filename(self.iso_code, dataset, self.out_dir, ext="tif")
    
            if not os.path.exists(local_file):
                if url.endswith(".tif"):
                    urllib.request.urlretrieve(url, local_file)
    
        return local_file

        
    def download_zip(self, url: str, dataset: str, out_file: str):
        zip_file = os.path.join(self.global_dir, f"{dataset}.zip")
        zip_dir = os.path.join(self.global_dir, dataset)
    
        if not os.path.exists(zip_file):
            urllib.request.urlretrieve(url, zip_file)
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(zip_dir)
            os.remove(zip_file)
    
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

    
    def download_fathom(self):
        fathom_folder = f"{self.iso_code}_{self.fathom_name}".upper()
        fathom_dir = os.path.join(self.out_dir, fathom_folder)
    
        full_data_file = data_utils._build_filename(
            self.iso_code, self.fathom_name, self.global_dir, ext="geojson"
        )
        os.path.join(self.out_dir, f"{self.iso_code}_{self.fathom_name}_{self.adm_level}.geojson")

        if not os.path.exists(fathom_dir):
            return None
    
        if not os.path.exists(full_data_file):
            full_data = None
            folders = next(os.walk(fathom_dir))[1]
            for index, folder in enumerate(folders):
                logging.info(f"({index+1}/{len(folders)}) Processing {folder.lower()}...")
                name = f"{self.iso_code}_{folder}_rp{self.fathom_rp}".upper()
                raw_tif_file = os.path.join(fathom_dir, f"{name}.tif")
                proc_tif_file = os.path.join(self.out_dir, f"{name}.tif")
    
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
    
                admin = self.download_geoboundary().dissolve(by="iso_code")
                nodata = self.config["nodata"][folder.lower()]
                data_utils._clip_raster(raw_tif_file, proc_tif_file, admin, nodata)
    
                exposure_file = data_utils._build_filename(
                    self.iso_code, f"{folder}_exposure", self.out_dir, ext="tif"
                )
    
                data_utils._generate_exposure(
                    self.asset_file,
                    proc_tif_file,
                    exposure_file,
                    self.config["threshold"][folder.lower()],
                )
    
                def custom(x):
                    return np.sum(x > self.fathom_threshold) / x.size
    
                add_stats = {"custom": custom}
                data = data_utils._calculate_zonal_stats(
                    proc_tif_file,
                    adm_level=self.adm_level,
                    column=folder.lower(),
                    add_stats=add_stats,
                )
                full_data = (
                    data
                    if full_data is None
                    else data_utils._merge_data([full_data, data], columns=self.merge_columns)
                )
                if os.path.exists(exposure_file):
                    exposure = data_utils._calculate_zonal_stats(
                        exposure_file,
                        adm_level=self.adm_level,
                        column=folder.lower(),
                        suffix="exposure",
                    )
                    full_data = data_utils._merge_data(
                        [full_data, exposure], columns=self.merge_columns
                    )
    
            full_data.to_file(full_data_file)
    
        full_data = gpd.read_file(full_data_file).to_crs(self.crs)
        return full_data


    def download_hazards(self, datasets: list = None):
        if datasets is None:
            datasets = self.config["datasets"]
    
        full_data_file = data_utils._build_filename(
            self.iso_code, f"hazards_{self.adm_level}", self.out_dir, ext="geojson"
        )
    
        if self.overwrite or not os.path.exists(full_data_file):
            full_data = None
            for index, dataset in enumerate(datasets):
                logging.info(f"({index+1}/{len(datasets)}) Downloading {dataset}...")
                local_file = self.download_url(dataset)
                dataset_name = dataset.replace(f"global_", "")
                exposure_file = data_utils._build_filename(
                    self.iso_code, f"{dataset_name}_exposure", self.out_dir, ext="tif"
                )
    
                if dataset != self.asset:
                    data_utils._generate_exposure(
                        self.asset_file, local_file, exposure_file, self.config["threshold"][dataset]
                    )
    
                stats_agg = ["sum"] if dataset == "worldpop" else ["mean"]
                data = data_utils._calculate_zonal_stats(
                    local_file,
                    adm_level=self.adm_level,
                    column=dataset_name,
                    stats_agg=stats_agg,
                )
    
                full_data = (
                    data
                    if full_data is None
                    else data_utils._merge_data([full_data, data], columns=self.merge_columns)
                )
                if os.path.exists(exposure_file):
                    exposure = data_utils._calculate_zonal_stats(
                        exposure_file,
                        adm_level=self.adm_level,
                        column=dataset_name,
                        suffix="exposure",
                    )
                    full_data = data_utils._merge_data(
                        [full_data, exposure], columns=self.merge_columns
                    )
    
            full_data.to_file(full_data_file)
    
        full_data = gpd.read_file(full_data_file).to_crs(self.crs)
        return full_data
