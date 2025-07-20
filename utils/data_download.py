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

logging.basicConfig(level=logging.INFO)


def generate_datasets(
    iso_code: str,
    acled_key: str = None,
    acled_email: str = None,
    acled_start_date: str = None,
    acled_end_date: str = None,
    acled_limit: str = None,
    acled_exclude: str = None,
    acled_country: str = None,
    fathom_year: int = 2020,
    fathom_rp: int = 50,
    fathom_threshold: int = 50,
    adm_level: str = "ADM3",
    datasets: list = None,
    data_dir: str = "data",
    config_file: str = None,
    crs: str = "EPSG:4326",
    asset: str = "worldpop",
    overwrite: bool = False,
):
    merge_columns = ["iso_code", adm_level, f"{adm_level}_ID", "geometry"]
    download_geoboundary(
        iso_code=iso_code,
        adm_level=adm_level,
        config_file=config_file,
        crs=crs,
        data_dir=data_dir,
        overwrite=overwrite,
    )

    data = []
    hazards = download_hazards(
        iso_code=iso_code,
        adm_level=adm_level,
        datasets=datasets,
        data_dir=data_dir,
        config_file=config_file,
        crs=crs,
        asset=asset,
    )
    if hazards is not None:
        hazards = hazards.mask(hazards.isna(), 0)
        data.append(hazards)

    fathom = download_fathom(
        iso_code,
        year=fathom_year,
        rp=fathom_rp,
        threshold=fathom_threshold,
        adm_level=adm_level,
        data_dir=data_dir,
        config_file=config_file,
        crs=crs,
        asset=asset,
    )
    if fathom is not None:
        fathom = fathom.mask(fathom.isna(), 0)
        data.append(fathom)

    acled = download_acled(
        iso_code=iso_code,
        acled_key=acled_key,
        acled_email=acled_email,
        start_date=acled_start_date,
        end_date=acled_end_date,
        limit=acled_limit,
        exclude=acled_exclude,
        country=acled_country,
        config_file=config_file,
        crs=crs,
        data_dir=data_dir,
        adm_level=adm_level,
        aggregate=True,
        overwrite=overwrite,
        asset=asset,
    )
    if acled is not None:
        data.append(acled)

    data = data_utils._merge_data(data, columns=merge_columns)
    for column in data.columns:
        if "exposure" in column:
            data[f"{column}_relative"] = data[column] / data[asset]

    mhs_utils.calculate_multihazard_score(data)

    return data


def download_hazards(
    iso_code: str,
    adm_level: str = "ADM3",
    datasets: list = None,
    data_dir: str = "data",
    config_file: str = None,
    crs: str = "EPSG:4326",
    asset: str = "worldpop",
    overwrite: bool = False,
):
    config = data_utils.read_config(config_file)
    if datasets is None:
        datasets = config["datasets"]

    out_dir = os.path.join(os.getcwd(), data_dir, iso_code)
    os.makedirs(out_dir, exist_ok=True)

    full_data_file = os.path.join(out_dir, f"{iso_code}_HAZARDS_{adm_level}.geojson")

    merge_columns = ["iso_code", adm_level, f"{adm_level}_ID", "geometry"]
    if not os.path.exists(full_data_file) or overwrite is True:
        full_data = None
        for index, dataset in enumerate(datasets):
            logging.info(f"({index+1}/{len(datasets)}) Downloading {dataset}...")
            local_file = download_url(iso_code, config, dataset, adm_level)
            dataset_name = dataset.replace(f"global_", "")
            exposure_file = os.path.join(
                out_dir, f"{iso_code}_{dataset_name.upper()}_EXPOSURE.tif"
            )

            if dataset != asset:
                asset_file = os.path.join(out_dir, f"{iso_code}_{asset.upper()}.tif")
                data_utils._generate_exposure(
                    asset_file, local_file, exposure_file, config["threshold"][dataset]
                )

            stats_agg = ["sum"] if dataset == "worldpop" else ["mean"]
            data = data_utils._calculate_zonal_stats(
                local_file,
                adm_level=adm_level,
                column=dataset_name,
                stats_agg=stats_agg,
            )

            full_data = (
                data
                if full_data is None
                else data_utils._merge_data([full_data, data], columns=merge_columns)
            )
            if os.path.exists(exposure_file):
                exposure = data_utils._calculate_zonal_stats(
                    exposure_file,
                    adm_level=adm_level,
                    column=dataset_name,
                    suffix="exposure",
                )
                full_data = data_utils._merge_data(
                    [full_data, exposure], columns=merge_columns
                )

        full_data.to_file(full_data_file, engine="fiona")

    full_data = gpd.read_file(full_data_file).to_crs(crs)
    return full_data


def download_fathom(
    iso_code: str,
    year: int = 2020,
    rp: int = 50,
    threshold: float = 50,
    adm_level: str = "ADM3",
    data_dir: str = "data",
    config_file: str = None,
    crs: str = "EPSG:4326",
    asset: str = "worldpop",
    fathom_name: str = "fathom",
):
    config = data_utils.read_config(config_file)
    out_dir = os.path.join(os.getcwd(), data_dir, iso_code)
    os.makedirs(out_dir, exist_ok=True)

    fathom_folder = f"{iso_code}_{fathom_name}".upper()
    fathom_dir = os.path.join(out_dir, fathom_folder)

    full_data_file = os.path.join(out_dir, f"{iso_code}_FATHOM_{adm_level}.geojson")
    merge_columns = ["iso_code", adm_level, f"{adm_level}_ID", "geometry"]

    if not os.path.exists(fathom_dir):
        return None

    if not os.path.exists(full_data_file):
        full_data = None
        folders = next(os.walk(fathom_dir))[1]
        for index, folder in enumerate(folders):
            logging.info(f"({index+1}/{len(folders)}) Downloading {folder.lower()}...")
            name = f"{iso_code}_{folder}_rp{rp}".upper()
            raw_tif_file = os.path.join(fathom_dir, f"{name}.tif")
            proc_tif_file = os.path.join(out_dir, f"{name}.tif")

            if not os.path.exists(proc_tif_file):
                flood_dir = os.path.join(fathom_dir, folder, str(year), f"1in{rp}")
                merged_file = os.path.join(fathom_dir, f"{name}.vrt")

                subprocess.call(
                    ["gdalbuildvrt", merged_file, f"{flood_dir}/*.tif"], shell=True
                )
                subprocess.call(
                    ["gdal_translate", "-co", "TILED=YES", merged_file, raw_tif_file],
                    shell=True,
                )

            admin = download_geoboundary(iso_code, adm_level).dissolve(by="iso_code")
            nodata = config["nodata"][folder.lower()]
            data_utils._clip_raster(raw_tif_file, proc_tif_file, admin, nodata)

            asset_file = os.path.join(out_dir, f"{iso_code}_{asset.upper()}.tif")
            exposure_file = os.path.join(out_dir, f"{iso_code}_{folder}_EXPOSURE.tif")

            data_utils._generate_exposure(
                asset_file,
                proc_tif_file,
                exposure_file,
                config["threshold"][folder.lower()],
            )

            def custom(x):
                return np.sum(x > threshold) / x.size

            add_stats = {"custom": custom}
            data = data_utils._calculate_zonal_stats(
                proc_tif_file,
                adm_level=adm_level,
                column=folder.lower(),
                add_stats=add_stats,
            )
            full_data = (
                data
                if full_data is None
                else data_utils._merge_data([full_data, data], columns=merge_columns)
            )
            if os.path.exists(exposure_file):
                exposure = data_utils._calculate_zonal_stats(
                    exposure_file,
                    adm_level=adm_level,
                    column=folder.lower(),
                    suffix="exposure",
                )
                full_data = data_utils._merge_data(
                    [full_data, exposure], columns=merge_columns
                )

        full_data.to_file(full_data_file, engine="fiona")

    full_data = gpd.read_file(full_data_file).to_crs(crs)
    return full_data


def download_zip(url: str, dataset: str, out_dir: str, out_file: str):
    zip_file = os.path.join(out_dir, f"{dataset}.zip")
    zip_dir = os.path.join(out_dir, dataset)

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
        tif_file = raster_files[0]

    shutil.copyfile(os.path.join(zip_dir, tif_file), out_file)
    shutil.rmtree(zip_dir)


def download_url(
    iso_code: str,
    config: dict,
    dataset: str,
    adm_level: str = "ADM3",
    data_dir: str = "data",
    global_name: str = "global",
):
    out_dir = os.path.join(os.getcwd(), data_dir)
    os.makedirs(out_dir, exist_ok=True)

    global_dir = os.path.join(out_dir, global_name.upper())
    os.makedirs(global_dir, exist_ok=True)

    local_dir = os.path.join(out_dir, iso_code)
    os.makedirs(local_dir, exist_ok=True)

    global_file = os.path.join(global_dir, f"{dataset.upper()}.tif")

    url_name = f"{dataset}_url"
    if url_name in config["urls"]:
        url = config["urls"][url_name].format(iso_code, iso_code.lower())

    if global_name in dataset:
        if not os.path.exists(global_file):
            if url.endswith(".zip"):
                download_zip(url, dataset, out_dir=global_dir, out_file=global_file)
            elif url.endswith(".tif"):
                urllib.request.urlretrieve(url, global_file)

        dataset_name = dataset.replace(f"{global_name}_", "")
        local_file = os.path.join(local_dir, f"{iso_code}_{dataset_name.upper()}.tif")

        nodata = []
        if dataset in config["nodata"]:
            nodata = config["nodata"][dataset]
        admin = download_geoboundary(iso_code, adm_level).dissolve(by="iso_code")
        data_utils._clip_raster(global_file, local_file, admin, nodata)

    else:
        local_file = os.path.join(local_dir, f"{iso_code}_{dataset.upper()}.tif")

        if not os.path.exists(local_file):
            if url.endswith(".tif"):
                urllib.request.urlretrieve(url, local_file)

    return local_file


def download_acled(
    iso_code: str,
    acled_key: str = None,
    acled_email: str = None,
    start_date: str = None,
    end_date: str = None,
    limit: dict = None,
    exclude: dict = None,
    country: str = None,
    population: str = "full",
    config_file: str = None,
    crs: str = "EPSG:4326",
    data_dir: str = "data",
    adm_level: str = "ADM3",
    aggregate: bool = False,
    overwrite: bool = False,
    asset: str = "worldpop",
) -> gpd.GeoDataFrame:

    config = data_utils.read_config(config_file)
    if country is None:
        country = pycountry.countries.get(alpha_3=iso_code)

    out_dir = os.path.join(os.getcwd(), data_dir, iso_code)
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, f"{iso_code}_ACLED.geojson")
    agg_file = os.path.join(out_dir, f"{iso_code}_ACLED_{adm_level}.geojson")
    exposure_raster = os.path.join(out_dir, f"{iso_code}_ACLED_EXPOSURE.tif")
    exposure_vector = os.path.join(
        out_dir, f"{iso_code}_ACLED_EXPOSURE_{adm_level}.geojson"
    )
    asset_file = os.path.join(out_dir, f"{iso_code}_{asset.upper()}.tif")

    if not os.path.exists(out_file) or overwrite is True:
        if end_date is None:
            end_date = datetime.date.today().isoformat()

        logging.info(f"Downloading ACLED data for {iso_code}...")

        params = dict(
            key=acled_key,
            email=acled_email,
            country=country.name,
            event_date=f"{start_date}|{end_date}",
            event_date_where="BETWEEN",
            population=population,
            page=1,
        )
        acled_url = config["urls"]["acled_url"]
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
            geometry=gpd.points_from_xy(data["longitude"], data["latitude"], crs=crs),
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
        logging.info(f"acled file saved to {out_file}.")
        data.to_file(out_file)

    acled = gpd.read_file(out_file).to_crs(crs)

    if limit is not None:
        acled = data_utils._limit_filter(acled, limit)

    if exclude is not None:
        acled = data_utils._exclude_filter(acled, exclude)

    if aggregate:
        admin = download_geoboundary(iso_code, adm_level)
        acled = data_utils._aggregate_acled(
            admin=admin,
            acled_file=out_file,
            asset_file=asset_file,
            agg_file=agg_file,
            exposure_raster=exposure_raster,
            exposure_vector=exposure_vector,
            adm_level=adm_level,
        )

    return acled


def download_geoboundary(
    iso_code: str,
    adm_level: str = "ADM3",
    config_file: str = None,
    crs: str = "EPSG:4326",
    data_dir: str = "data",
    overwrite: bool = False,
) -> gpd.GeoDataFrame:

    config = data_utils.read_config(config_file)
    out_dir = os.path.join(os.getcwd(), data_dir, iso_code)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{iso_code}_{adm_level}.geojson")

    gbhumanitarian_url = config["urls"]["gbhumanitarian_url"]
    gbopen_url = config["urls"]["gbopen_url"]

    if not os.path.exists(out_file) or overwrite is True:
        logging.info(f"Downloading geoboundary for {iso_code}...")
        url = f"{gbhumanitarian_url}{iso_code}/{adm_level}/"
        try:
            r = requests.get(url)
            download_path = r.json()["gjDownloadURL"]
        except Exception:
            # Fallback to GBOpen URL if GBHumanitarian URL fails
            url = f"{gbopen_url}{iso_code}/{adm_level}/"
            r = requests.get(url)
            download_path = r.json()["gjDownloadURL"]

        # Download and save the GeoJSON data
        geoboundary = requests.get(download_path).json()
        with open(out_file, "w") as file:
            geojson.dump(geoboundary, file)

        # Read the downloaded GeoJSON into a GeoDataFrame
        geoboundary = gpd.read_file(out_file)
        geoboundary["iso_code"] = iso_code

        # Select relevant columns and rename them
        geoboundary = geoboundary[["iso_code", "shapeName", "shapeID", "geometry"]]
        geoboundary.columns = ["iso_code", adm_level, f"{adm_level}_ID", "geometry"]
        logging.info(f"Geoboundary file saved to {out_file}.")
        geoboundary.to_file(out_file, engine="fiona")

    geoboundary = gpd.read_file(out_file).to_crs(crs)
    return geoboundary
