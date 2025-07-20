import geopandas as gpd

from utils import data_utils
from utils import mhs_utils


def calculate_multihazard_score(
    data: gpd.GeoDataFrame,
    config_file: dict = None,
    conflict_column: str = "dfcv_conflict",
    suffixes = ["exposure_relative", "exposure"]
):
    if config_file is None:
        config = data_utils.read_config(config_file)

    for suffix in suffixes:
        mhs, total_weight = 0, 0
        for hazard, weight in config["weights"].items():
            if suffix is not None:
                hazard = f"{hazard}_{suffix}"

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
