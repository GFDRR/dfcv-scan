<div align="center">

# DFCV-Scan: Disaster-FCV Co-location Mapping
Mapping Multi-hazard and Conflict Co-location in Fragile, Conflict, and Violence (FCV)-affected Countries

</div>

<!-- ABOUT THE PROJECT -->
## Overview
D-FCV Scan is a Python library to support the rapid mapping and assessment of multi‑hazard and conflict exposure at subnational scales.

![title](https://github.com/GFDRR/dfcv-scan/blob/master/assets/figure.png?raw=true)

This library automates the download and processing of globally accessible asset, hazard, conflict, and displacement data, with the goal of mapping the spatial distribution of co-occurring multi-hazard and FCV exposure. This work aims to guide high-level, evidence-based DRM decision-making in FCV contexts and enable them to efficiently identify priority areas for more strategic resource allocation at the Disaster–FCV nexus.

<!-- TABLE OF CONTENTS -->
## Table of Contents
  <ol>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#quick-start">Quick Start</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#usage-examples">Usage Examples</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#citation">Citation</a></li>
  </ol>

<!-- GETTING STARTED -->
## Installation

```sh
pip install dfcv-colocation-mapping
```

#### GDAL Installation
To install GDAL, run:
```sh
conda install gdal
```

For Linux systems: 
```sh
apt install gdal-bin
```

## Tutorial Notebook
<a target="_blank" href="https://colab.research.google.com/github/GFDRR/dfcv-scan/blob/master/notebooks/demo2.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


## Quick Start

```py
from dfcv_colocation_mapping import data_download
from dfcv_colocation_mapping import map_utils
from dfcv_colocation_mapping import widgets

# Instantiate dataset manager
dm = data_download.DatasetManager(
    iso_code="SLE",
    adm_level="ADM3",
    dtm_key=DTM_API_KEY,
    idmc_key=IDMC_API_KEY,
    acled_username=ACLED_USERNAME,
    acled_password=ACLED_PASSWORD,
    conflict_start_date="2020-01-01",
    conflict_end_date="2025-12-31",
    displacement_start_year=2020,
    displacement_end_year=2025
)

# Download datasets
dm.download_datasets()
geoplot = map_utils.GeoPlot(dm)

# Plot assets
widget = widgets.MapWidget(
    geoplot=geoplot,
    var_list=dm.asset_names,
    var_label="Asset",
    out_dir="assets"
)
widget.show()
```

## Features
- <b>Multi-Source Data Integration</b>: Combine admin boundaries (e.g., GADM, geoBoundaries), population (e.g., WorldPop, LandScan), infrastructure (e.g., WorldCover), hazards (e.g., CDRI, UNEP, JRC), conflict data (e.g., ACLED, UCDP), displacement data (e.g. IOM DTM, IDMC GIDD), and OSM layers using a unified data manager.
- <b>Interactive Geospatial Visualization</b>: Build static (PNG) and interactive (HTML) maps for exploring spatial data directly in Jupyter
- <b>Choropleth and Bi-variate Choropleth Mapping</b>: Generate customizable choropleth and bi-variate choropleth maps or countries, regions, and subregions
- <b>Hazard & Conflict Exposure Analysis</b>: Support for hazard exposure, conflict exposure, and multi-hazard exposure calculation with configurable aggregation methods.
- <b>Save & Reproducibility Support</b>: Built-in controls to customize maps and export plots to an output directory.
- <b>Extensible Widget Architecture</b>: Modular widget classes designed for easy extension.

## Usage Examples
### Download Data
```py
dm = data_download.DatasetManager(
    iso_code="SLE",                   # 3-letter ISO-code
    adm_level="ADM3",                 # Administrative level 
    dtm_key="DTM_API_KEY",            # IOM DTM API key
    idmc_key="IDMC_API_KEY",          # IDMC API key
    acled_username="ACLED_USERNAME",  # ACLED username
    acled_password="ACLED_PASSWORD",  # ACLED password
    conflict_start_date="2020-01-01", # Conflict start date
    conflict_end_date="2025-12-31",   # Conflict end date
    displacement_start_year=2020,     # Displacement start year
    displacement_end_year=2025        # Displacement end year
)
dm.download_datasets()
```

#### Parameters

- **iso_code**: ISO country code.
- **adm_level**: Administrative level (default: `ADM3`).
- **adm_source**: Source of administrative boundaries (default: `geoboundaries`).
- **crs**: Coordinate reference system (default: `EPSG:4326`).
- **acled_username**: ACLED API username (default: `None`).
- **acled_password**: ACLED API password (default: `None`).
- **conflict_start_date**: Conflict start date (default: defaults to the last 5 years).
- **conflict_end_date**: Conflict end date (defaults to the current date).
- **conflict_last_n_years**: Lookback window for conflict data (default: `5`).
- **dtm_key**: DTM API key (default: `None`).
- **dtm_adm_level**: Admin level for DTM data (only `ADM1` or `ADM2`).
- **idmc_key**: IDMC API key (default: `None`).
- **displacement_start_year**: Displacement start year (defaults to the last 5 years).
- **displacement_end_year**: Displacement end year (defaults current date).
- **displacement_last_n_years**: Lookback window for displacement data (default: `5`).
- **mhs_aggregation**: Aggregation method for MHS indicators (default: `arithmetic_mean`; other values: `geometric_mean`, `power_mean`, `quadratic_mean`).
- **config_file**: Main data config YAML file (defaults to [data_config.yaml](https://github.com/GFDRR/dfcv-scan/blob/master/src/dfcv_colocation_mapping/configs/data_config.yaml)).
- **osm_config_file**: OSM config file (defaults to [osm_config.yaml](https://github.com/GFDRR/dfcv-scan/blob/master/src/dfcv_colocation_mapping/configs/osm_config.yaml)).
- **acled_config_file**: ACLED config file (defaults to [acled_config.yaml](https://github.com/GFDRR/dfcv-scan/blob/master/src/dfcv_colocation_mapping/configs/acled_config.yaml)).
- **data_dir**: Base data directory (default: `data`).

### Customize Datasets to Download
To specify which datasets to download, specify the dataset category, run the following code, and save your selection before calling `dm.download_datasets()`. 
```py
category = "assets" # Choose from: "assets", "hazards", "conflict", "displacement", "osm"

def save_selection(result):
    dm.config[f"{category}_selected"] = result
    dm.set_selected_datasets()

selector = widgets.MultiSelectorWidget(
    category,
    dm.config[f"{category}_all"],
    dm.config[f"{category}_selected"],
    save_callback=save_selection
)
selector.show()
```

### Customize ACLED Categories
To specify which ACLED categories to include, specify the asset category, run the following code, and save your selection before calling `dm.download_datasets()`.
```py
asset_category = "population"  # @param ["population", "infrastructure", "agriculture"]

def save_selection(result):
    dm.acled_selected[asset_category] = result

selector = widgets.HierarchicalCheckboxes(
    asset_category,
    hierarchy=dm.acled_hierarchy,
    selected=dm.acled_selected[asset_category],
    save_callback=save_selection,
    save_label="Save ACLED Selection"
)
selector.show()
```

### Create GeoPlot Instance
```py
geoplot = map_utils.GeoPlot(
  dm=dm,                  # Dataset Manager instance
  map_config_file=None    # Mapping config file
)
```
#### Parameters
- **dm**: Dataset Manager instance.
- **map_config_file**: Mapping config file (defaults to [map_config.yaml](https://github.com/GFDRR/dfcv-scan/blob/master/src/dfcv_colocation_mapping/configs/map_config.yaml)).


### Map Creation with Widgets
```py
widget = widgets.MapWidget(
    geoplot=geoplot,    
    var_list=None,              # List of variables to map; if None, the dropdown will contain all variables
    map_mode="choropleth",      # Map display mode ("choropleth" or "bivariate_choropleth").
    zoom_to_region=False,       # Whether to zoom map to selected region 
    plot_conflict=False,        # Plot aggregated conflict data, i.e. ACLED or UCDP
    plot_conflict_points=False, # Plot disaggregated conflict data points
    plot_displacement=False,    # Plot aggregated displacement data, i.e. IDMC GIDD or IOM DTM
    plot_displacement_points=False, # Plot disaggregated displacement data points 
    plot_hazard_exposure=False, # Plot single hazard exposure map
    plot_mhs_exposure=False     # Plot multi-hazard exposure map
    plot_osm_networks=False,    # Plot OSM networks, e.g. roads, railways, waterways
    plot_osm_points=False,      # Plot OSM point-of-interest data, e.g. hospitals, schools, banks
    out_dir="OUTPUT_DIRECTORY"  # Output directory name (saves a static PNG, an interactive HTML folium map, and a CSV)
)
widget.show()
```

#### Parameters
- **geoplot**: Geoplot object containing the data manager and datasets.
- **map_mode**: Map display mode (default: `choropleth`).
- **var_list**: List of variables to choose from (optional).
- **var_label**: Label for the variable dropdown (default: `Variable:`).
- **zoom_to_region**: Whether to zoom map to selected region (default: `False`).
- **overwrite_titles**: Allow overwriting default titles (default: `False`).
- **plot_displacement**: Plot displacement data (default: `False`).
- **plot_displacement_points**: Plot IDP points (default: `False`).
- **plot_conflict**: Plot conflict data (default: `False`).
- **plot_conflict_points**: Plot conflict points (default: `False`).
- **plot_conflict_exposure**: Plot conflict exposure map (default: `False`).
- **plot_hazard_exposure**: Plot hazard exposure map (default: `False`).
- **plot_mhs_exposure**: Plot multi-hazard exposure map (default: `False`).
- **plot_osm_points**: Plot OpenStreetMap point data (default: `False`).
- **plot_osm_networks**: Plot OpenStreetMap network data (default: `False`).
- **out_dir**: Output directory for saving plots (optional).

Note: To overlay the different maps, you may set multiple parameters to `True`. 


<!-- CONTRIBUTING -->
## Contributing

Interested in contributing? Check out the contribution guidelines at `CONTRIBUTION.md`.

<!-- LICENSE -->
## License

Distributed under the Apache 2.0 License. See `LICENSE.txt` for more information.

<!-- CITATION -->
## Citation

```
@misc{tingzon2025mapping,
  title={Mapping Multi-hazard and Conflict Co-location in Fragile, Conflict, and Violence (FCV)-affected Countries},
  author={Tingzon, Isabelle},
  year={2025},
  organization={The World Bank Group},
  type={Tutorial},
  howpublished={\url{https://github.com/GFDRR/disaster-fcv-colocation-mapping}}
}
```