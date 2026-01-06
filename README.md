<div align="center">

# DFCV-Scan: Disaster-FCV Co-location Mapping
Mapping Multi-hazard and Conflict Co-location in Fragile, Conflict, and Violence (FCV)-affected Countries

</div>

<!-- ABOUT THE PROJECT -->
## About the Project
D-FCV Scan is an open‑source, globally applicable tool for the rapid mapping and assessment of multi‑hazard and conflict exposure at subnational scales.

![title](https://github.com/GFDRR/dfcv-scan/blob/master/assets/figure.png?raw=true)

Our tool automates the download and processing of globally accessible asset, hazard, conflict, and displacement data, with the goal of mapping the spatial distribution of co-occurring multi-hazard and FCV exposure. Our work is designed to guide high-level, evidence-based DRM decision-making in FCV contexts and enable them to efficiently identify priority areas for more strategic resource allocation at the Disaster–FCV nexus.

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

### GDAL Installation
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
- <b>Multi-Source Data Integration</b>: Combine administrative boundaries, population assets, hazards, conflict data, displacement data, and OpenStreetMap layers from a unified data manager.
- <b>Interactive Geospatial Visualization</b>: Build rich, interactive map widgets for exploring spatial datasets directly in your Jupyter notebooks
- <b>Choropleth and Bi-variate Choropleth Mapping</b>: Generate customizable choropleth and bi-variate choropleth maps with zoom-in features
- <b>Hazard & Conflict Exposure Analysis</b>: Support for hazard exposure, conflict exposure, and multi-hazard (MHS) exposure with configurable aggregation methods.
- <b>Save & Reproducibility Support</b>: Built-in controls to save selections and export plots to a configurable output directory.
- <b>Extensible Widget Architecture</b>: Modular widget classes designed for easy extension.

## Usage Examples
You can overlay the different maps by setting multiple parameters to `True`. 
```py
widget = widgets.MapWidget(
    geoplot=geoplot,    
    var_list=None               # List of variables to map; if None, the dropdown will contain all variables
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
    out_dir="OUTPUT_DIRECTORY_NAME" # Output directory name (saves a static PNG, an interactive HTML folium map, and a CSV)
)
widget.show()
```

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