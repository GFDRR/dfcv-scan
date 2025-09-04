<div align="center">

# Disaster-FCV Co-location Mapping
Mapping Multi-hazard and Conflict Co-location in Fragile, Conflict, and Violence (FCV)-affected Countries

</div>

<!-- ABOUT THE PROJECT -->
## About the Project
We've developed an open‑source, globally applicable toolkit for the rapid mapping and assessment of multi‑hazard and conflict exposure at subnational scales. This toolkit uses globally accessible hazard maps and conflict data to map the spatial distribution of co-occurring multi-hazard and conflict exposure. Our work is designed to guide high-level, evidence-based DRM decision-making in FCV contexts and enable them to efficiently identify priority areas for more strategic resource allocation at the Disaster–FCV nexus. 

<!-- TABLE OF CONTENTS -->
## Table of Contents
  <ol>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#accessing-acled">Accessing ACLED</a></li>
    <li><a href="#installing-ogr/gdal">Installing OGR/GDAL</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#references">References</a></li>
    <li><a href="#citation">Citation</a></li>
  </ol>

<!-- GETTING STARTED -->
## Installation

```sh
pip install dfcv-colocation-mapping
```

## Accessing ACLED
To access ACLED conflict data, you must register for an ACLED API Key. Please refer to the official [ACLED Access Guide](https://acleddata.com/methodology/acled-access-guide) for instructions.

## Installing OGR/GDAL 
The simplest way to install OGR/GDAL is to execute the following command:
```sh
conda install gdal
```

Alternatively, if you're using Linux, run the command: 
```sh
apt install gdal-bin
```

For more information on how to install OGR/GDAL, see [this guide](https://ljvmiranda921.github.io/notebook/2019/04/13/install-gdal/).


## Getting Started

### Demo Notebook
<a target="_blank" href="https://colab.research.google.com/github/GFDRR/disaster-fcv-colocation-mapping/blob/master/examples/demo2.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


### Example Usage
At minimum, you will need to specify the country's [ISO code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3) and administrative level (e.g. ADM1, ADM2, etc.). If you have an ACLED API key, you may also  specify it here. 

```py
from dfcv_colocation_mapping import data_download
from dfcv_colocation_mapping import map_utils

dm = data_download.DatasetManager(
    iso_code="RWA", 
    adm_level="ADM3",
    acled_key=ACLED_API_KEY,
    acled_email=ACLED_EMAIL,
)
geoplot = map_utils.GeoPlot(dm)
```

### AHP Calculation [Optional]
For the calculation of the Multi-hazard Score (MHS), you can generate custom weights per hazard using Analytic Hierarchy Process (AHP), as implemented in [AHPy](https://github.com/PhilipGriffith/AHPy). 
```py
dm.calculate_ahp()
```

### Geoboundaries Map
```py
geoplot.plot_geoboundaries(
    adm_level="ADM2", 
    group="ADM1",
)
```

![geoboundaries](https://github.com/GFDRR/disaster-fcv-colocation-mapping/blob/master/assets/NPL_geoboundaries.png?raw=true)

### Hazard Raster Map
```py
geoplot.plot_raster("earthquake")
```
![raster](https://github.com/GFDRR/disaster-fcv-colocation-mapping/blob/master/assets/NPL_raster.png?raw=true)


### Choropleth Map
```py
geoplot.plot_choropleth("earthquake_worldpop_exposure_relative")
```
![choropleth](https://github.com/GFDRR/disaster-fcv-colocation-mapping/blob/master/assets/NPL_choropleth.png?raw=true)

### Choropleth Map Zoomed
```py
geoplot.plot_choropleth(
    var=f"wbg_acled_worldpop_exposure_relative",
    zoom_to={"ADM1: "Lumbini"},
)
```
![choropleth](https://github.com/GFDRR/disaster-fcv-colocation-mapping/blob/master/assets/NPL_choropleth_zoomed.png?raw=true)



### Bi-variate Choropleth Map
```py
geoplot.plot_bivariate_choropleth( 
    var1="wbg_acled_worldpop_exposure_relative",
    var2="mhs_worldpop_exposure_intensity_weighted_relative"
)
```
![choropleth](https://github.com/GFDRR/disaster-fcv-colocation-mapping/blob/master/assets/NPL_bivariate_choropleth.png?raw=true)


### Bi-variate Choropleth Map Zoomed
```py
geoplot.plot_bivariate_choropleth( 
    var1="wbg_acled_worldpop_exposure_relative",
    var2="mhs_worldpop_exposure_intensity_weighted_relative",
    zoom_to={"ADM1: "Bagmati"},
)
```
![choropleth](https://github.com/GFDRR/disaster-fcv-colocation-mapping/blob/master/assets/NPL_bivariate_choropleth_zoomed.png?raw=true)


## Data Sources

### Administrative Boundaries
- [GADM](https://gadm.org)
- [geoBoundaries](https://www.geoboundaries.org)

### Asset Data
- [WorldPop Population Estimates](www.worldpop.org)

### Conflict Data
- [Armed Conflict Location and Event Data (ACLED)](https://acleddata.com/)
- [Uppsala Conflict Data Program (UCDP)](https://ucdp.uu.se/downloads/)

### Hazard Data
- [Global Landslide Hazard Map](https://datacatalog.worldbank.org/search/dataset/0037584)
- [Global Seismic Hazard Map (475-year RP)](https://www.globalquakemodel.org/product/global-seismic-hazard-map)
- [Global Model of Cyclone Wind (100-year RP)](https://data.humdata.org/dataset/cyclone-wind-100-years-return-period)
- [Standardized Precipitation and Evapotranspiration Index (SPEI)](https://www.drought.gov/data-download)
- [Flood Hazard Map of the World (100-year RP)](https://data.jrc.ec.europa.eu/dataset/jrc-floods-floodmapgl_rp100y-tif)
- [GloUTCI-M: A Global monthly 1 km Universal Thermal Climate Index](https://zenodo.org/records/8310513) 



<small>*RP = return period</small>

<!-- CONTRIBUTING -->
## Contributing

Interested in contributing? Check out the contribution guidelines at `CONTRIBUTION.md`.


<!-- LICENSE -->
## License

Distributed under the Apache 2.0 License. See `LICENSE.txt` for more information.




<details>
  <summary> <h2>References</h2></summary>

- GADM, https://gadm.org
- WorldPop, www.worldpop.org
- geoBoundaries, https://www.geoboundaries.org
- Runfola, D. et al. (2020) geoBoundaries: A global database of political administrative boundaries. PLoS ONE 15(4): e0231866. https://doi.org/10.1371/journal.pone.0231866
- ACLED, “Armed Conflict Location & Event Data (ACLED) Codebook,” 3 October 2024. www.acleddata.com.
- Clionadh Raleigh, Roudabeh Kishi, and Andrew Linke, “Political instability patterns are obscured by conflict dataset scope conditions, sources, and coding choices,” Humanities and Social Sciences Communications, 25 February 2023. https://doi.org/10.1057/s41599-023-01559-4
- Davies, S., Pettersson, T., Sollenberg, M., & Öberg, M. (2025). Organized violence 1989–2024, and the challenges of identifying civilian victims. Journal of Peace Research, 62(4). https://ucdp.uu.se/downloads
- Sundberg, Ralph and Erik Melander (2013) Introducing the UCDP Georeferenced Event Dataset. Journal of Peace Research 50(4).
- Bondarenko M., Kerr D., Sorichetta A., and Tatem, A.J. 2020. Census/projection-disaggregated gridded population datasets, adjusted to match the corresponding UNPD 2020 estimates, for 183 countries in 2020 using Built-Settlement Growth Model (BSGM) outputs. WorldPop, University of Southampton, UK. doi:10.5258/SOTON/WP00685
- K. Johnson, M. Villani, K. Bayliss, C. Brooks, S. Chandrasekhar, T. Chartier, Y. Chen, J. Garcia-Pelaez, R. Gee, R. Styron, A. Rood, M. Simionato, M. Pagani (2023). Global Earthquake Model (GEM) Seismic Hazard Map (version 2023.1 - June 2023), DOI: https://doi.org/10.5281/zenodo.8409647
- United Nations Office for Disaster Risk Reduction (UNDRR) (n.d.). Global model of cyclone wind 50, 100, 250, 500 and 1000 years return period. Humanitarian Data Exchange (HDX). https://data.humdata.org/dataset/cyclone-wind-100-years-return-period
- The World Bank Group (n.d.). Global landslide hazard map. World Bank Data Catalog. Creative Commons Attribution-Non Commercial 4.0 license. https://datacatalog.worldbank.org/search/dataset/0037584
- National Integrated Drought Information System (NIDIS) (n.d.). Drought.gov Data Download (GIS and Web-Ready) [web page]. U.S. Drought Portal. https://www.drought.gov/data-download
- Zhiwei Yang, Jian Peng, & Yanxu Liu. (2023). GloUTCI-M: A Global Monthly 1 km Universal Thermal Climate Index Dataset from 2000 to 2022 [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8310513
-  Yang, Z., Peng, J., Liu, Y., Jiang, S., Cheng, X., Liu, X., Dong, J., Hua, T., and Yu, X.: GloUTCI-M: A Global monthly 1 km Universal Thermal Climate Index dataset from 2000 to 2022, Earth Syst. Sci. Data, 16, 2407–2424, https://doi.org/10.5194/essd-16-2407-2024, 2024.
</details>



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