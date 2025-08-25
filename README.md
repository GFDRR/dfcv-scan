<div align="center">

# Disaster-FCV Co-location Mapping
Mapping Multi-hazard and Conflict Co-location in Fragile, Conflict, and Violence (FCV)-affected Countries

</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About the Project</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#citation">Citation</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About the Project
We've developed an open‑source, globally applicable toolkit for the rapid mapping and assessment of multi‑hazard and conflict exposure at subnational scales. This toolkit uses globally accessible hazard maps and conflict data to map the spatial distribution of exposure to co-occurring  hazards and conflict events. This work is designed to guide high-level, evidence-based DRM decision-making in FCV contexts, enabling them to efficiently identify priority areas and support more strategic resource allocation at the Disaster–FCV nexus. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

<!-- INSTALLATION -->
### Installation
```sh
pip install dfcv-colocation-mapping
```

### ACLED API
To access ACLED conflict data, you must [register for an ACLED API Key](https://acleddata.com/api-documentation/getting-started).

### OGR/GDAL Installation
To install OGR/GDAL, follow [these instructions](https://ljvmiranda921.github.io/notebook/2019/04/13/install-gdal/).


## Usage

### Demo Notebook
<a target="_blank" href="https://colab.research.google.com/github/GFDRR/disaster-fcv-colocation-mapping/blob/master/examples/demo2.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


### Examples
At minimum, you will need to specify the [ISO 3166-1 alpha-3 code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3) and administrative level (ADM1, ADM2, ADM3, etc.) for your country of interest. 

```py
from dfcv_colocation_mapping import data_download
from dfcv_colocation_mapping import map_utils

dm = data_download.DatasetManager(
    iso_code="RWA", 
    adm_level="ADM3",
    acled_key=<INSERT ACLED KEY HERE>,
    acled_email=<INSERT ACLED EMAIL HERE>,
)
geoplot = map_utils.GeoPlot(dm)
```

### Choropleth Map
To create a choropleth map showing the multi-hazard score, run:
```py
geoplot.plot_choropleth("mhs_exposure_relative");
```

### Bi-variate Choropleth Map
To create bivariate choropleth maps, run:
```py
geoplot.plot_bivariate_choropleth( 
    var1="wbg_conflict_exposure_relative",
    var2="mhs_exposure_relative"
);
```

### Hazard Raster Map
```py
hazard = "earthquake"
geoplot.plot_raster(hazard);
```


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Interested in contributing? Check out the contribution guidelines at `CONTRIBUTION.md`.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the Apache 2.0 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact
Issa Tingzon: tisabelle@worldbank.org or issatingzon@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CITATION -->
## Citation
If you find this repository useful, please consider giving a star ⭐ and citation 🦖:
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

<p align="right">(<a href="#readme-top">back to top</a>)</p>

