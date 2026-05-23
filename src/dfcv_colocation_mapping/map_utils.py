import re
import os
import math
import copy
import logging
import warnings
from datetime import datetime
import importlib_resources

import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerPatch
from matplotlib.legend import Legend
from shapely.geometry import Polygon, MultiPolygon

import colormaps as cmaps
import rasterio as rio
import geopandas as gpd
import geojson_rewind
import pandas as pd
import numpy as np
import pyproj
import pycountry
import folium
import json

import humanize
from stop_words import get_stop_words

import seaborn as sns
import pypalettes
import pyfonts

from rasterio.plot import show
import rasterio.mask

from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

from dfcv_colocation_mapping import common_utils
from vincenty import vincenty

from folium.plugins import TimestampedGeoJson
from shapely.geometry import mapping
from folium import MacroElement
from jinja2 import Template


WARNING = "\033[31m"
RESET = "\033[0m"

REGULAR_FONT = pyfonts.load_google_font("Roboto")
BOLD_FONT = pyfonts.load_google_font("Roboto", weight="bold")


class GeoPlot:
    def __init__(self, dm, map_config_file: str = None):
        """
        Initializes a GeoPlot instance.

        Args:
            dm: Data manager object that contains the dataset (`dm.data`).
            data_dir (str, optional): Path to the data directory. (default: 'data/')
            map_config_file (str, optional): Path to a YAML map configuration file.
        """

        # Load data manager and configs
        self.dm = dm
        resources = importlib_resources.files("dfcv_colocation_mapping")
        self.map_config_file = map_config_file or resources.joinpath(
            "configs", "map_config.yaml"
        )
        self.refresh()

    def refresh(self) -> dict:
        """
        Loads or reloads the map configuration from the YAML file.

        Returns:
            dict: The parsed map configuration.
        """
        self.map_config = common_utils.read_config(self.map_config_file)

        return self.map_config

    def update(self, key: str, kwargs: dict) -> None:
        """
        Updates a specific section of the map configuration with new values.

        Args:
            key (str): The key in the map configuration dictionary to update.
            kwargs (dict): A dictionary of values to merge into the existing configuration.
        """
        if kwargs is not None:
            self.map_config[key].update(kwargs)

    def plot_timestamped(
        self,
        data: gpd.GeoDataFrame = None,
        period: str = "M",
        fmap=None,
        radius: int = 1,
        zoom_start: int = 7,
        date_col: str = "event_date",
        agg_col: str = "event_period",
    ):
        """
        Plot timestamped point events on a Folium map.

        Aggregates point-based event data over a specified temporal period
        (daily ("D"), monthly ("M"), or yearly ("Y")) and visualizes them using
        a time slider via Folium's TimestampedGeoJson.

        Args:
            data (gpd.GeoDataFrame, optional): GeoDataFrame containing point
                geometries and a datetime column.
            period (str, optional): Pandas offset alias defining aggregation
                period (e.g., "Y", "M", "D"). Defaults to "M".
            fmap (folium.Map, optional): Existing Folium map to add layers to.
                If None, a new map is created.
            radius (int, optional): Radius multiplier for point markers.
                Defaults to 1.
            zoom_start (int, optional): Initial zoom level for the map.
                Defaults to 7.
            date_col (str, optional): Name of the datetime column in `data`.
                Defaults to "event_date".
            agg_col (str, optional): Name of the derived aggregation period
                column. Defaults to "event_period".

        Returns:
            folium.Map: Folium map with timestamped event visualization.
        """

        def gdf_to_timestamped_features(gdf, time_col):
            """Convert a GeoDataFrame into TimestampedGeoJson features."""
            features = []
            for index, row in gdf.iterrows():
                row_time = row[time_col] + pd.Timedelta(hours=12)
                features.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(row.geometry),
                        "properties": {
                            "time": row_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "icon": "circle",
                            "iconstyle": {
                                "radius": row["radius"],
                                "fillColor": "red",
                                "fillOpacity": 0.6,
                                "stroke": False,
                            },
                            "tooltip": (
                                f"Number of conflicts: {row['count']}<br>"
                                f"Date: {row[time_col].strftime('%Y-%m-%d')}"
                            ),
                        },
                    }
                )
            return features

        # Normalize dates and derive aggregation period
        data[date_col] = pd.to_datetime(data[date_col].dt.strftime("%Y-%m-%d"))
        data[agg_col] = data[date_col].dt.to_period(period)

        # Aggregate event counts by geometry and period
        agg_counts = (
            data.groupby([agg_col, "geometry"])
            .size()
            .rename("count")
            .reset_index()
        )
        agg_counts[agg_col] = agg_counts[agg_col].dt.to_timestamp()
        agg_counts["radius"] = agg_counts["count"] * radius
        features = {
            "type": "FeatureCollection",
            "features": gdf_to_timestamped_features(agg_counts, agg_col),
        }

        # Compute map center using country centroid
        original_crs = data.crs
        meter_crs = data.estimate_utm_crs()
        centroid = data.dissolve("iso_code").to_crs(meter_crs).centroid
        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS(meter_crs),
            pyproj.CRS(original_crs),
            always_xy=True,
        )
        x, y = transformer.transform(centroid.x.iloc[0], centroid.y.iloc[0])

        # Initialize map if not provided
        if fmap is None:
            fmap = folium.Map(location=[y, x], zoom_start=zoom_start)
            geoboundaries = self.dm.geoboundary.dissolve("iso_code")
            style_function = lambda x: {
                "color": "black",
                "weight": 0.8,
                "fillOpacity": 0.0,
            }
            folium.GeoJson(
                geoboundaries,
                style_function=style_function,
            ).add_to(fmap)

        # Configure date display format
        date_options = "YYYY"
        if period == "M" or period == "D":
            date_options += "-MM"
            if period == "D":
                date_options += "-DD"

        # Add timestamped layer
        TimestampedGeoJson(
            features,
            period=f"P1{period}",
            duration=f"P0{period}",
            add_last_point=False,
            auto_play=False,
            loop=False,
            max_speed=10,
            loop_button=True,
            date_options=date_options,
            time_slider_drag_update=True,
        ).add_to(fmap)

        # Add a static floating title
        title_html = f"""
        <div style="
            position: fixed;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 9999;
            font-size: 24px;
            background-color: white;
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid gray;">
            {self.dm.country} ACLED Conflicts
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(title_html))
        fmap.save(f"{self.dm.iso_code}_folium_map.html")

        return fmap

    def plot_folium(
        self,
        var: str,
        data: gpd.GeoDataFrame = None,
        var_title: str = None,
        adm_level: str = "ADM3",
        precision: int = None,
        kwargs: dict = None,
        key="folium",
    ):
        """
        Create an interactive Folium choropleth map for a given variable.

        Args:
            var (str): Name of the variable/column to visualize.
            data (gpd.GeoDataFrame, optional): GeoDataFrame containing the
                geometries and variable to plot (default: None).
            var_title (str, optional): Display title for the variable in the
                legend and tooltips (default: None).
            adm_level (str): Administrative level to visualize
                (default: 'ADM3').
            precision (int): Number of decimal places to round the variable
                values shown in tooltips (default: None).
            kwargs (dict, optional): Optional configuration overrides for the
                Folium map (default: None).
            key (str): Configuration key used to retrieve map settings
                (default: 'folium').

        Returns:
            folium.Map: Interactive Folium map with a choropleth layer,
            tooltips, and layer controls.
        """

        # Refresh and update configuration
        self.refresh()
        self.update(key, kwargs)
        config = self.map_config[key]

        # Get variable title
        var_title = var_title or self._get_title(var, "legend_titles")
        data = self.dm.data if data is None else data
        data = data.copy()
        original_crs = data.crs

        # Get centroid of the country for map centering
        meter_crs = data.estimate_utm_crs()
        centroid = data.dissolve("iso_code").to_crs(meter_crs).centroid
        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS(meter_crs),
            pyproj.CRS(original_crs),
            always_xy=True,
        )
        x, y = transformer.transform(centroid.x.iloc[0], centroid.y.iloc[0])

        # Initialize folium map
        fmap = folium.Map(
            location=[y, x],
            tiles=config["tiles"],
            zoom_start=config["zoom_start"],
        )
        key_on = f"feature.properties.{adm_level}_ID"

        # Add choropleth layer
        folium.Choropleth(
            data=data,
            geo_data=data.to_json(),
            columns=[f"{adm_level}_ID", var],
            key_on=key_on,
            fill_opacity=config["fill_opacity"],
            fill_color=config["fill_color"],
            line_color=config["line_color"],
            name=var_title,
            legend_name=var_title,
        ).add_to(fmap)

        # Style and highlight functions for tooltips
        style_function = lambda x: config["style_function"]
        highlight_function = lambda x: config["highlight_function"]

        # Add transformed variable column
        var_trans = var + "_transformed"
        data[var_trans] = data[var].apply(
            lambda x: f"{round(x, precision or (4 if x < 1 else 2)):,}"
        )

        # Create layer with tooltips
        adm_levels = [
            f"ADM{level}" for level in range(1, int(adm_level[-1]) + 1)
        ]
        fields = adm_levels + [var_trans]
        aliases = [f"{field}: " for field in fields][:-1] + [f"{var_title}: "]
        layer = folium.features.GeoJson(
            data,
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.features.GeoJsonTooltip(
                fields=fields,
                aliases=aliases,
            ),
            control=False,
        )

        # Add layer to the folium map
        fmap.add_child(layer)
        fmap.keep_in_front(layer)

        # Add layer controls
        folium.LayerControl().add_to(fmap)

        return fmap

    def plot_raster(
        self,
        raster_name: str,
        raster_file: str = None,
        data: gpd.GeoDataFrame = None,
        title: str = None,
        subtitle: str = None,
        legend_title: str = None,
        annotation: str = None,
        save: bool = False,
        kwargs: dict = None,
        out_dir: str = "outputs",
        key="raster",
    ) -> matplotlib.axes.Axes:
        """
        Plot a raster layer with administrative boundaries and annotations.

        Args:
            raster_name (str): Name of the raster to plot.
            data (gpd.GeoDataFrame, optional): GeoDataFrame containing country
                geometries used for outlines and metadata (default: None).
            title (str, optional): Main plot title (default: None).
            subtitle (str, optional): Subtitle displayed below the main title
                (default: None).
            legend_title (str, optional): Title for the colorbar legend
                (default: None).
            annotation (str, optional): Annotation text displayed on the map
                (default: None).
            save (bool): Whether to save the figure to disk (default: False).
            kwargs (dict, optional): Optional configuration overrides for the
                raster plot (default: None).
            out_dir (str): Outputdirectory for saved figures (default: 'outputs').
            key (str): Configuration key used to retrieve map settings
                (default: 'raster').

        Returns:
            matplotlib.axes.Axes: Matplotlib Axes containing the raster plot.
        """

        # Refresh config and apply updates
        self.refresh()
        self.update(key, kwargs)
        config = self.map_config[key]

        # Copy data and country values
        data = self.dm.data if data is None else data
        data = data.copy()
        iso_code = data.iso_code.values[0]
        raster_file = raster_file or os.path.join(
            self.dm.data_dir,
            f"{iso_code}/{iso_code}_{raster_name.upper()}.tif",
        )

        # Instantiate plot
        fig, ax = plt.subplots(
            figsize=(config["figsize_x"], config["figsize_y"]),
            dpi=config["dpi"],
        )

        # Load raster file
        with rio.open(raster_file) as src:
            out_image = src.read(1)
            plot_data = np.array(out_image, dtype=np.float32)

            # Handle special cases
            if "drought" not in raster_name:
                plot_data[plot_data == src.nodata] = np.nan
            if "heat_stress" in raster_name.lower():
                plot_data = plot_data / 100

            img = ax.imshow(
                plot_data,
                extent=[
                    src.bounds.left,
                    src.bounds.right,
                    src.bounds.bottom,
                    src.bounds.top,
                ],
                cmap=getattr(cmaps, config["cmap"]),
                origin="upper",
            )

        # Setup colorbar properties
        bbox_anchor = [
            config["cbar_bbox_x"],
            config["cbar_bbox_y"],
            config["cbar_bbox_width"],
            config["cbar_bbox_height"],
        ]

        # Create inset axes
        axins = inset_axes(
            ax,
            width=config["cbar_width"],
            height=config["cbar_height"],
            loc=config["cbar_loc"],
            bbox_to_anchor=bbox_anchor,
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        # Create colorbar
        cbar = fig.colorbar(
            img,
            cax=axins,
            orientation="vertical",
            pad=config["cbar_pad"],
        )
        cbar.ax.set_yticklabels(
            cbar.ax.get_yticklabels(), fontsize=config["cbar_fontsize"]
        )

        # Get legend title
        legend_title = legend_title or self._get_title(
            raster_name, "legend_titles"
        )

        # Add title to colorbar
        cbar.ax.set_title(
            legend_title,
            fontsize=config["legend_title_fontsize"],
            loc=config["legend_title_loc"],
            x=config["legend_title_x"],
            y=config["legend_title_y"],
        )

        # Get left legend position for alignment
        tight_bbox = cbar.ax.get_tightbbox(fig.canvas.get_renderer())
        tight_bbox_fig = tight_bbox.transformed(fig.transFigure.inverted())
        xpos = tight_bbox_fig.x0

        # Add dissolved country boundary
        dissolved = data.dissolve("iso_code")
        dissolved.geometry = dissolved.geometry.apply(self._fill_holes)
        dissolved.plot(ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none")

        # Add titles and annotations
        title = title or self._get_title(raster_name, "var_titles")
        title = config["title"].format(title, self.dm.country)
        subtitle = subtitle or self._get_subtitle(raster_name)
        annotation = annotation or self._get_annotation(
            [raster_name], add_adm=False
        )
        self._add_titles_and_annotations(
            fig, ax, config, title, subtitle, annotation, x=xpos
        )
        ax.axis("off")

        # Save raster map
        if save:
            sub_dir = os.path.join(
                out_dir, self.dm.iso_code, f"{self.dm.iso_code}_{key}"
            )
            os.makedirs(sub_dir, exist_ok=True)
            filename = f"{self.dm.iso_code}_{raster_name}"
            out_path = os.path.join(sub_dir, filename)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")

        return ax

    def plot_lines(
        self,
        data: gpd.GeoDataFrame = None,
        dataset: str = "osm",
        column: str = "tag",
        osm_tags: list = [],
        simplify: bool = True,
        title: str = None,
        subtitle: str = None,
        annotation: str = None,
        legend_title: str = None,
        ax: matplotlib.axes.Axes = None,
        xpos: float = None,
        zoom_to: dict = None,
        zorder: int = 1,
        kwargs: dict = None,
        key: str = "lines",
    ):
        """
        Plot line features by category on top of administrative boundaries.

        Args:
            data (gpd.GeoDataFrame, optional): GeoDataFrame containing line
                geometries to plot (default: None).
            dataset (str): Dataset name used to determine the data source
                (default: 'osm').
            column (str): Column used to categorize and color line features
                (default: 'tag').
            osm_tags (list): List of OSM tags to include when dataset is OSM
                (default: []).
            simplify (bool): Whether to simplify line geometries for faster
                plotting (default: True).
            title (str, optional): Main plot title (default: None).
            subtitle (str, optional): Subtitle displayed below the main title
                (default: None).
            annotation (str, optional): Annotation text displayed on the map
                (default: None).
            legend_title (str, optional): Title for the legend
                (default: None).
            ax (matplotlib.axes.Axes, optional): Existing Matplotlib Axes to
                plot on (default: None).
            xpos (float, optional): X-position for legend anchoring
                (default: None).
            zoom_to (dict, optional): Dictionary of administrative attributes
                and values used to spatially subset the plot
                (default: None).
            zorder (int): Z-order for plotted line features (default: 1).
            kwargs (dict, optional): Optional configuration overrides for line
                plotting (default: None).
            key (str): Configuration key used to retrieve map settings
                (default: 'lines').

        Returns:
            tuple: Tuple containing:
                - matplotlib.axes.Axes: Axes with plotted line features.
                - float: X-position used for legend placement.
        """
        # Refresh config and apply updates
        self.refresh()
        self.update(key, kwargs)
        config = self.map_config[key]

        # Plot geoboundary if ax and xpos are None
        if ax is None or xpos is None:
            ax, xpos = self.plot_geoboundaries(
                adm_level=self.dm.adm_level,
                zoom_to=zoom_to,
                title=title,
                subtitle=subtitle,
                annotation=annotation,
                legend_title=legend_title,
            )

        # Get x and y positions
        xpos = config.get("legend_x", xpos)
        ypos = config.get("legend_y", 0.3)
        bbox_to_anchor = [xpos, ypos]

        # Get OSM networks if dataset is OSM
        if dataset == "osm":
            data = self.dm.collate_osm_tags(self.dm.osm_networks, osm_tags)
            data[column] = data[column].str.replace("_", " ").str.title()

        # Limit geoboundary if zoom_to values are not None
        geoboundary = self.dm.geoboundary
        if zoom_to is not None:
            geoboundaries = []
            for key, value in zoom_to.items():
                selected = self.dm.data[
                    self.dm.data[key].isin([value])
                ].to_crs(config["crs"])
                geoboundaries.append(selected)

            geoboundary = gpd.GeoDataFrame(
                pd.concat(geoboundaries), geometry="geometry"
            )

        # Update geoboundary CRS
        geoboundary = geoboundary.to_crs(config["crs"])

        # Clip networks and simplify for faster plotting
        networks = gpd.clip(data.to_crs(config["crs"]), geoboundary)
        if simplify:
            networks.geometry = networks.geometry.simplify(
                tolerance=config["tolerance"], preserve_topology=False
            )

        # Get unique categories
        categories = networks[column].unique()
        cmap = getattr(cmaps, config["cmap"])
        colors = {cat: cmap(i) for i, cat in enumerate(categories)}

        # Plot each category
        for cat, color in colors.items():
            subset = networks[networks[column] == cat].to_crs(config["crs"])
            subset.plot(
                ax=ax,
                color=color,
                alpha=config["alpha"],
                lw=config["linewidth"],
                label=cat,
                zorder=zorder,
            )

        # Add legend
        handles = [
            mlines.Line2D(
                [], [], color=color, lw=config["linewidth"], label=cat
            )
            for cat, color in colors.items()
        ]
        legend = Legend(
            ax,
            labels=categories,
            handles=handles,
            loc="center left",
            fontsize=config["legend_label_fontsize"],
            title_fontsize=config["legend_title_fontsize"],
        )
        legend.set_bbox_to_anchor(
            bbox_to_anchor, transform=ax.figure.transFigure
        )
        ax.add_artist(legend)

        return ax, xpos

    def plot_points(
        self,
        column: str = None,
        data: gpd.GeoDataFrame = None,
        dataset: str = "",
        asset: str = "worldpop",
        osm_tags: list = [],
        idmc_year: int = 2024,
        value_col: str = None,
        label_col: str = None,
        title: str = None,
        subtitle: str = None,
        annotation: str = None,
        legend_title: str = None,
        ax: matplotlib.axes.Axes = None,
        xpos: float = None,
        zorder: int = 1,
        zoom_to: dict = None,
        kwargs: dict = None,
        key: str = "points",
    ):
        """
        Plot point features on top of administrative boundaries.

        Args:
            column (str, optional): Column used to categorize or color point
                features (default: None).
            data (gpd.GeoDataFrame, optional): GeoDataFrame containing point
                geometries to plot (default: None).
            dataset (str): Dataset name used to determine the data source
                (default: '').
            asset (str): Asset name used when loading asset-based datasets
                (default: 'worldpop').
            osm_tags (list): List of OSM tags to include when dataset is OSM
                (default: []).
            value_col (str, optional): Column used to scale or weight point
                symbols (default: None).
            label_col (str, optional): Column used for point labels
                (default: None).
            title (str, optional): Main plot title (default: None).
            subtitle (str, optional): Subtitle displayed below the main title
                (default: None).
            annotation (str, optional): Annotation text displayed on the map
                (default: None).
            legend_title (str, optional): Title for the legend
                (default: None).
            ax (matplotlib.axes.Axes, optional): Existing Matplotlib Axes to
                plot on (default: None).
            xpos (float, optional): X-position for legend anchoring
                (default: None).
            zorder (int): Z-order for plotted point features (default: 1).
            zoom_to (dict, optional): Dictionary of administrative attributes
                and values used to spatially subset the plot
                (default: None).
            kwargs (dict, optional): Optional configuration overrides for point
                plotting (default: None).
            key (str): Configuration key used to retrieve map settings
                (default: 'points').

        Returns:
            tuple: Tuple containing:
                - matplotlib.axes.Axes: Axes with plotted point features.
                - float: X-position used for legend placement.
        """

        # Refresh config and apply updates
        self.refresh()
        self.update(key, kwargs)
        config = self.map_config[key]

        # Plot geoboundary if ax and xpos are None
        if ax is None or xpos is None:
            ax, xpos = self.plot_geoboundaries(
                adm_level=self.dm.adm_level,
                zoom_to=zoom_to,
                title=title,
                subtitle=subtitle,
                annotation=annotation,
                legend_title=legend_title,
            )

        # Get x and y positions
        xpos = config.get("legend1_x", xpos - 0.005)
        ypos = config.get("legend1_y", 0.3)
        bbox_to_anchor = [xpos, ypos]

        # Add stacked circle legend titles
        stacked_circle_title = "Number of events"
        if "dtm" in dataset or "idmc" in dataset:
            stacked_circle_title = "Number of IDPs"

        loaders = {
            "acled": lambda: self.dm.acled[asset],
            "ucdp": lambda: self.dm.ucdp,
            "idmc_gidd_disaster": lambda: self.dm.idmc_gidd_disaster,
            "idmc_gidd_conflict": lambda: self.dm.idmc_gidd_conflict,
            "idmc_gidd_combined": lambda: self.dm.idmc_gidd_combined,
            "osm": lambda: self.dm.collate_osm_tags(
                self.dm.osm_pois, osm_tags
            ),
        }
        try:
            data = loaders[dataset]().copy()
            if "idmc" in dataset:
                data = data[idmc_year]
        except KeyError:
            raise ValueError(f"Dataset not supported: {dataset}")

        if len(data) == 0:
            warnings.warn(f"{dataset.upper()} is empty.")
            return None, None

        # Format column values
        column = "tag" if dataset == "osm" else (column or "iso_code")
        if dataset == "osm":
            data[column] = data[column].str.replace("_", " ").str.title()
        elif column == "iso_code":
            data[column] = self.dm.iso_code

        # Limit geoboundary if zoom_to values are not None
        if zoom_to is not None:
            subdata = []
            for key, value in zoom_to.items():
                selected = data[data[key].isin([value])].to_crs(config["crs"])
                if selected.empty:
                    logging.info(
                        f"{WARNING}No points available for {value}.{RESET}"
                    )
                subdata.append(selected)
            data = gpd.GeoDataFrame(pd.concat(subdata), geometry="geometry")

        # If there are no points to plot, return
        if len(data) == 0:
            logging.info(f"{WARNING}{dataset.upper()} is empty.{RESET}")
            return None, None

        # Set CRS
        data = data.to_crs(self.dm.crs).copy()
        data["lon"] = data.geometry.x
        data["lat"] = data.geometry.y

        # Get unique categories
        categories = sorted(data[column].unique())
        cmap = getattr(cmaps, config["cmap"])

        if dataset == "osm":
            # Get unique categories and colors
            colors = {cat: cmap(i) for i, cat in enumerate(categories)}
            for cat, color in colors.items():
                subset = data[data[column] == cat].to_crs(config["crs"])
                subset.plot(
                    ax=ax,
                    color=color,
                    marker=config["marker"],
                    markersize=config["markerscale"],
                    alpha=config["alpha"],
                    lw=config["linewidth"],
                    label=cat,
                    zorder=zorder,
                )

            # Draw legend
            handles = [
                mlines.Line2D(
                    [],
                    [],
                    color=color,
                    linestyle="None",
                    marker=config["marker"],
                    markersize=np.sqrt(config["markerscale"]) * 2,
                    label=cat,
                )
                for cat, color in colors.items()
            ]
            legend = Legend(
                ax,
                labels=categories,
                handles=handles,
                loc="center left",
                fontsize=config["legend_label_fontsize"],
                title_fontsize=config["legend_title_fontsize"],
            )
            legend.set_bbox_to_anchor(
                bbox_to_anchor, transform=ax.figure.transFigure
            )
            ax.add_artist(legend)

        else:
            colors = [
                matplotlib.colors.rgb2hex(color) for color in cmap.colors
            ][: len(categories)]

            points = []
            for category, color in zip(categories, colors):
                subdata = data[data[column] == category].copy()
                records = self._compute_overlap_points(
                    subdata, color, category, value_col
                )
                points.extend(records)

            points = pd.DataFrame(points)
            points = gpd.GeoDataFrame(
                points,
                geometry=gpd.points_from_xy(points["lon"], points["lat"]),
                crs=self.dm.crs,
            )

            max_count = points["count"].max()
            for threshold in [1_000_000, 100_000, 1_000, 100]:
                if max_count >= threshold:
                    config["markerscale"] /= threshold
                    break

            points["count_scaled"] = points["count"] * config["markerscale"]
            points = points.sort_values(by="count", ascending=False)
            points.to_crs(config["crs"]).plot(
                ax=ax,
                facecolor=points["color"],
                legend=False,
                marker="o",
                markersize="count_scaled",
                alpha=config["alpha"],
                lw=0.1,
                zorder=zorder,
            )

            # Draw first legend
            handles = [
                mlines.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=label,
                )
                for label, color in zip(categories, colors)
            ]
            title = legend_title or self._get_title(column, "legend_titles")

            legend1 = ax.legend(
                handles=handles,
                title=title,
                loc="center left",
                markerscale=0.75,
                fontsize=config["legend_label_fontsize"],
                title_fontsize=config["legend_title_fontsize"],
                bbox_to_anchor=bbox_to_anchor,
                bbox_transform=ax.figure.transFigure,
            )
            ax.add_artist(legend1)

            # Create second legend of stacked circles
            ticks = self._make_legend_ticks(points["count"].max())
            legends = [
                mlines.Line2D(
                    [],
                    [],
                    color="silver",
                    lw=0,
                    marker="o",
                    mec="silver",
                    markeredgewidth=1,
                    markersize=np.sqrt(tick * config["markerscale"]),
                    label=tick,
                )
                for tick in ticks
            ]

            xpos = config.get("legend2_x", xpos + 0.035)
            ypos = config.get("legend2_y", ypos)
            bbox_to_anchor = [xpos, ypos]

            sizes = [h.get_markersize() for h in legends]
            labels = [h.get_label() for h in legends]
            dummy = Circle((0, 0), radius=1)

            # Create temporary legend to measure heights
            temp_legend = ax.legend(
                [dummy],
                [""],
                handler_map={
                    Circle: HandlerStackedCircles(
                        sizes, labels, stacked_circle_title
                    )
                },
                loc="center left",
                frameon=False,
                bbox_to_anchor=bbox_to_anchor,
                bbox_transform=ax.figure.transFigure,
            )
            ax.add_artist(temp_legend)
            ax.figure.canvas.draw()

            # Render temporary legend and get extents
            renderer = ax.figure.canvas.get_renderer()
            bb1 = legend1.get_window_extent(renderer).transformed(
                ax.figure.transFigure.inverted()
            )
            bb2 = temp_legend.get_window_extent(renderer).transformed(
                ax.figure.transFigure.inverted()
            )

            # Position second legend right below first
            h1, h2 = bb1.height, bb2.height
            center1 = bb1.y0 + h1 / 2
            new_y = center1 - (h1 / 2 + h2 / 2) - 0.065

            # Remove temporary legend
            temp_legend.remove()

            # Draw second (final) legend
            ypos = config.get("legend2_y", new_y)
            bbox_to_anchor = [xpos, ypos]

            legend2 = ax.legend(
                [dummy],
                [""],
                handler_map={
                    Circle: HandlerStackedCircles(
                        sizes, labels, stacked_circle_title
                    )
                },
                loc="center left",
                frameon=False,
                borderpad=1,
                handletextpad=2,
                labelspacing=config["labelspacing"],
                fontsize=config["legend_label_fontsize"],
                bbox_to_anchor=bbox_to_anchor,
                bbox_transform=ax.figure.transFigure,
            )
            ax.add_artist(legend2)

        # Add labels if specified
        if label_col is not None:
            data.dropna(subset=[label_col]).to_crs(config["crs"]).apply(
                lambda x: ax.annotate(
                    text=x[label_col].replace("(", "\n("),
                    xy=x.geometry.centroid.coords[0],
                    ha="center",
                    fontsize=config["fontsize"],
                    bbox=dict(
                        facecolor=config["label_facecolor"],
                        edgecolor=config["label_edgecolor"],
                        lw=config["label_linewidth"],
                        alpha=config["label_alpha"],
                        boxstyle=config["label_boxstyle"],
                    ),
                ),
                axis=1,
            )

        return ax, xpos

    def plot_geoboundaries(
        self,
        adm_level: str,
        data: gpd.GeoDataFrame = None,
        title: str = None,
        subtitle: str = None,
        legend_title: str = None,
        annotation: str = None,
        group: str = "group",
        max_adms: int = 50,
        max_groups: int = 20,
        zoom_to: dict = None,
        show_adm_names: bool = True,
        kwargs: dict = None,
        save: bool = False,
        out_dir: str = "outputs",
        key="geoboundaries",
    ):
        """
        Plot administrative boundaries with optional grouping, labeling, and zooming.

        Args:
            adm_level (str): Administrative level column to dissolve and plot
                (e.g., "adm1", "adm2").
            data (gpd.GeoDataFrame, optional): GeoDataFrame containing administrative
                boundaries (default: data manager boundaries).
            title (str, optional): Map title (default: country-based title from config).
            subtitle (str, optional): Subtitle text (default: auto-generated).
            legend_title (str, optional): Title for the legend when grouping is used
                (default: config value).
            annotation (str, optional): Footer or annotation text
                (default: auto-generated).
            group (str): Column used to group and color boundaries
                (default: "group").
            max_adms (int): Maximum number of administrative units for which names
                are shown (default: 50).
            max_groups (int): Maximum number of groups allowed for categorical
                coloring and legend creation (default: 20).
            zoom_to (dict, optional): Dictionary of administrative attributes and
                values used to spatially subset the map (default: None).
            show_adm_names (bool): Whether to label administrative units when the
                count is below `max_adms` (default: True).
            kwargs (dict, optional): Optional configuration overrides for plotting
                (default: None).
            save (bool): Whether to save the generated map to disk
                (default: False).
            out_dir (str): Output directory for saved maps (default: "outputs").
            key (str): Configuration key used to retrieve map settings
                (default: "geoboundaries").

        Returns:
            tuple:
                - matplotlib.axes.Axes: Axes containing the plotted map.
                - float: X-position used for legend alignment and inset placement.
        """

        # Refresh config and apply any updates
        self.refresh()
        self.update(key, kwargs)
        config = self.map_config[key]

        data = self.dm.data if data is None else data
        data = data.copy().to_crs(config["crs"])

        # Get dissolved country boundary
        dissolved = data.dissolve("iso_code")
        dissolved_zoomed = None
        if zoom_to is not None:
            data_temp = []
            for key, value in zoom_to.items():
                selected = data[data[key].isin([value])].to_crs(config["crs"])
                data_temp.append(selected)

            data = gpd.GeoDataFrame(pd.concat(data_temp), geometry="geometry")
            dissolved_zoomed = data.dissolve("iso_code")

        # Initialize figure
        fig, ax = plt.subplots(
            figsize=(config["figsize_x"], config["figsize_y"]),
            dpi=config["dpi"],
        )
        data_adm = data.dissolve(adm_level).reset_index()

        # Set default legend title if none provided
        if legend_title is None:
            legend_title = config["legend_title"]

        xpos = 0
        if group in data.columns and data[group].nunique() < max_groups:
            cmap = getattr(cmaps, config["cmap"])
            edgecolor = config["edgecolor_with_group"]
            linewidth = config["linewidth_with_group"]

            # Plot grouped boundaries with color mapping and legend
            data.dissolve(group).reset_index().to_crs(config["crs"]).plot(
                group,
                ax=ax,
                cmap=cmap,
                legend=True,
                categorical=True,
                linewidth=config["group_linewidth"],
                edgecolor=config["group_edgecolor"],
                legend_kwds={
                    "loc": config["group_legend_loc"],
                    "fontsize": config["group_legend_fontsize"],
                    "title_fontsize": config["group_legend_title_fontsize"],
                },
            )

            # Customize legend position and appearance
            legend = ax.get_legend()
            legend.set_bbox_to_anchor(
                [
                    config["group_legend_x"],
                    config["group_legend_y"],
                    config["group_legend_width"],
                    config["group_legend_height"],
                ]
            )
            legend.set_title(legend_title)
            legend._legend_box.align = config["group_legend_box_align"]

            # Determine leftmost position of legend for alignment
            fig.canvas.draw()
            bbox = legend.get_window_extent(fig.canvas.get_renderer())
            bbox_fig = bbox.transformed(fig.transFigure.inverted())
            xpos = bbox_fig.x0
        else:
            # Fallback style
            linewidth = config["linewidth_no_group"]
            edgecolor = config["edgecolor_no_group"]

        # Plot administrative boundaries
        data_adm.to_crs(config["crs"]).plot(
            ax=ax, facecolor="none", edgecolor=edgecolor, lw=linewidth
        )

        # Add labels if number of units is below threshold
        if len(data_adm) < max_adms and show_adm_names is True:
            data_adm.to_crs(config["crs"]).apply(
                lambda x: ax.annotate(
                    text=x[adm_level].replace("(", "\n("),
                    xy=x.geometry.centroid.coords[0],
                    ha="center",
                    fontsize=config["fontsize"],
                    bbox=dict(
                        facecolor=config["label_facecolor"],
                        edgecolor=config["label_edgecolor"],
                        lw=config["label_linewidth"],
                        alpha=config["label_alpha"],
                        boxstyle=config["label_boxstyle"],
                    ),
                ),
                axis=1,
            )

        # Add dissolved country outline
        if dissolved_zoomed is not None:
            dissolved_zoomed.plot(
                ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none"
            )
        else:
            dissolved = data.dissolve("iso_code")
            dissolved.geometry = dissolved.geometry.apply(self._fill_holes)
            dissolved.to_crs(config["crs"]).plot(
                ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none"
            )

        # Zoom to region, if specified
        country = self.dm.country
        if zoom_to is not None:
            subunit = ", ".join([value for value in zoom_to.values()])
            country = f"{subunit}, {country}"
            self._plot_tiny_map(
                zoom_to=zoom_to,
                country=country,
                subunit=subunit,
                data=data,
                dissolved=dissolved,
                fig=fig,
                ax1=ax,
                ax2=None,
                config=config,
                x=xpos,
            )

        # Add titles and annotations
        title = title or config["title"].format(self.dm.country)
        subtitle = subtitle or self._get_subtitle()
        annotation = annotation or self._get_annotation()

        self._add_titles_and_annotations(
            fig, ax, config, title, subtitle, annotation, x=xpos
        )
        ax.axis("off")

        # Save map
        if save:
            sub_dir = os.path.join(
                out_dir, self.dm.iso_code, f"{self.dm.iso_code}_{key}"
            )
            os.makedirs(sub_dir, exist_ok=True)
            filename = f"{self.dm.iso_code}_{group}_{adm_level}"
            out_path = os.path.join(sub_dir, filename)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")

        return ax, xpos

    def plot_choropleth(
        self,
        var: str,
        data: gpd.GeoDataFrame = None,
        var_title: str = None,
        title: str = None,
        subtitle: str = None,
        legend_title: str = None,
        annotation: str = None,
        add_annotation: str = None,
        var_bounds: list = [None, None],
        nbins: int = 4,
        zorder: int = 1,
        binning: str = "equal_intervals",
        show_labels: bool = True,
        zoom_to: dict = None,
        kwargs: dict = None,
        key="choropleth",
    ):
        """
        Plot a choropleth map of a variable on administrative boundaries.

        Args:
            var (str): Column name in `data` to visualize.
            data (gpd.GeoDataFrame, optional): GeoDataFrame containing administrative
                boundaries and variable values (default: data manager boundaries).
            var_title (str, optional): Display title for the variable (default: derived
                from variable name).
            title (str, optional): Map title (default: auto-generated from country name).
            subtitle (str, optional): Subtitle text (default: auto-generated).
            legend_title (str, optional): Legend title (default: derived from variable
                title or config).
            annotation (str, optional): Footer or annotation text (default: auto-generated).
            add_annotation (str, optional): Additional annotation to include on the map.
            var_bounds (list, optional): Minimum and maximum values for the color scale
                (default: [None, None], which uses the data min/max).
            nbins (int, optional): Number of bins for discretizing the variable (default: 4).
            zorder (int, optional): Z-order for the choropleth layer (default: 1).
            binning (str, optional): Method for binning the variable. Supported methods
                include "equal_intervals", "quantiles", etc. (default: "equal_intervals").
            zoom_to (dict, optional): Dictionary of administrative attributes and values
                to spatially subset the map (default: None).
            kwargs (dict, optional): Optional configuration overrides for plotting (default: None).
            key (str, optional): Configuration key used to retrieve map settings (default: "choropleth").

        Returns:
            tuple:
                - matplotlib.axes.Axes: Axes containing the plotted map.
                - float: X-position used for legend alignment and inset placement.
        """

        # Refresh config and apply any updates
        self.refresh()
        self.update(key, kwargs)
        config = self.map_config[key]

        data = self.dm.data if data is None else data
        data = data.copy().to_crs(config["crs"])

        # Create figure and axis
        fig, ax = plt.subplots(
            figsize=(config["figsize_x"], config["figsize_y"]),
            dpi=config["dpi"],
        )
        cmap = getattr(cmaps, config["cmap"])

        # Dissolve geometries for plotting boundaries
        dissolved = data.dissolve("iso_code")
        dissolved.geometry = dissolved.geometry.apply(self._fill_holes)

        # Zoom to region, if specified
        dissolved_zoomed = None
        if zoom_to is not None:
            data = []
            for key, value in zoom_to.items():
                selected = self.dm.data[
                    self.dm.data[key].isin([value])
                ].to_crs(config["crs"])
                data.append(selected)

            data = gpd.GeoDataFrame(pd.concat(data), geometry="geometry")
            dissolved_zoomed = data.dissolve("iso_code")

        # Get min, max bounds
        legend_title = legend_title or self._get_title(var, "legend_titles")
        vmin = var_bounds[0] if var_bounds[0] is not None else data[var].min()
        vmax = var_bounds[1] if var_bounds[1] is not None else data[var].max()
        var_bounds = [vmin, vmax]

        fig.canvas.draw()
        xpos = None

        # Handle case when all values are the same (single color map)
        if data[var].nunique() == 1:
            # Transform value and get color
            unique_value = data[var].dropna().unique()[0]
            if unique_value <= 1:
                color = cmap(unique_value)
            else:
                color = cmap(0.5)

            # Plot single-color map
            data.plot(
                ax=ax,
                color=color,
                edgecolor=config["edgecolor"],
                linewidth=config["linewidth"],
                zorder=zorder,
            )

            # Add legend showing value
            if unique_value > 1:
                label_text = self._humanize(int(unique_value) * 1.0)
            else:
                label_text = self._humanize(unique_value)

            # Create a single-color legend patch
            legend_patch = mpatches.Patch(
                facecolor=color,
                edgecolor=config["edgecolor"],
                label=label_text,
            )

            # Add legend with title on the LEFT
            legend = ax.legend(
                handles=[legend_patch],
                frameon=False,
                fontsize=config["legend_label_fontsize"],
                loc="center left",
                bbox_to_anchor=(-0.1, 0.5),
                title=legend_title if legend_title else var,
                title_fontsize=config["legend_title_fontsize"],
            )
            ax.add_artist(legend)

            # Determine left position of legend for alignment
            fig.canvas.draw()
            tight_bbox = legend.get_window_extent(fig.canvas.get_renderer())
            tight_bbox_fig = tight_bbox.transformed(fig.transFigure.inverted())
            xpos = tight_bbox_fig.x0

            iax = fig.add_axes(tight_bbox_fig)
            iax.set_axis_off()

        elif config["legend_type"] == "bins":
            # Apply binning method for both variables
            if binning == "quantiles":
                var_categories, var_bins = pd.qcut(
                    data[var], nbins, labels=range(nbins), retbins=True
                )
            elif binning == "equal_intervals":
                var_categories, var_bins = self._cut(
                    data[var], var_bounds, nbins
                )

            # Determine colors for bins
            cmap = getattr(cmaps, config["cmap"])
            colors = [cmap(i / (nbins - 1)) for i in range(nbins)]

            # Create human-readable labels for bins
            labels = [
                f"{self._humanize(var_bins[i])} – {self._humanize(var_bins[i+1])}"
                for i in range(nbins)
            ]

            # Plot choropleth using the colors
            color_mapping = {str(i): c for i, c in enumerate(colors)}

            missing_color = "white"
            color_mapping["nan"] = missing_color
            data["bins"] = var_categories
            data["bins"] = data["bins"].astype(str)
            data["color"] = data["bins"].map(color_mapping)
            data["color"] = data["color"].fillna(missing_color)

            data.plot(
                ax=ax,
                color=data["color"],
                edgecolor=config["edgecolor"],
                linewidth=config["linewidth"],
                zorder=zorder,
            )
            # Manually create legend
            patches = [
                mpatches.Patch(
                    facecolor=color, edgecolor=config["edgecolor"], label=label
                )
                for color, label in zip(reversed(colors), reversed(labels))
            ]
            legend = ax.legend(
                handles=patches,
                loc="center left",
                bbox_to_anchor=(-0.3, 0.5),
                title=legend_title,
                fontsize=config["legend_label_fontsize"],
                title_fontsize=config["legend_title_fontsize"],
            )
            ax.add_artist(legend)

            fig.canvas.draw()
            tight_bbox = legend.get_window_extent(fig.canvas.get_renderer())
            tight_bbox_fig = tight_bbox.transformed(fig.transFigure.inverted())
            xpos = tight_bbox_fig.x0  # left edge for alignment

            # 3. Create a dummy iax somewhere else (will not hide legend)
            iax = fig.add_axes(tight_bbox_fig)
            iax.set_axis_off()

        elif config["legend_type"] == "colorbar":
            legend_kwds = {
                "shrink": config["legend_shrink"],
                "location": "left",
            }
            data[var] = data[var].astype(float)
            data.plot(
                var,
                ax=ax,
                legend=True,
                cmap=cmap,
                edgecolor=config["edgecolor"],
                lw=config["linewidth"],
                legend_kwds=legend_kwds,
                vmin=vmin,
                vmax=vmax,
                zorder=zorder,
            )

            # Get colorbar axis and set titles, labels
            iax = fig.axes[1]

            # Reposition colorbar depending on zoom
            pos = iax.get_position()
            cbar_width = pos.width
            cbar_height = pos.height

            if "legend_x" in config:
                cbar_x = config["legend_x"]
            if "legend_y" in config:
                cbar_y = config["legend_y"]
            elif zoom_to is not None:
                cbar_y = (
                    ax.get_position().y0
                    + 0.5 * (ax.get_position().height - cbar_height) / 5
                )
            else:
                cbar_y = (
                    ax.get_position().y0
                    + 2 * (ax.get_position().height - cbar_height) / 5
                )

            cbar_x = pos.x0
            iax.set_position([cbar_x, cbar_y, cbar_width, cbar_height])
            iax.tick_params(labelsize=config["legend_label_fontsize"])
            iax.set_title(
                legend_title, fontsize=config["legend_title_fontsize"]
            )
            iax.yaxis.set_major_formatter(
                mticker.FuncFormatter(self._humanize)
            )

            # Determine left position of legend for alignment
            tight_bbox = iax.get_tightbbox(fig.canvas.get_renderer())
            tight_bbox_fig = tight_bbox.transformed(fig.transFigure.inverted())
            xpos = tight_bbox_fig.x0

        elif config["legend_type"] == "barplot":
            # Position inset axis for barplot relative to map axis
            ax_pos = ax.get_position()
            barplot_width = config["barplot_width"]
            barplot_height = config["barplot_height"]

            # Different x and y position depending on zoom mode
            if zoom_to is not None:
                barplot_y = (
                    ax_pos.y0 + 2 * (ax_pos.height - barplot_height) / 5
                )
            else:
                barplot_y = (
                    ax_pos.y0 + 4 * (ax_pos.height - barplot_height) / 5
                )

            barplot_x = (
                ax_pos.x0 - 2 * barplot_width + config["barplot_x_offset"]
            )
            barplot_y += config["barplot_y_offset"]

            # Create inset axis for histogram barplot
            iax = ax.inset_axes(
                bounds=[barplot_x, barplot_y, barplot_width, barplot_height]
            )
            iax.set_xticks([])
            iax.spines[["top", "right", "bottom"]].set_visible(False)

            # Bin variable values into categories for histogram
            nbins = min(data[var].nunique(), config["barplot_nbins"])
            categories, bins = self._cut(data[var], [vmin, vmax], nbins)
            data["categories"] = categories.astype("Int64").fillna(-1)

            # Map bins to colors using cmap
            bin_width = bins[1] - bins[0]
            y_ticks = bins[:-1] + bin_width / 2

            # Map bins to colors using cmap
            colors = [
                cmap((val - min(bins)) / (max(bins) - min(bins)))
                for val in bins
            ]
            color_mapping = {
                category: color
                for category, color in zip(range(nbins), colors)
            }
            color_mapping[-1] = config["missing_color"]

            # Plot choropleth with bin-based colors
            data["colors"] = data["categories"].map(color_mapping)
            data.plot(
                ax=ax,
                color=data["colors"],
                edgecolor=config["edgecolor"],
                lw=config["linewidth"],
                vmin=vmin,
                vmax=vmax,
                zorder=zorder,
            )

            # Draw histogram bars in inset axis
            n = iax.hist(
                data[var], bins=bins, orientation="horizontal", alpha=0
            )[0]
            iax.barh(y_ticks, n, height=bin_width, color=colors)

            # Format y-axis ticks with bin ranges
            iax.set_yticks(
                y_ticks,
                labels=[
                    f"{self._humanize(edge)} to {self._humanize(edge+bin_width)}"
                    for edge in bins[:-1]
                ],
                size=config["barplot_tick_size"],
            )

            # Align title with leftmost tick label
            iax.figure.canvas.draw()
            tick_label_boxes = [
                label.get_window_extent()
                for label in iax.get_yticklabels()
                if label.get_text()
            ]

            if tick_label_boxes:
                # Leftmost edge of all tick labels (min x value)
                leftmost = min(box.x0 for box in tick_label_boxes)

                # Convert display coords to axis coords
                inv = iax.transAxes.inverted()
                leftmost_axes = inv.transform((leftmost, 0))[0]

                # Place title aligned to leftmost tick label
                iax.text(
                    leftmost_axes,
                    config["legend_title_gap"],
                    legend_title,
                    transform=iax.transAxes,
                    fontsize=config["legend_title_fontsize"],
                    va="bottom",
                    ha="left",
                )

            # Add bar labels showing counts + percentages
            for index, (x, y) in enumerate(zip(y_ticks, n)):
                percent = y / sum(n)
                label = (
                    r"$\bf{"
                    + str(self._humanize(y))
                    + "}$"
                    + f" ({percent * 100:.0f}%)"
                )
                y_range = max(n) - min(n)
                y += 0.035 * y_range
                iax.text(
                    y,
                    x,
                    s=label,
                    color=config["barplot_label_color"],
                    size=config["barplot_label_size"],
                    va="center",
                )
            iax.tick_params(axis="y", length=2)

            # Determine left position of legend for alignment
            tight_bbox = iax.get_tightbbox(fig.canvas.get_renderer())
            tight_bbox_fig = tight_bbox.transformed(fig.transFigure.inverted())
            xpos = tight_bbox_fig.x0

        # Plot boundaries (zoomed or full)
        if dissolved_zoomed is not None:
            dissolved_zoomed.plot(
                ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none"
            )
        else:
            dissolved.plot(
                ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none"
            )

        # Plot missing data and legend
        if data[var].isnull().any():
            data_missing = data[data[var].isna()]
            ax = self._plot_missing(ax, data_missing, config)

        # Get variable legend title texts
        country = self.dm.country
        var_title = var_title or self._get_title(var, "var_titles")
        title = title or config["title"].format(var_title, country)
        subtitle = subtitle or self._get_subtitle(var)
        annotation = annotation or self._get_annotation([var])
        if add_annotation is not None:
            annotation = annotation + add_annotation

        # Plot tiny map
        if zoom_to is not None:
            subunit = ", ".join([value for value in zoom_to.values()])
            country = f"{subunit}, {country}"
            self._plot_tiny_map(
                zoom_to=zoom_to,
                country=country,
                subunit=subunit,
                data=data,
                dissolved=dissolved,
                fig=fig,
                ax1=ax,
                ax2=iax,
                config=config,
                x=xpos,
            )

        if show_labels:
            data.apply(
                lambda x: ax.annotate(
                    text=x[self.dm.adm_level].replace("(", "\n("),
                    xy=x.geometry.centroid.coords[0],
                    ha="center",
                    fontsize=config["fontsize"],
                    bbox=dict(
                        facecolor=config["label_facecolor"],
                        edgecolor=config["label_edgecolor"],
                        lw=config["label_linewidth"],
                        alpha=config["label_alpha"],
                        boxstyle=config["label_boxstyle"],
                    ),
                ),
                axis=1,
            )

        # Add title, subtitle, and annotations
        self._add_titles_and_annotations(
            fig, ax, config, title, subtitle, annotation, x=xpos
        )
        ax.axis("off")

        return ax, xpos

    def plot_bivariate_choropleth(
        self,
        var1: str,
        var2: str,
        data: gpd.GeoDataFrame = None,
        var1_bounds: list = None,
        var2_bounds: list = None,
        var1_title: str = None,
        var2_title: str = None,
        legend1_title: str = None,
        legend2_title: str = None,
        legend_title: str = None,
        title: str = None,
        subtitle: str = None,
        annotation: str = None,
        add_annotation: str = None,
        binning: str = "quantiles",
        show_labels: bool = False,
        nbins: int = 4,
        zoom_to: dict = None,
        zorder: int = 1,
        kwargs: dict = None,
        key="bivariate_choropleth",
    ):
        """
        Plot a bivariate choropleth map using two variables on administrative boundaries.

        Args:
            var1 (str): Name of the first variable to visualize.
            var2 (str): Name of the second variable to visualize.
            data (gpd.GeoDataFrame, optional): GeoDataFrame containing administrative
                boundaries and both variables (default: data manager boundaries).
            var1_bounds (list, optional): Minimum and maximum bounds for `var1`
                when using equal-interval binning (default: None).
            var2_bounds (list, optional): Minimum and maximum bounds for `var2`
                when using equal-interval binning (default: None).
            var1_title (str, optional): Display title for the first variable
                (default: derived from variable name).
            var2_title (str, optional): Display title for the second variable
                (default: derived from variable name).
            legend1_title (str, optional): Axis title for the first variable in the
                bivariate legend (default: derived from variable name).
            legend2_title (str, optional): Axis title for the second variable in the
                bivariate legend (default: derived from variable name).
            legend_title (str, optional): Overall legend title (currently unused;
                reserved for future extensions).
            title (str, optional): Main map title (default: auto-generated).
            subtitle (str, optional): Subtitle text (default: auto-generated from
                both variables).
            annotation (str, optional): Footer or annotation text (default:
                auto-generated).
            add_annotation (str, optional): Additional annotation appended to the
                default annotation.
            binning (str, optional): Binning method for both variables. Supported
                values include "quantiles" and "equal_intervals" (default: "quantiles").
            nbins (int, optional): Number of bins per variable used to construct the
                bivariate color matrix (default: 4).
            zoom_to (dict, optional): Dictionary of administrative attributes and
                values used to spatially subset the map (default: None).
            zorder (int, optional): Z-order for the choropleth layer (default: 1).
            kwargs (dict, optional): Optional configuration overrides for plotting
                (default: None).
            key (str, optional): Configuration key used to retrieve map settings
                (default: "bivariate_choropleth").

        Returns:
            tuple:
                - matplotlib.axes.Axes: Axes object containing the bivariate choropleth map.
                - float: X-position used for legend alignment and layout adjustments.
        """

        # Refresh config and apply any updates
        self.refresh()
        self.update(key, kwargs)
        config = self.map_config[key]

        # Copy and reproject data
        data = self.dm.data if data is None else data
        data = data.copy().to_crs(config["crs"])

        # Create figure
        fig, ax = plt.subplots(
            figsize=(config["figsize_x"], config["figsize_y"]),
            dpi=config["dpi"],
        )

        # Dissolve national geometry and fill geometry holes
        dissolved = data.dissolve("iso_code")
        dissolved.geometry = dissolved.geometry.apply(self._fill_holes)

        # Apply zoom if requested
        dissolved_zoomed = None
        if zoom_to is not None:
            data = []
            for key, value in zoom_to.items():
                selected = self.dm.data[
                    self.dm.data[key].isin([value])
                ].to_crs(config["crs"])
                data.append(selected)

            data = gpd.GeoDataFrame(pd.concat(data), geometry="geometry")
            dissolved_zoomed = data.dissolve("iso_code")

        # Apply binning method for both variables
        if binning == "quantiles":
            var1_categories, var1_bins = pd.qcut(
                data[var1], nbins, labels=range(nbins), retbins=True
            )
            var2_categories, var2_bins = pd.qcut(
                data[var2], nbins, labels=range(nbins), retbins=True
            )
        elif binning == "equal_intervals":
            var1_categories, var1_bins = self._cut(
                data[var1], var1_bounds, nbins
            )
            var2_categories, var2_bins = self._cut(
                data[var2], var2_bounds, nbins
            )

        var1_edges = list(var1_bins)
        var2_edges = list(var2_bins)

        # Assign bivariate categories and colormap
        data_plot = data.copy()
        data_plot["bivariate"] = var1_categories.astype(
            "str"
        ) + var2_categories.astype("str")
        cmap = config[f"cmap{nbins}"]

        # Build color lookup dictionary
        index = 0
        cmap_dict = dict()
        for i in range(nbins):
            for j in range(nbins):
                cmap_dict[f"{i}{j}"] = cmap[index]
                index += 1

        # Assign colors
        data_plot["cmap"] = data_plot["bivariate"].map(cmap_dict)
        data_missing = data_plot[data_plot["cmap"].isna()]
        data_plot["cmap"] = data_plot["cmap"].fillna(config["missing_color"])
        color = data_plot["cmap"]

        # Plot main choropleth
        data.to_crs(config["crs"]).plot(
            ax=ax,
            color=color,
            edgecolor=config["edgecolor"],
            lw=config["linewidth"],
            zorder=zorder,
        )
        if len(data_missing) > 0:
            ax = self._plot_missing(ax, data_missing, config)

        # Plot dissolved outline (national boundary)
        if dissolved_zoomed is not None:
            dissolved_zoomed.plot(
                ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none"
            )
        else:
            dissolved.plot(
                ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none"
            )

        ax.axis("off")

        # Legend subplot settings
        ncols, nrows = nbins, nbins
        alpha = 1

        # Get main ax position in figure coords
        fig.canvas.draw()
        ax_pos = ax.get_position()

        # Width and height of your legend (tweak as needed or keep from config)
        legend_width = 0.1
        legend_height = 0.1

        # Align legend vertically centered with ax and move outside left
        legend_x = ax_pos.x0 - legend_width - 0.05

        if zoom_to is not None:
            legend_y = ax_pos.y0 + 2 * (ax_pos.height - legend_height) / 5
        else:
            legend_y = ax_pos.y0 + 4 * (ax_pos.height - legend_height) / 5

        # Create legend as inset axis
        ax2 = fig.add_axes([legend_x, legend_y, legend_width, legend_height])
        ax2.set_aspect("equal", adjustable="box")

        # Draw legend grid
        col_width = 1 / ncols
        row_height = 1 / nrows
        color_index = 0

        for col in range(ncols):
            for row in range(nrows):
                xmin = col * col_width
                xmax = (col + 1) * col_width
                ymin = row * row_height
                ymax = (row + 1) * row_height

                ax2.axvspan(
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                    alpha=alpha,
                    color=cmap[color_index],
                )
                color_index += 1

        ax2.margins(x=0)
        ax2.spines[["right", "top"]].set_visible(False)

        # Legend tick labels
        var1_labels = [self._humanize(x) for x in var1_edges]
        var2_labels = [self._humanize(x) for x in var2_edges]
        tickpos = np.linspace(0, 1, nbins + 1)
        ax2.set_xticks(
            tickpos, var1_labels, fontsize=config["legend_fontsize"]
        )
        ax2.set_yticks(
            tickpos, var2_labels, fontsize=config["legend_fontsize"]
        )

        # Legend axis titles
        if legend1_title is None:
            legend1_title = self._get_title(var1, "legend_titles")
        if legend2_title is None:
            legend2_title = self._get_title(var2, "legend_titles")

        ax2.set_xlabel(legend1_title, fontsize=6, ha="left")
        ax2.yaxis.set_label_coords(-0.35, 0)

        ax2.set_ylabel(legend2_title, fontsize=6, ha="left")
        ax2.xaxis.set_label_coords(0, -0.25)

        # Determine left position of legend for alignment
        tight_bbox = ax2.get_tightbbox(fig.canvas.get_renderer())
        tight_bbox_fig = tight_bbox.transformed(fig.transFigure.inverted())
        xpos = tight_bbox_fig.x0

        # Build titles and annotations
        if var1_title is None:
            var1_title = self._get_title(var1, "var_titles")
        if var2_title is None:
            var2_title = self._get_title(var2, "var_titles")
        subtitle = subtitle or self._get_subtitle(
            var1
        ) + "\n" + self._get_subtitle(var2)
        annotation = annotation or self._get_annotation([var1, var2])

        if add_annotation is not None:
            annotation = annotation + add_annotation

        country = self.dm.country

        # If zoomed, adjust titles and add tiny map
        if zoom_to is not None:
            subunit = ", ".join([value for value in zoom_to.values()])
            country = f"{subunit}, {country}"
            self._plot_tiny_map(
                zoom_to=zoom_to,
                country=country,
                subunit=subunit,
                data=data,
                dissolved=dissolved,
                fig=fig,
                ax1=ax,
                ax2=ax2,
                config=config,
                x=xpos,
            )

        def remove_duplicates(s1: str, s2: str) -> str:
            words1, words2 = s1.split(), s2.split()
            i = 0
            while i < min(len(words1), len(words2)) and words1[i] == words2[i]:
                i += 1
            if i == 0:  # no shared prefix
                return s1, s2
            return s1, " ".join(words2[i:])

        var1_title, var2_title = remove_duplicates(var1_title, var2_title)

        # Get title text
        if title is None:
            title = config["title"].format(var1_title, var2_title, country)

        if show_labels:
            data.apply(
                lambda x: ax.annotate(
                    text=x[self.dm.adm_level].replace("(", "\n("),
                    xy=x.geometry.centroid.coords[0],
                    ha="center",
                    fontsize=config["fontsize"],
                    bbox=dict(
                        facecolor=config["label_facecolor"],
                        edgecolor=config["label_edgecolor"],
                        lw=config["label_linewidth"],
                        alpha=config["label_alpha"],
                        boxstyle=config["label_boxstyle"],
                    ),
                ),
                axis=1,
            )

        # Add titles and annotations with layout adjusted to legend
        self._add_titles_and_annotations(
            fig, ax, config, title, subtitle, annotation, x=xpos
        )
        ax.axis("off")

        return ax, xpos

    def _plot_missing(
        self,
        ax: matplotlib.axes.Axes,
        data_missing: gpd.GeoDataFrame,
        config: dict,
    ) -> matplotlib.axes.Axes:
        """
        Plot areas with missing data using a hatched overlay and legend.

        This helper function overlays polygons corresponding to missing values
        on an existing map axis, applies a configurable hatch pattern, and
        adds a small legend entry labeled "No data".

        Args:
            ax (matplotlib.axes.Axes): Axes object to plot on.
            data_missing (gpd.GeoDataFrame): GeoDataFrame containing geometries
                with missing data values.
            config (dict): Plot configuration dictionary containing styling
                parameters such as colors, hatching, line widths, and CRS.

        Returns:
            matplotlib.axes.Axes: Axes with missing data overlay and legend added.
        """

        # Set hatch linewidth (applies to all hatching in the plot)
        mpl.rcParams["hatch.linewidth"] = config["missing_hatch_linewidth"]

        # Plot missing data regions with hatching
        data_missing.to_crs(config["crs"]).plot(
            ax=ax,
            facecolor=config["missing_color"],
            hatch=config["missing_hatch"],
            edgecolor=config["missing_edgecolor"],
            lw=config["missing_linewidth"],
            legend=True,
        )

        # Create a custom legend patch for "No data"
        mpatch = [
            mpatches.Patch(
                facecolor=config["missing_color"],
                hatch=config["missing_hatch"],
                edgecolor=config["missing_edgecolor"],
                linewidth=config["missing_linewidth"] * 1.5,
                label="No data",
            )
        ]

        # Get axis position in figure coordinates
        pos = ax.get_position()
        axis_height = pos.height

        # Dynamic padding = 1% of axis height
        padding = 0.01 * axis_height

        # Place legend just below the lower-right corner of the axis
        legend = ax.legend(
            handles=mpatch,
            loc="upper right",
            bbox_to_anchor=(pos.x1, pos.y0 - padding),
            frameon=False,
            fontsize=8,
            bbox_transform=ax.figure.transFigure,
        )

        ax.add_artist(legend)

        return ax

    def _plot_tiny_map(
        self,
        zoom_to: str,
        country: str,
        subunit: str,
        data: gpd.GeoDataFrame,
        dissolved: gpd.GeoDataFrame,
        fig,
        ax1: matplotlib.axes.Axes,
        ax2: matplotlib.axes.Axes,
        config: dict,
        x: float,
    ) -> None:
        """
        Add a small inset map highlighting a zoomed administrative subregion.

        Args:
            zoom_to (str): Administrative attribute used for zooming (e.g., ADM name).
            country (str): Country name used for contextual labeling.
            subunit (str): Name of the selected subregion to highlight.
            data (gpd.GeoDataFrame): GeoDataFrame containing the zoomed geometries.
            dissolved (gpd.GeoDataFrame): Dissolved country geometry used as
                background for the inset map.
            fig (matplotlib.figure.Figure): Matplotlib figure to add the inset to.
            ax1 (matplotlib.axes.Axes): Main map axes.
            ax2 (matplotlib.axes.Axes): Legend axes, if present; used for layout
                alignment (can be None).
            config (dict): Plot configuration dictionary with styling parameters.
            x (float): Left x-coordinate (figure space) used to align the inset map.

        Returns:
            None
        """

        # Get main axes and legend axes positions
        ax1_pos = ax1.get_position()
        iax_height = max_height = (ax1_pos.y1 - ax1_pos.y0) / 3
        iax_width = ax1_pos.x0 - x
        iax_y = (ax1_pos.y1 - ax1_pos.y0) * 3 / 4

        if ax2 is not None:
            ax2_pos = ax2.get_position()

            natural_height = ax1_pos.y1 - ax2_pos.y1
            total_gap = ax1_pos.y1 - ax2_pos.y1
            iax_height = min(natural_height, max_height)
            iax_y = ax2_pos.y1 + (total_gap - iax_height) / 2

        # Create tiny map axes in figure coordinates
        iax = fig.add_axes([x, iax_y, iax_width, iax_height])
        iax.set_axis_off()

        # Plot background (dissolved country) and highlight the selected region
        dissolved.plot(
            ax=iax, facecolor="lightgray", edgecolor="lightgray", lw=1
        )
        data.dissolve("iso_code").plot(
            ax=iax, facecolor="bisque", edgecolor="sienna", lw=0.25
        )

        # Get bounding box for annotation placement
        xmin, ymin, xmax, ymax = data.dissolve("iso_code").total_bounds

        # Add subunit label centered above the selected area
        iax.annotate(
            text=subunit,
            xy=(xmin + abs(xmin - xmax) / 2, ymax),
            xytext=(0, 5),  # offset upwards by 5 points
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=config["fontsize"],
            bbox=dict(
                facecolor=config["label_facecolor"],
                edgecolor=config["label_edgecolor"],
                lw=config["label_linewidth"],
                alpha=config["label_alpha"],
                boxstyle=config["label_boxstyle"],
            ),
        )

    def _add_titles_and_annotations(
        self,
        fig,
        ax: matplotlib.axes.Axes,
        config: dict,
        title: str,
        subtitle: str,
        annotation: str,
        x: float = None,
    ) -> None:
        """
        Add title, subtitle, and annotation text to a map figure.

        This helper positions figure-level text elements relative to the
        map axes, dynamically adjusting vertical spacing to avoid overlaps.
        Text placement can be overridden via configuration values.

        Args:
            fig (matplotlib.figure.Figure): Figure object to add text to.
            ax (matplotlib.axes.Axes): Axes used to determine layout bounds.
            config (dict): Plot configuration dictionary containing font sizes,
                spacing, colors, and optional explicit text coordinates.
            title (str): Main title text (can be None to skip).
            subtitle (str): Subtitle text displayed below the title
                (can be None to skip).
            annotation (str): Annotation text displayed below the plot
                (can be None to skip).
            x (float, optional): Left x-coordinate (figure space) used to align
                text elements horizontally (default: None).

        Returns:
            None
        """

        # Get axis vertical bounds
        y0 = ax.get_position().y0
        y1 = ax.get_position().y1

        if title is not None:
            title_x = config.get("title_x", x)
            title_y = config.get("title_y", y1)

            title = fig.text(
                x=title_x,
                y=title_y,
                s=title,
                size=config["title_fontsize"],
                font=BOLD_FONT,
            )

        if subtitle is not None:
            text_height = self._get_text_height(
                fig, subtitle, config["subtitle_fontsize"]
            )
            subtitle_x = config.get("subtitle_x", x)
            subtitle_y = y1 - text_height - config["subtitle_gap"]
            subtitle_y = config.get("subtitle_y", subtitle_y)

            # Adjust title position to leave space for subtitle
            title_y += (
                self._get_text_height(
                    fig,
                    subtitle,
                    fontsize=config["subtitle_fontsize"],
                )
                + config["subtitle_gap"]
            )

            fig.text(
                x=subtitle_x,
                y=subtitle_y,
                s=subtitle,
                size=config["subtitle_fontsize"],
                font=REGULAR_FONT,
            )

        # Add annotation (if provided)
        if annotation is not None:
            annotation_x, annotation_y = x, y0
            annotation_x = config.get("annotation_x", title_x)

            text_height = self._get_text_height(
                fig, annotation, config["annotation_fontsize"]
            )
            annotation_y = y0 - text_height - config["annotation_gap"]
            annotation_y = config.get("annotation_y", annotation_y)

            fig.text(
                x=annotation_x,
                y=annotation_y,
                s=annotation,
                size=config["annotation_fontsize"],
                color=config["annotation_color"],
                font=REGULAR_FONT,
            )

    def _get_title(
        self,
        var: str,
        config_key: str,
        mhs_name: str = "Multihazards",
        conflict_name: str = "Conflicts",
    ) -> str:
        """
        Generate a human-readable title or legend label for a variable.

        Args:
            var (str): Variable name used to determine the display title.
            config_key (str): Configuration key pointing to title or legend
                templates in the map configuration.
            mhs_name (str): Display name for multihazard variables
                (default: 'Multihazards').
            conflict_name (str): Display name for conflict-related variables
                (default: 'Conflicts').

        Returns:
            str: Formatted title or legend label suitable for map display.
        """

        legend_titles = self.map_config[config_key]
        legend = "legend" in config_key
        mhs_name = mhs_name[:-1] if legend else mhs_name
        conflict_name = conflict_name[:-1] if legend else conflict_name

        asset_name = self._find_matching_alias(
            var, self.map_config["assets_alias"]
        )
        title = self._find_matching_alias(
            var, self.map_config["hazards_alias"]
        )

        for key, template in legend_titles.items():
            if key == var:
                title = template
                break

            elif key in var:
                if var.startswith("mhs"):
                    category = var.split("_")[1]
                    fill = (
                        mhs_name
                        if category == "all"
                        else f"{category.title()} {mhs_name}"
                    )
                    if any(tag in var for tag in ("acled", "ucdp")):
                        fill = f"{fill} & {conflict_name}"

                    if legend:
                        title = template.format(fill, asset_name)
                    else:
                        title = template.format(asset_name, fill)

                    title = self._capitalize(title)
                    break

                else:
                    title = title or var
                    if "exposure" in var:
                        title = title.replace(f"_{var}", "").replace(var, "")
                    if "acled" in var or "ucdp" in var:
                        title = template.format(asset_name, conflict_name)
                    else:
                        title = title.replace(f"_{key}", "").replace("_", " ")
                        title = template.format(asset_name, title)

                    title = self._capitalize(title)
                    break

        if title is None:
            exclude_columns = ["cause", "category", "type"]
            if "_" in var or "type" in var:
                title = self._capitalize(var.replace("_", " ").title())
            elif not any(col in var for col in exclude_columns):
                title = self._capitalize(f"{var} Risk")
            else:
                title = self._capitalize(var)

        no_relative = "relative" not in var
        no_conflict = "conflict" not in var
        no_fatalities = "fatalities" not in var

        if legend and no_relative and no_conflict and no_fatalities:
            if "bem" in var:
                title += "\n(Total US Dollar)"
            elif "worldcover" in var:
                title += " (km$^2$)"

        return title

    def _get_subtitle(self, var: str = None):
        """
        Generate a contextual subtitle for a map based on the variable name.

        This helper derives dataset-specific subtitle text, such as reporting
        years, averaging periods, or conflict date ranges, by inspecting the
        variable name and associated data sources.

        Args:
            var (str, optional): Variable name used to infer the subtitle
                content (default: None).

        Returns:
            str: Subtitle text describing temporal coverage or reporting
            context for the variable. Returns an empty string if no subtitle
            applies.
        """

        subtitle = ""
        if var is None:
            return subtitle

        if "dtm" in var:
            year = self.dm.dtm.yearReportingDate.unique()[0]
            round_number = self.dm.dtm.roundNumber.unique()[0]
            subtitle = f"Internal displacement figures as of {year} (Round {round_number})"

        elif "idmc" in var:
            if "mean" in var:
                years = [
                    str(year) for year in self.dm.idmc_gidd_combined.keys()
                ]
                subtitle = f"Internal displacement figures averaged over {', '.join(years)}"
            else:
                subtitle = (
                    f"Internal displacement figures as of {var.split('_')[-1]}"
                )

        elif "acled" in var or "ucdp" in var:
            if "acled" in var:
                asset = self._find_matching_alias(
                    var, self.map_config["assets_alias"], return_var=True
                )
                conflict_start_date = (
                    self.dm.acled[asset]["event_date"]
                    .min()
                    .strftime("%Y-%m-%d")
                )
                conflict_end_date = (
                    self.dm.acled[asset]["event_date"]
                    .max()
                    .strftime("%Y-%m-%d")
                )

            elif "ucdp" in var:
                conflict_start_date = (
                    self.dm.ucdp["date_start"].min().strftime("%Y-%m-%d")
                )
                conflict_end_date = (
                    self.dm.ucdp["date_start"].max().strftime("%Y-%m-%d")
                )

            start_date = datetime.strptime(conflict_start_date, "%Y-%m-%d")
            end_date = datetime.strptime(conflict_end_date, "%Y-%m-%d")
            subtitle = (
                f"Conflict events from {start_date.year} to {end_date.year}"
            )

        return subtitle

    def _get_annotation(self, var_list: list = [], add_adm: bool = True):
        """
        Construct a source annotation string based on variables used in the map.

        This helper matches variable names against configured annotation keys
        and concatenates unique source descriptions into a single annotation
        block. Optionally, the administrative boundary source is included.

        Args:
            var_list (list, optional): List of variable names used in the map.
                Defaults to an empty list.
            add_adm (bool, optional): Whether to include the administrative
                boundary data source in the annotation (default: True).

        Returns:
            str: A formatted multi-line annotation string listing data sources.
        """

        # Avoid mutable default arguments by initializing inside
        if var_list is None:
            var_list = []

        annotations = self.map_config["annotations"]
        annotation = "Source: \n"

        # Optionally add administrative source to variable list
        if add_adm:
            var_list += [self.dm.adm_source.lower()]

        # Track unique annotations to avoid duplicates
        anns = []
        for var in var_list:
            for key, ann in annotations.items():
                if key in var:
                    if ann not in anns:
                        anns.append(ann)
                        annotation += ann + "\n"

        return annotation

    def _cut(self, series: pd.Series, var_bounds: list, nbins: int) -> tuple:
        """
        Bin a pandas Series into discrete intervals.

        If explicit bounds are provided, they are used directly when their
        length matches ``nbins + 1``; otherwise, evenly spaced bin edges are
        generated between the minimum and maximum bounds. If no bounds are
        provided, pandas determines the bins automatically.

        Args:
            series (pd.Series): Values to be binned.
            var_bounds (list | None): Optional list of bin edges or value bounds.
                If length equals ``nbins + 1``, it is used directly as bin edges.
            nbins (int): Number of bins to create.

        Returns:
            tuple:
                - pd.Series: Categorical bin labels (0 to ``nbins - 1``).
                - np.ndarray: Array of bin edge values.
        """

        if var_bounds is not None:
            if len(var_bounds) == nbins + 1:
                var_bins = var_bounds
            else:
                var_bins = np.linspace(
                    var_bounds[0], var_bounds[-1], nbins + 1
                )

            return pd.cut(
                series,
                bins=var_bins,
                labels=range(nbins),
                retbins=True,
                include_lowest=True,
            )
        else:
            return pd.cut(
                series,
                nbins,
                labels=range(nbins),
                retbins=True,
                include_lowest=True,
            )

    def _get_text_height(
        self, fig: plt.Figure, text: str, fontsize: float
    ) -> float:
        """
        Compute the relative height of a text string within a figure.

        The height is measured in figure coordinate units by rendering the
        text off-canvas and comparing its bounding box height to the total
        figure height.

        Args:
            fig (plt.Figure): Matplotlib figure used for rendering.
            text (str): Text whose rendered height will be measured.
            fontsize (float): Font size used for rendering the text.

        Returns:
            float: Text height expressed as a fraction of the figure height.
        """

        renderer = fig.canvas.get_renderer()
        text = plt.text(0, 0, text, fontsize=fontsize)
        bbox = text.get_window_extent(renderer=renderer)
        text.remove()

        # Return text height relative to figure height
        return bbox.height / fig.bbox.height

    def _compute_overlap_points(
        self,
        subdata: gpd.GeoDataFrame,
        color: str,
        category: str,
        value_col: str = None,
    ):
        """
        Aggregate overlapping point features by identical coordinates.

        Args:
            subdata (gpd.GeoDataFrame): Data containing point coordinates with
                `lat` and `lon` columns.
            color (str): Color assigned to the aggregated points.
            category (str): Category label assigned to the aggregated points.
            value_col (str, optional): Column to sum when aggregating points.
                If None or not present, points are counted instead
                (default: None).

        Returns:
            list[dict]: List of dictionaries representing aggregated point
            features, including coordinates, counts, color, and category.
        """
        if value_col in subdata.columns:
            grouped = (
                subdata.groupby(["lat", "lon"], as_index=False)[value_col]
                .sum()
                .rename(columns={value_col: "count"})
            )
        else:
            grouped = (
                subdata.groupby(["lat", "lon"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
            )

        grouped["color"] = color
        grouped["category"] = category

        return grouped.to_dict("records")

    def _make_legend_ticks(self, max_count: int) -> list:
        """
        Generate human-friendly legend tick values based on a maximum count.

        Args:
            max_count (int): Maximum value represented in the legend.

        Returns:
            list[int]: List of tick values suitable for use in map or chart legends.
        """

        min_val = 5 if max_count <= 20 else 10
        max_val = self._round_for_display(max_count)

        nice_values = [
            1,
            50,
            100,
            500,
            1_000,
            5_000,
            10_000,
            50_000,
            100_000,
            500_000,
        ]

        # Base candidates
        ticks = [min_val] + [v for v in nice_values if v < max_val] + [max_val]

        # Thinning rules by scale
        thinning_rules = [
            (1_000_000, lambda t: t == 1_000 or t >= 100_000),
            (100_000, lambda t: t == 100 or t >= 10_000),
            (10_000, lambda t: t in (1, 100) or t >= 1_000),
        ]

        for threshold, keep in thinning_rules:
            if max_val > threshold:
                ticks = [t for t in ticks if keep(t) or t == max_val]
                break

        # Ensure at least 3 ticks
        if len(ticks) < 3:
            mid = self._round_for_display((min_val + max_val) // 2)
            ticks.insert(1, mid)

        return ticks

    def _round_for_display(self, x) -> int:
        """
        Round a numeric value up to a readable threshold for display.

        Args:
            x (int | float): Numeric value to round.

        Returns:
            int: Rounded value using predefined thresholds.
        """

        for limit, base in (
            (10, 1),
            (50, 10),
            (100, 50),
            (500, 100),
            (1000, 500),
        ):
            if x <= limit:
                return math.ceil(x / base) * base

        return math.ceil(x / 1000) * 1000

    def _fill_holes(self, geometry) -> object:
        """
        Removes interior holes from Polygon or MultiPolygon geometries.

        Args:
            geometry (Polygon | MultiPolygon | object):
                The input geometry to process.
                - If Polygon, returns a new Polygon with only the exterior ring.
                - If MultiPolygon, returns a MultiPolygon with holes removed from each Polygon.
                - Other geometry types are returned unchanged.

        Returns:
            object: Geometry with holes removed if Polygon/MultiPolygon,
                    otherwise returns the input geometry unchanged.
        """

        if isinstance(geometry, Polygon):
            return Polygon(geometry.exterior)

        elif isinstance(geometry, MultiPolygon):
            return MultiPolygon([Polygon(p.exterior) for p in geometry.geoms])

        return geometry

    def _humanize(self, value: float, number: int = None) -> str:
        """
        Convert a numeric value into a compact, human-readable string.

        Args:
            value (float | int): Numeric value to format.
            number (optional): Placeholder for future use.

        Returns:
            str: Human-readable string representation.
        """
        if value <= 0:
            return "0"

        # Large numbers
        if value >= 10:
            formatter = (
                "%.1f"
                if value < 100_000
                else "%.0f" if value < 1_000_000 else "%.1f"
            )

            text = humanize.intword(value, formatter)
            return (
                text.replace(" thousand", "k")
                .replace(" million", "M")
                .replace(" billion", "B")
                .replace(".0k", "k")
                .replace(".0M", "M")
                .replace(".0B", "B")
            )

        # Small decimals
        if value < 1:
            if value < 0.01:
                return f"{value:.4f}"
            if value < 0.1:
                return f"{value:.3f}"
            return f"{value:.2f}"

        # Between 1 and 10
        return str(int(value)) if float(value).is_integer() else f"{value:.1f}"

    def _find_matching_alias(
        self, var: str, alias_map: dict, return_var: bool = False
    ) -> str:
        """
        Find and return the first matching alias for a variable name.

        Args:
            var (str): Variable name to match against alias keys.
            alias_map (dict): Mapping of alias keys to human-readable labels.
            return_var (bool, optional): If True, return the matching alias
                key instead of its mapped value. Defaults to False.

        Returns:
            str | None: The matched alias value (or key if `return_var=True`);
            returns None if no match is found.
        """
        for alias_key, alias_value in alias_map.items():
            if alias_key in var:
                if return_var:
                    return alias_key
                return alias_value
        return

    def _capitalize(self, string: str):
        """
        Capitalize a string while preserving common stop words and newlines.

        Args:
            string (str): Input string to be formatted.

        Returns:
            str: Capitalized string with stop words preserved.
        """

        word_list = re.split(" ", string)
        final = [word_list[0].capitalize()]
        stop_words = get_stop_words("en") + ["IDPs"]

        for word in word_list[1:]:
            newline = False
            if word.startswith("\n"):
                word = word[1:]
                newline = True
            if word not in stop_words:
                word = word.capitalize()
                word = "\n" + word if newline else word
            final.append(word)

        final = " ".join(final)

        return final


class HandlerStackedCircles(HandlerPatch):
    """
    Custom legend handler that renders a stacked circles, with size indicating
    magnitude.

    Attributes:
        sizes (list[float]): Circle diameters used to represent values.
        labels (list[str | int]): Labels corresponding to each circle.
        title (str): Title displayed above the stacked circles.
        color (str): Edge color for all circles.
    """

    def __init__(
        self,
        sizes,
        labels,
        title,
        color="silver",
        **kwargs,
    ):
        """
        Initialize the stacked circle legend handler.

        Args:
            sizes (list[float]): Circle diameters for legend symbols.
            labels (list[str | int]): Text labels for each circle.
            title (str): Title displayed above the legend symbols.
            color (str, optional): Circle edge color. Defaults to "silver".
            **kwargs: Additional keyword arguments passed to HandlerPatch.
        """
        super().__init__(**kwargs)

        self.sizes = sizes
        self.labels = labels
        self.title = title
        self.color = color

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        """
        Create legend artists for stacked circles.

        This method is called internally by Matplotlib when rendering the legend.

        Args:
            legend: Matplotlib legend instance.
            orig_handle: Original handle passed to the legend.
            xdescent (float): Horizontal descent of the legend box.
            ydescent (float): Vertical descent of the legend box.
            width (float): Available width for the legend entry.
            height (float): Available height for the legend entry.
            fontsize (float): Font size used in the legend.
            trans: Transform applied to legend artists.

        Returns:
            list: List of Matplotlib artists composing the legend entry.
        """

        artists = []

        # Largest radius determines the overall stack height
        max_radius = max(self.sizes) / 2

        # Horizontal center for circles
        center_x = width / 2 - xdescent

        # Bottom alignment of the stacked circles
        bottom_y = height / 2 - ydescent - max_radius

        # x-position for numeric labels, offset to the right of circles
        label_x = center_x + max_radius + 5

        # Draw circles and labels from largest to smallest
        for size, label in sorted(zip(self.sizes, self.labels), reverse=True):
            radius = size / 2

            # Circle representing magnitude
            circle = Circle(
                (center_x, bottom_y + radius),
                radius=radius,
                facecolor="none",
                edgecolor=self.color,
                lw=1,
            )

            circle.set_transform(trans)
            artists.append(circle)

            # Numeric label aligned with the circle
            text = plt.Text(
                x=label_x,
                y=bottom_y + 1.85 * radius,
                text=str(int(label)),
                va="center_baseline",
                ha="left",
                fontsize=fontsize,
            )

            text.set_transform(trans)
            artists.append(text)

        # Title positioned above the largest circle
        title_y = bottom_y + 2 * max_radius + fontsize
        title = plt.Text(
            x=center_x,
            y=title_y,
            text=self.title,
            va="bottom",
            ha="center",
            fontsize=fontsize,
            fontweight="bold",
        )

        title.set_transform(trans)
        artists.append(title)

        return artists
