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
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerPatch
from matplotlib.legend import Legend

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

import seaborn as sns
import pypalettes
import pyfonts

from rasterio.plot import show
import rasterio.mask

from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

from dfcv_colocation_mapping import data_utils
from vincenty import vincenty

WARNING = "\033[31m"
RESET = "\033[0m"


class GeoPlot:
    def __init__(
        self, dm, data_dir: str = "data/", map_config_file: str = None
    ):
        """
        Initializes a GeoPlot instance.

        Args:
            dm: Data manager object that contains the dataset (`dm.data`).
            data_dir (str, optional): Path to the data directory. Defaults to "./data/".
            map_config_file (str, optional): Path to a YAML map configuration file.
                If None, the default config in the package is used.

        Raises:
            FileNotFoundError: If the map configuration file does not exist.
        """

        self.dm = dm
        self.data_dir = data_dir

        resources = importlib_resources.files("dfcv_colocation_mapping")
        self.map_config_file = map_config_file or resources.joinpath(
            "configs", "map_config.yaml"
        )

        self.regular_font = pyfonts.load_google_font("Roboto")
        self.bold_font = pyfonts.load_google_font("Roboto", weight="bold")

        # Refresh configuration from the YAML file
        self.refresh()

    def refresh(self) -> dict:
        """
        Loads or reloads the map configuration from the YAML file.

        Returns:
            dict: The parsed map configuration.

        Raises:
            FileNotFoundError: If the map configuration file does not exist.
            yaml.YAMLError: If the YAML file contains invalid syntax.
        """

        # Read the configuration using the utility function
        self.map_config = data_utils.read_config(self.map_config_file)

        # Return the loaded configuration
        return self.map_config

    def update(self, key: str, kwargs: dict) -> None:
        """
        Updates a specific section of the map configuration with new values.

        Args:
            key (str): The key in the map configuration dictionary to update.
            kwargs (dict): A dictionary of values to merge into the existing configuration.
        """

        # Update the configuration for the specified key
        self.map_config[key].update(kwargs)

    def plot_folium(
        self,
        var: str,
        data: gpd.GeoDataFrame = None,
        var_title: str = None,
        adm_level: str = "ADM3",
        precision: int = 4,
        kwargs: dict = None,
        key="folium",
    ):
        """Create an interactive Folium choropleth map for a given variable.

        Args:
            var (str): Column name in the data to visualize.
            var_title (str, optional): Display title for the variable. Defaults to None.
            adm_level (str, optional): Administrative level ID for mapping. Defaults to "ADM3".
            precision (int, optional): Number of decimal places for tooltip values. Defaults to 4.
            kwargs (dict, optional): Configuration overrides. Defaults to None.
            key (str, optional): Map configuration key. Defaults to "folium".

        Returns:
            folium.Map: Folium Map object with the choropleth and tooltips added.
        """
        # Refresh configuration and apply any overrides
        self.refresh()
        if kwargs is not None:
            self.update(key, kwargs)
        config = self.map_config[key]

        # Default variable title
        if var_title is None:
            var_title = self._get_title(var, "var_titles", legend=True)

        if data is None:
            data = self.dm.data.copy()
        original_crs = data.crs

        # Ensure data is not empty
        if data.empty:
            raise ValueError("Data is empty. Cannot create folium map.")

        # Ensure the variable exists
        if var not in data.columns:
            raise ValueError(f"Variable '{var}' not found in data columns.")

        # Get centroid of the country for map centering
        centroid = (
            data.dissolve("iso_code").to_crs(config["meter_crs"]).centroid
        )
        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS(config["meter_crs"]),
            pyproj.CRS(original_crs),
            always_xy=True,
        )
        x, y = transformer.transform(centroid.x.iloc[0], centroid.y.iloc[0])

        # Initialize folium map
        m = folium.Map(
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
        ).add_to(m)

        # Style and highlight functions for tooltips
        style_function = lambda x: config["style_function"]
        highlight_function = lambda x: config["highlight_function"]

        # Add transformed variable column for tooltips
        var_trans = var + "_transformed"
        data[var_trans] = data[var].apply(lambda x: round(x, precision))

        # Add GeoJson layer with tooltips
        nil = folium.features.GeoJson(
            data,
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.features.GeoJsonTooltip(
                fields=[adm_level, var_trans],
                aliases=[f"{adm_level}: ", f"{var_title}: "],
            ),
            control=False,
        )
        m.add_child(nil)
        m.keep_in_front(nil)

        # Add layer control
        folium.LayerControl().add_to(m)

        return m

    def plot_raster(
        self,
        raster_name: str,
        data: gpd.GeoDataFrame = None,
        title: str = None,
        subtitle: str = None,
        legend_title: str = None,
        annotation: str = None,
        save: bool = False,
        kwargs: dict = None,
        base_folder: str = "outputs",
        key="raster",
    ) -> matplotlib.axes.Axes:
        """Plot a raster layer for a country with optional titles and colorbar.

        This function reads a raster file (GeoTIFF) for the specified country,
        applies a colormap, and plots it with a colorbar. Titles, subtitles,
        and annotations can be added using configuration settings.

        Args:
            raster_name (str): Name of the raster to plot.
            title (str, optional): Main title for the plot. Defaults to formatted country/raster name.
            subtitle (str, optional): Subtitle text.
            legend_title (str, optional): Title for the colorbar. Defaults to raster variable title.
            annotation (str, optional): Annotation text for the figure.
            kwargs (dict, optional): Configuration overrides for plotting.
            key (str, optional): Configuration key from `map_config`. Defaults to `"raster"`.

        Returns:
            matplotlib.axes.Axes: Axes object containing the plotted raster.

        Raises:
            FileNotFoundError: If the raster file does not exist.
            ValueError: If `self.dm.data` is empty or `iso_code` cannot be found.
        """

        # Refresh config and apply any updates
        self.refresh()
        if kwargs is not None:
            self.update(key, kwargs)
        config = self.map_config[key]

        if data is None:
            data = self.dm.data.copy()
        if data.empty:
            raise ValueError("Data is empty. Cannot plot raster.")
        if "iso_code" not in data.columns:
            raise ValueError("'iso_code' column not found in data.")

        iso_code = data.iso_code.values[0]
        raster_file = os.path.join(
            self.data_dir, f"{iso_code}/{iso_code}_{raster_name.upper()}.tif"
        )
        if not os.path.exists(raster_file):
            raise FileNotFoundError(f"Raster file not found: {raster_file}")

        # Create figure and axis
        fig, ax = plt.subplots(
            figsize=(config["figsize_x"], config["figsize_y"]),
            dpi=config["dpi"],
        )

        # Open raster and mask no-data values
        with rio.open(raster_file) as src:
            out_image = src.read(1)
            plot_data = np.array(np.copy(out_image), dtype=np.float32)
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

        # Setup colorbar axes and properties
        bbox_anchor = [
            config["cbar_bbox_x"],
            config["cbar_bbox_y"],
            config["cbar_bbox_width"],
            config["cbar_bbox_height"],
        ]

        axins = inset_axes(
            ax,
            width=config["cbar_width"],
            height=config["cbar_height"],
            loc=config["cbar_loc"],
            bbox_to_anchor=bbox_anchor,
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        cbar = fig.colorbar(
            img,
            cax=axins,
            orientation="vertical",
            # format=mticker.FuncFormatter(data_utils._humanize),
            pad=config["cbar_pad"],
        )
        cbar.ax.set_yticklabels(
            cbar.ax.get_yticklabels(), fontsize=config["cbar_fontsize"]
        )

        if legend_title is None:
            legend_title = self._get_title(
                raster_name, "legend_titles", legend=True
            )

        # Add title to colorbar
        cbar.ax.set_title(
            legend_title,
            fontsize=config["legend_title_fontsize"],
            loc=config["legend_title_loc"],
            x=config["legend_title_x"],
            y=config["legend_title_y"],
        )

        # Determine left position of legend for alignment
        tight_bbox = cbar.ax.get_tightbbox(fig.canvas.get_renderer())
        tight_bbox_fig = tight_bbox.transformed(fig.transFigure.inverted())
        xpos = tight_bbox_fig.x0

        # Add dissolved country outline
        dissolved = data.dissolve("iso_code")
        dissolved.geometry = dissolved.geometry.apply(data_utils._fill_holes)
        dissolved.plot(ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none")

        # Add titles and annotations
        title = title or self._get_title(raster_name, "var_titles")
        title = config["title"].format(title, self.dm.country)
        subtitle = subtitle or self._get_subtitle(raster_name)
        annotation = annotation or self._get_annotation(
            [raster_name], add_adm=False
        )

        # Add titles and annotations with layout adjusted to legend
        self._add_titles_and_annotations(
            fig, ax, config, title, subtitle, annotation, x=xpos
        )
        ax.axis("off")

        if save:
            sub_folder = os.path.join(
                base_folder, self.dm.iso_code, f"{self.dm.iso_code}_{key}"
            )
            os.makedirs(sub_folder, exist_ok=True)
            filename = f"{self.dm.iso_code}_{raster_name}"
            out_path = os.path.join(sub_folder, filename)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")

        return ax

    def _collate_osm_tags(self, osm_data, tags):
        """
        Combine OSM tag layers into one GeoDataFrame, ordering features
        so that sparser layers are plotted on top when calling .plot().
        """
        # Collect GeoDataFrames and record their sizes
        gdfs_with_counts = []
        for tag in tags:
            gdf = osm_data[tag].copy()
            gdfs_with_counts.append((len(gdf), gdf))

        # Sort by feature count
        gdfs_with_counts.sort(key=lambda x: x[0], reverse=True)

        # Concatenate — denser ones first, sparser last
        ordered_gdfs = [gdf for _, gdf in gdfs_with_counts]
        combined = gpd.GeoDataFrame(pd.concat(ordered_gdfs, ignore_index=True))

        return combined

    def plot_lines(
        self,
        data: gpd.GeoDataFrame = None,
        dataset: str = "osm",
        osm_tags: list = [],
        ax: matplotlib.axes.Axes = None,
        xpos: float = None,
        zoom_to: dict = None,
        zorder: int = 1,
        kwargs: dict = None,
        key: str = "lines",
    ):
        self.refresh()
        if kwargs is not None:
            self.update(key, kwargs)
        config = self.map_config[key]

        if ax is None or xpos is None:
            ax, xpos = self.plot_geoboundaries(
                adm_level=self.dm.adm_level, zoom_to=zoom_to
            )

        xpos = config.get("legend_x", xpos)
        ypos = config.get("legend_y", 0.3)
        bbox_to_anchor = [xpos, ypos]

        if dataset == "osm":
            column = "tag"
            data = self._collate_osm_tags(self.dm.osm_networks, osm_tags)
            data[column] = data[column].str.replace("_", " ").str.title()

        if len(data) == 0:
            warnings.warn(f"{dataset.upper()} is empty.")
            return

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

        geoboundary = geoboundary.to_crs(config["crs"])

        networks = gpd.clip(data.to_crs(config["crs"]), geoboundary)
        networks.geometry = networks.geometry.simplify(
            tolerance=config["tolerance"], preserve_topology=False
        )

        # Unique categories
        categories = networks[column].unique()
        cmap = getattr(cmaps, config["cmap"])
        colors = {cat: cmap(i) for i, cat in enumerate(categories)}

        # Plot each category manually (so colors match handles)
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
        value_col: str = None,
        label_col: str = None,
        title: str = None,
        subtitle: str = None,
        legend_title: str = None,
        ax: matplotlib.axes.Axes = None,
        xpos: float = None,
        zorder: int = 1,
        zoom_to: dict = None,
        kwargs: dict = None,
        key: str = "points",
    ):

        self.refresh()
        if kwargs is not None:
            self.update(key, kwargs)
        config = self.map_config[key]

        if ax is None or xpos is None:
            ax, xpos = self.plot_geoboundaries(
                adm_level=self.dm.adm_level,
                zoom_to=zoom_to,
                title=title,
                subtitle=subtitle,
                legend_title=legend_title,
            )

        xpos = config.get("legend1_x", xpos - 0.005)
        ypos = config.get("legend1_y", 0.3)
        bbox_to_anchor = [xpos, ypos]

        stacked_circle_title = "Number of events"
        if "dtm" in dataset or "idmc" in dataset:
            stacked_circle_title = "Number of IDPs"

        if dataset == "acled":
            data = self.dm.acled[asset]
        elif dataset == "ucdp":
            data = self.dm.ucdp
        elif dataset == "idmc_gidd_disaster":
            data = self.dm.idmc_gidd_disaster
        elif dataset == "idmc_gidd_conflict":
            data = self.dm.idmc_gidd_disaster
        elif dataset == "idmc_gidd_combined":
            data = self.dm.idmc_gidd_combined
        elif dataset == "osm":
            column = "tag"
            data = self._collate_osm_tags(self.dm.osm_pois, osm_tags)
            data[column] = data[column].str.replace("_", " ").str.title()

        if len(data) == 0:
            warnings.warn(f"{dataset.upper()} is empty.")
            return

        data["iso_code"] = self.dm.iso_code
        if column is None:
            column = "iso_code"
        elif column not in data.columns:
            warnings.warn(f"{column} is not in the {dataset.upper()} dataset.")
            return

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

        if len(data) == 0:
            logging.info(f"{WARNING}{dataset.upper()} is empty.{RESET}")
            return

        data = data.to_crs(self.dm.crs).copy()
        data["lon"] = data.geometry.x
        data["lat"] = data.geometry.y

        categories = sorted(data[column].unique())
        cmap = getattr(cmaps, config["cmap"])
        colors = [matplotlib.colors.rgb2hex(c) for c in cmap.colors][
            : len(categories)
        ]

        all_points = []
        handles = []

        def make_legend_ticks(max_count: int):
            min_val = 5 if max_count <= 20 else 10
            max_val = data_utils._nice_round(max_count)
            nice_values = [
                1,
                50,
                100,
                500,
                1000,
                5000,
                10000,
                20000,
                50000,
                100000,
                500000,
            ]
            multiples = [v for v in nice_values if v < max_val]
            ticks = [min_val] + multiples + [max_val]

            if max_val > 1000000:
                ticks = [
                    t
                    for t in ticks
                    if t == 1000 or t >= 100000 or t == max_val
                ]
            elif max_val > 100000:
                ticks = [
                    t for t in ticks if t == 100 or t >= 10000 or t == max_val
                ]
            elif max_val > 10000:
                ticks = [
                    t
                    for t in ticks
                    if t == 1 or t == 100 or t >= 1000 or t == max_val
                ]

            if len(ticks) < 3:
                mid = (min_val + max_val) // 2
                ticks.insert(1, data_utils._nice_round(mid))
            return ticks

        def compute_overlap_points(subdata, color, category, value_col=None):
            """Group by identical lat/lon (no clustering).
            If value_col is provided, sum that column instead of counting points.
            """
            if value_col and value_col in subdata.columns:
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

        if "osm" in dataset:
            categories = data[column].unique()
            colors = {cat: cmap(i) for i, cat in enumerate(categories)}

            # Plot each category manually (so colors match handles)
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

            handles = [
                mlines.Line2D(
                    [],
                    [],
                    color=color,
                    linestyle="None",
                    marker=config["marker"],
                    markersize=config["markerscale"],
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
            for leg in legend.legend_handles:
                leg.set_markersize(np.sqrt(config["markerscale"]) * 2)
                leg.set_marker(config["marker"])
            legend.set_bbox_to_anchor(
                bbox_to_anchor, transform=ax.figure.transFigure
            )
            ax.add_artist(legend)

        else:
            for category, color in zip(categories, colors):
                subdata = data[data[column] == category].copy()
                records = compute_overlap_points(
                    subdata, color, category, value_col
                )
                all_points.extend(records)

            all_points = pd.DataFrame(all_points)
            all_points = gpd.GeoDataFrame(
                all_points,
                geometry=gpd.points_from_xy(
                    all_points["lon"], all_points["lat"]
                ),
                crs="EPSG:4326",
            )

            max_count = all_points["count"].max()
            for threshold in [1_000_000, 100_000, 1_000, 100, 10]:
                if max_count >= threshold:
                    config["markerscale"] /= threshold
                    break

            all_points["count_scaled"] = (
                all_points["count"] * config["markerscale"]
            )
            all_points = all_points.sort_values(by="count", ascending=False)

            all_points.to_crs(config["crs"]).plot(
                ax=ax,
                facecolor=all_points["color"],
                legend=False,
                marker="o",
                markersize="count_scaled",
                alpha=config["alpha"],
                lw=0.1,
                zorder=zorder,
            )

            handles = [
                Line2D(
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
            title = self._get_title(column, "legend_titles")

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

            ticks = make_legend_ticks(all_points["count"].max())
            legends = [
                mlines.Line2D(
                    [],
                    [],
                    color="silver",
                    lw=0,
                    marker="o",
                    mec="silver",
                    markeredgewidth=1,
                    markersize=np.sqrt(n * config["markerscale"]),
                    label=n,
                )
                for n in ticks
            ]

            class HandlerStackedCircles(HandlerPatch):
                def __init__(
                    self,
                    sizes,
                    labels,
                    title,
                    color="silver",
                    **kwargs,
                ):
                    super().__init__(**kwargs)
                    self.sizes, self.labels, self.title, self.color = (
                        sizes,
                        labels,
                        title,
                        color,
                    )

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
                    artists = []
                    max_r = max(self.sizes) / 2
                    center_x = width / 2 - xdescent
                    bottom_y = height / 2 - ydescent - max_r
                    label_x = center_x + max_r + 5
                    for s, lbl in sorted(
                        zip(self.sizes, self.labels), reverse=True
                    ):
                        r = s / 2
                        c = Circle(
                            (center_x, bottom_y + r),
                            radius=r,
                            facecolor="none",
                            edgecolor=self.color,
                            lw=1,
                        )
                        c.set_transform(trans)
                        artists.append(c)
                        t = plt.Text(
                            x=label_x,
                            y=bottom_y + 1.85 * r,
                            # text=str(data_utils._humanize(int(lbl))),
                            text=str(int(lbl)),
                            va="center_baseline",
                            ha="left",
                            fontsize=fontsize,
                        )
                        t.set_transform(trans)
                        artists.append(t)
                    title_y = bottom_y + 2 * max_r + fontsize
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

            def add_count_legend(ax, legends, xpos, ypos):
                sizes = [h.get_markersize() for h in legends]
                labels = [h.get_label() for h in legends]
                dummy = Circle((0, 0), radius=1)

                # draw final version
                xpos = config.get("legend2_x", xpos + 0.035)
                ypos = config.get("legend2_y", ypos)
                bbox_to_anchor = [xpos, ypos]

                # create temporary legend2 to measure heights
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

                renderer = ax.figure.canvas.get_renderer()
                bb1 = legend1.get_window_extent(renderer).transformed(
                    ax.figure.transFigure.inverted()
                )
                bb2 = temp_legend.get_window_extent(renderer).transformed(
                    ax.figure.transFigure.inverted()
                )
                h1 = bb1.height
                h2 = bb2.height
                center1 = bb1.y0 + h1 / 2

                # position second legend right below first
                new_y = center1 - (h1 / 2 + h2 / 2) - 0.065
                temp_legend.remove()

                # draw final version
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

            add_count_legend(ax, legends, xpos, ypos)

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

    def plot_hatches(
        self,
        adm_level: str,
        column: str,
        data: gpd.GeoDataFrame = None,
        ax: matplotlib.axes.Axes = None,
        xpos: float = None,
        zoom_to: dict = None,
        zorder: int = 1,
        kwargs: dict = None,
        title: str = "",
        key="hatches",
    ):
        self.refresh()
        if kwargs is not None:
            self.update(key, kwargs)
        config = self.map_config[key]

        if ax is None or xpos is None:
            ax, xpos = self.plot_geoboundaries(
                adm_level=self.dm.adm_level, zoom_to=zoom_to
            )

        if data is None:
            data = self.dm.data.copy()

        xpos = config.get("legend_x", xpos - 0.005)
        ypos = config.get("legend_y", 0.3)
        bbox_to_anchor = [xpos, ypos]

        if zoom_to is not None:
            data_temp = []
            for key, value in zoom_to.items():
                selected = data[data[key].isin([value])].to_crs(config["crs"])
                data_temp.append(selected)
            data = gpd.GeoDataFrame(pd.concat(data_temp), geometry="geometry")

        data = data.sort_values(column, ascending=False)

        patches, labels = [], []
        hatches = config["hatches"]
        for item, hatch in enumerate(hatches):
            data.iloc[[item]].to_crs(config["crs"]).plot(
                ax=ax,
                column=adm_level,
                facecolor="none",
                edgecolor="black",
                lw=config["linewidth"],
                hatch=hatch,
                legend=False,
                zorder=zorder,
            )
            label = f"{item+1}. {data.iloc[[item]][adm_level].values[0]}"
            patch = mpatches.Patch(
                facecolor="none", alpha=1, hatch=hatch, label=label
            )
            patches.append(patch)
            labels.append(label)

        legend = Legend(
            ax,
            labels=labels,
            handles=patches,
            loc="center left",
            fontsize=config["legend_label_fontsize"],
            title_fontsize=config["legend_title_fontsize"],
        )
        legend.set_title(title)
        legend.set_bbox_to_anchor(
            bbox_to_anchor, transform=ax.figure.transFigure
        )
        ax.add_artist(legend)

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
        base_folder: str = "outputs",
        key="geoboundaries",
    ) -> matplotlib.axes.Axes:
        """
        Plot administrative boundaries (geo-boundaries) with optional grouping and labeling.

        This function generates a map of administrative boundaries at the specified level,
        optionally grouping units by a categorical variable. Boundaries are styled according
        to configuration settings, and small units can be labeled directly on the map.

        Args:
            adm_level (str): Column name representing the administrative level to dissolve and plot.
            title (str, optional): Main title for the plot. Defaults to formatted country name.
            subtitle (str, optional): Subtitle for the plot.
            legend_title (str, optional): Title for the legend. Defaults to config value.
            annotation (str, optional): Extra annotation text to display on the figure.
            group (str, optional): Column name used to group and color administrative units.
                Defaults to `'group'`.
            max_units (int, optional): Maximum number of administrative units to annotate directly.
                Defaults to 50.
            kwargs (dict, optional): Configuration overrides for plotting.
            key (str, optional): Configuration key from `map_config`. Defaults to `"geoboundaries"`.

        Returns:
            matplotlib.axes.Axes: Axes object with plotted boundaries.

        Raises:
            ValueError: If `self.data` is empty or `adm_level` is not in data columns.
        """
        # Refresh config and apply any updates
        self.refresh()
        if kwargs is not None:
            self.update(key, kwargs)
        config = self.map_config[key]

        if data is None:
            data = self.dm.data.copy()

        if data.empty:
            raise ValueError("Data is empty. Cannot plot geoboundaries.")
        if adm_level not in data.columns:
            raise ValueError(f"Column '{adm_level}' not found in data.")

        data = data.to_crs(config["crs"])
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
            # No grouping: fallback style
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
            dissolved.geometry = dissolved.geometry.apply(
                data_utils._fill_holes
            )
            dissolved.to_crs(config["crs"]).plot(
                ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none"
            )

        country = self.dm.country
        if zoom_to is not None:
            subunit = ", ".join([value for value in zoom_to.values()])
            country = f"{subunit}, {country}"
            self._plot_tiny_map(
                zoom_to,
                country,
                subunit,
                data,
                dissolved,
                fig,
                ax,
                None,
                config,
                x=xpos,
            )

        title = title or config["title"].format(self.dm.country)
        subtitle = subtitle or self._get_subtitle()
        annotation = annotation or self._get_annotation()

        # Add titles and annotations with layout adjusted to legend
        self._add_titles_and_annotations(
            fig, ax, config, title, subtitle, annotation, x=xpos
        )
        ax.axis("off")

        if save:
            sub_folder = os.path.join(
                base_folder, self.dm.iso_code, f"{self.dm.iso_code}_{key}"
            )
            os.makedirs(sub_folder, exist_ok=True)
            filename = f"{self.dm.iso_code}_{group}_{adm_level}"
            out_path = os.path.join(sub_folder, filename)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")

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
        nbins: int = 4,
        zoom_to: dict = None,
        zorder: int = 1,
        kwargs: dict = None,
        key="bivariate_choropleth",
    ) -> matplotlib.axes.Axes:
        """
        Plot a bivariate choropleth map combining two variables.

        Args:
            var1 (str): First variable to map.
            var2 (str): Second variable to map.
            var1_bounds (list, optional): Custom bin edges for var1 if using equal_intervals.
            var2_bounds (list, optional): Custom bin edges for var2 if using equal_intervals.
            var1_title (str, optional): Custom title for var1 axis in legend.
            var2_title (str, optional): Custom title for var2 axis in legend.
            legend1_title (str, optional): Title for legend x-axis.
            legend2_title (str, optional): Title for legend y-axis.
            legend_title (str, optional): Title for overall legend.
            title (str, optional): Main title of the map.
            subtitle (str, optional): Subtitle text.
            annotation (str, optional): Annotation text.
            binning (str, optional): Method for binning ("quantiles" or "equal_intervals"). Default "quantiles".
            nbins (int, optional): Number of bins for classification. Default 4.
            zoom_to (dict, optional): Filter regions for zoomed view. Default None.
            kwargs (dict, optional): Additional config overrides.
            key (str, optional): Map config key. Default "bivariate_choropleth".

        Returns:
            matplotlib.axes.Axes: Matplotlib Axes with the bivariate choropleth.

        Raises:
            ValueError: If self.data is empty or required variables are missing.
        """
        # Refresh config and apply any updates
        self.refresh()
        if kwargs is not None:
            self.update(key, kwargs)
        config = self.map_config[key]

        # Copy and reproject data
        if data is None:
            data = self.dm.data.copy()

        # Ensure data is not empty
        if data.empty:
            raise ValueError(
                "Data is empty. Cannot plot bivariate choropleth."
            )
        # Ensure both variables exist
        for var in [var1, var2]:
            if var not in data.columns:
                raise ValueError(
                    f"Variable '{var}' not found in self.data columns."
                )

        var1 = var1
        var2 = var2

        data = data.to_crs(config["crs"])

        # Create figure
        fig, ax = plt.subplots(
            figsize=(config["figsize_x"], config["figsize_y"]),
            dpi=config["dpi"],
        )

        # Dissolve national geometry and fill geometry holes
        dissolved = data.dissolve("iso_code")
        dissolved.geometry = dissolved.geometry.apply(data_utils._fill_holes)

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
            legend_y = (
                ax_pos.y0 + 2 * (ax_pos.height - legend_height) / 5
            )  # vertically centered
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
        var1_labels = [data_utils._humanize(x) for x in var1_edges]
        var2_labels = [data_utils._humanize(x) for x in var2_edges]
        tickpos = np.linspace(0, 1, nbins + 1)
        ax2.set_xticks(
            tickpos, var1_labels, fontsize=config["legend_fontsize"]
        )
        ax2.set_yticks(
            tickpos, var2_labels, fontsize=config["legend_fontsize"]
        )

        # Legend axis titles
        if legend1_title is None:
            legend1_title = self._get_title(var1, "legend_titles", legend=True)
        if legend2_title is None:
            legend2_title = self._get_title(var2, "legend_titles", legend=True)

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
                zoom_to,
                country,
                subunit,
                data,
                dissolved,
                fig,
                ax,
                ax2,
                config,
                x=xpos,
            )

        def remove_duplicates(s1: str, s2: str) -> str:
            """Merge two titles by removing shared prefix from the second string."""
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

        # Add titles and annotations with layout adjusted to legend
        self._add_titles_and_annotations(
            fig, ax, config, title, subtitle, annotation, x=xpos
        )
        ax.axis("off")

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
        zoom_to: dict = None,
        kwargs: dict = None,
        key="choropleth",
    ) -> matplotlib.axes.Axes:
        """
        Plot a choropleth map of the given variable, with support for single-value maps,
        colorbar legends, barplot legends, missing data hatching, zooming, and annotations.

        Args:
            var (str): Column name in `self.data` to visualize.
            var_title (str, optional): Title for the variable; defaults to None.
            title (str, optional): Main plot title; defaults to None.
            subtitle (str, optional): Subtitle for the plot; defaults to None.
            legend_title (str, optional): Legend title; defaults to None.
            annotation (str, optional): Additional annotation text; defaults to None.
            var_bounds (list, optional): [min, max] bounds for color scaling; defaults to [None, None].
            zoom_to (dict, optional): Dictionary to select subregions to zoom; defaults to None.
            update_config (dict, optional): Configuration updates; defaults to None.

        Returns:
            matplotlib.axes.Axes: The main axis containing the choropleth map.

        Raises:
            ValueError: If `self.data` is empty, or if the variable column is missing.
            ValueError: If `binning` method is invalid.
        """
        # Refresh config and apply any updates
        self.refresh()
        if kwargs is not None:
            self.update(key, kwargs)
        config = self.map_config[key]

        # Ensure CRS matches map config
        if data is None:
            data = self.dm.data.copy()

        if data.empty:
            raise ValueError("self.dm.data is empty. Cannot plot choropleth.")
        if var not in data.columns:
            raise ValueError(
                f"Variable '{var}' not found in self.dm.data columns."
            )

        # ISO code for country labeling
        data = data.to_crs(config["crs"])

        legend_title = legend_title or self._get_title(var, "legend_titles")

        # Create figure and axis
        fig, ax = plt.subplots(
            figsize=(config["figsize_x"], config["figsize_y"]),
            dpi=config["dpi"],
        )

        # Choose colormap
        cmap = getattr(cmaps, config["cmap"])

        # Dissolve geometries for plotting boundaries
        dissolved = data.dissolve("iso_code")
        dissolved.geometry = dissolved.geometry.apply(data_utils._fill_holes)

        # Optionally zoom to subregions
        dissolved_zoomed = None
        if zoom_to is not None:
            data = []
            for key, value in zoom_to.items():
                selected = self.dm.data[
                    self.dm.data[key].isin([value])
                ].to_crs(config["crs"])
                if selected.empty:
                    raise ValueError(f"{value} is not in {key}.")
                data.append(selected)

            data = gpd.GeoDataFrame(pd.concat(data), geometry="geometry")
            dissolved_zoomed = data.dissolve("iso_code")

        # Determine min/max bounds
        vmin, vmax = var_bounds
        if vmin is None:
            vmin = data[var].min()
        if vmax is None:
            vmax = data[var].max()
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
            label_text = data_utils._humanize(int(unique_value) * 1.0)

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

        elif config["legend_type"] == "default":
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
            # cmap = plt.get_cmap(config["cmap"])
            cmap = getattr(cmaps, config["cmap"])
            colors = [cmap(i / (nbins - 1)) for i in range(nbins)]

            # Create human-readable labels for bins
            labels = [
                f"{data_utils._humanize(var_bins[i])} – {data_utils._humanize(var_bins[i+1])}"
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
                    facecolor=c, edgecolor=config["edgecolor"], label=l
                )
                for c, l in zip(reversed(colors), reversed(labels))
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
                mticker.FuncFormatter(data_utils._humanize)
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
                legend_kwds=legend_kwds,
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
                    f"{data_utils._humanize(edge)} to {data_utils._humanize(edge+bin_width)}"
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
                    + str(data_utils._humanize(y))
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
                zoom_to,
                country,
                subunit,
                data,
                dissolved,
                fig,
                ax,
                iax,
                config,
                x=xpos,
            )

        # Add title, subtitle, and annotations
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
        Plot missing data regions with hatching and add a custom legend entry.

        Args:
            ax (matplotlib.axes.Axes): Matplotlib axis where the missing data will be plotted.
            data_missing (gpd.GeoDataFrame): GeoDataFrame containing geometries of regions with missing data.
            config (dict): Configuration dictionary with required style keys:
                - "crs": Coordinate reference system for plotting
                - "missing_color": Fill color for missing data
                - "missing_hatch": Hatch pattern for missing data
                - "missing_edgecolor": Border color
                - "missing_linewidth": Line width for borders

        Returns:
            matplotlib.axes.Axes: Axis with missing data plotted and legend added.
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
        Plot a small inset map (overview map) alongside the main map and legend.

        Args:
            zoom_to (str): The region or boundary to zoom into (currently unused but kept for future flexibility).
            country (str): Country name (currently unused inside this function, but useful for labeling context).
            subunit (str): Subunit name to display as a label (e.g., province or region).
            data (geopandas.GeoDataFrame): GeoDataFrame containing the main geometries for the region.
            dissolved (geopandas.GeoDataFrame): GeoDataFrame with dissolved country-level boundaries (background).
            fig (matplotlib.figure.Figure): Matplotlib figure object.
            ax (matplotlib.axes.Axes): Main map axes.
            ax2 (matplotlib.axes.Axes): Legend axes (used to align the tiny map vertically).
            config (dict): Configuration dictionary controlling label appearance (fontsize, bbox styles, etc.).
            x (float): Left x-coordinate for the tiny map, aligned relative to the legend.

        Returns:
            None: The function modifies the given `fig` by adding a tiny inset map.
        """
        # Get main axes and legend axes positions (in figure coordinates)
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
        Add title, subtitle, and annotation text to a matplotlib figure.

        Args:
            fig (matplotlib.figure.Figure): The figure to which text elements are added.
            ax (matplotlib.axes.Axes): The axes used to determine relative positioning.
            config (dict): Configuration dictionary containing text formatting options
                such as font sizes, colors, and positioning offsets.
            title (str): Main title text. If None, no title is added.
            subtitle (str): Subtitle text displayed below the title. If None, no subtitle is added.
            annotation (str): Annotation text (e.g., data sources) displayed below the plot.
                If None, no annotation is added.
            x (float, optional): The default x-position for all text elements.
                Overridden by values in `config` if present.

        Returns:
            None

        Raises:
            KeyError: If required keys are missing in `config` (e.g., font sizes or gap settings).
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
                font=self.bold_font,
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
                font=self.regular_font,
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
                font=self.regular_font,
            )

    def _get_title(
        self,
        var: str,
        config_key: str,
        legend: bool = False,
        mhs_name: str = "Multihazards",
        conflict_name: str = "Conflicts",
    ) -> str:
        """
        Generate a formatted legend title for a given variable based on configuration.

        Args:
            var (str): Variable name to match against legend title keys.
            config_key (str): Key in ``self.map_config`` containing legend title mappings.
            legend (bool, optional): If True, append extra legend text for BEM variables.

        Returns:
            str: The formatted legend title string. Falls back to a title-cased version
                 of the variable name with " Risk" appended if no match is found.

        Raises:
            AttributeError: If ``self.map_config`` does not contain the given
                ``config_key``.
        """
        legend_titles = self.map_config[config_key]
        mhs_name = mhs_name[:-1] if legend else mhs_name
        conflict_name = conflict_name[:-1] if legend else conflict_name

        asset_name = data_utils.find_matching_alias(
            var, self.map_config["assets_alias"]
        )
        title = data_utils.find_matching_alias(
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

                    title = data_utils.capitalize(title)
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

                    title = data_utils.capitalize(title)
                    break

        if title is None:
            exclude_columns = ["cause", "category", "type"]
            if "_" in var or "type" in var:
                title = data_utils.capitalize(var.replace("_", " ").title())
            elif not any(col in var for col in exclude_columns):
                title = data_utils.capitalize(f"{var} Risk")
            else:
                title = data_utils.capitalize(var)

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
                asset = data_utils.find_matching_alias(
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
        Build an annotation string from variable names and configured annotation sources.

        Args:
            var_list (list, optional): List of variable names to search for in
                the configured annotations. Defaults to an empty list.
            add_adm (bool, optional): Whether to append the administrative
                source (from ``self.dm.adm_source``) to the variable list.
                Defaults to True.

        Returns:
            str: A formatted annotation string that begins with "Source:"
            followed by matched annotations, each on a new line.

        Raises:
            AttributeError: If ``self.map_config`` does not contain an
                "annotations" key.
        """

        # Avoid mutable default arguments by initializing inside
        if var_list is None:
            var_list = []

        annotations = self.map_config["annotations"]
        annotation = "Source: \n"

        # Optionally add administrative source to variable list
        if add_adm:
            var_list += [self.dm.adm_source.lower()]

        anns = []  # Track unique annotations to avoid duplicates
        for var in var_list:
            for key, ann in annotations.items():
                if key in var:
                    if ann not in anns:
                        anns.append(ann)
                        annotation += ann + "\n"

        return annotation

    def _cut(self, series: pd.Series, var_bounds: list, nbins: int) -> tuple:
        """
        Bin a numeric series into discrete intervals, either using user-defined
        bounds or evenly spaced intervals.

        Args:
            series (pd.Series): Input numeric data to be binned.
            var_bounds (list): List of bin boundaries. If provided, must either
                contain exactly ``nbins + 1`` elements or will be linearly spaced
                between the first and last values.
            nbins (int): Number of bins to create.

        Returns:
            tuple:
                - pd.Series: Categorical Series with bin labels (0 to nbins-1).
                - np.ndarray: Array of bin edges used for cutting.
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
        Computes the relative height of a text string within a Matplotlib figure.

        Args:
            fig (plt.Figure): The Matplotlib figure object.
            text (str): The text string to measure.
            fontsize (float): The font size of the text.

        Returns:
            float: The height of the text relative to the figure's height (0-1 scale).
        """
        # Get the renderer for the figure
        renderer = fig.canvas.get_renderer()

        text = plt.text(0, 0, text, fontsize=fontsize)

        # Get bounding box
        bbox = text.get_window_extent(renderer=renderer)

        # Remove the temporary text
        text.remove()

        # Return text height relative to figure height
        return bbox.height / fig.bbox.height
