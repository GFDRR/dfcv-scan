import os
import math
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

import copy
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


class GeoPlot:
    def __init__(
        self, 
        dm,
        data_dir: str = "data/",
        map_config_file: str = None
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
        self.data = dm.data  # Store the dataset from the data manager
        self.data_dir = data_dir

        # Load package resources
        resources = importlib_resources.files("dfcv_colocation_mapping")

        # Use default map config if none provided
        if map_config_file is None:
            map_config_file = resources.joinpath("configs", "map_config.yaml")
            
        self.map_config_file = map_config_file

        self.regular_font = pyfonts.load_google_font("Roboto")
        self.bold_font = pyfonts.load_google_font("Roboto", weight="bold")

        # Load or refresh configuration from the YAML file
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
    
        Raises:
            KeyError: If the specified key does not exist in the current map configuration.
            TypeError: If `kwargs` is not a dictionary.
        """
        # Ensure the new config is a dictionary
        if not isinstance(kwargs, dict):
            raise TypeError(f"`config` must be a dictionary, got {type(kwargs).__name__}")
    
        # Ensure the key exists in the current map configuration
        if key not in self.map_config:
            raise KeyError(f"Key '{key}' not found in map configuration")

        # Update the configuration for the specified key
        self.map_config[key].update(kwargs)      


    def plot_folium(
        self,
        var: str, 
        var_title: str = None, 
        adm_level: str = "ADM3",
        precision: int = 4,
        kwargs: dict = None,
        key = "folium"
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
    
        Raises:
            ValueError: If `self.data` is empty or if `var` is not in the data columns.
        """
        # Ensure data is not empty
        if self.data.empty:
            raise ValueError("Data is empty. Cannot create folium map.")
        
        # Ensure the variable exists
        if var not in self.data.columns:
            raise ValueError(f"Variable '{var}' not found in data columns.")

        # Refresh configuration and apply any overrides
        self.refresh()
        if kwargs is not None:
            self.update(key, kwargs)
        config = self.map_config[key]

        # Default variable title
        if var_title is None:
            var_title = self._get_title(var, "var_titles").title()

        data = self.data.copy()
        original_crs = data.crs

        # Get centroid of the country for map centering
        centroid = data.dissolve("iso_code").to_crs(config["meter_crs"]).centroid
        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS(config["meter_crs"]), pyproj.CRS(original_crs), always_xy=True
        )
        x, y = transformer.transform(centroid.x.iloc[0], centroid.y.iloc[0])

        # Initialize folium map
        m = folium.Map(location=[y, x], tiles=config["tiles"], zoom_start=config["zoom_start"])
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
            legend_name=var_title
        ).add_to(m)

        # Style and highlight functions for tooltips
        style_function = lambda x: config["style_function"]
        highlight_function = lambda x: config["highlight_function"]

        # Add transformed variable column for tooltips
        var_trans = var+"_transformed"
        data[var_trans] = data[var].apply(lambda x: round(x, precision))   

        # Add GeoJson layer with tooltips
        nil = folium.features.GeoJson(
            data,
            style_function=style_function, 
            highlight_function=highlight_function, 
            tooltip=folium.features.GeoJsonTooltip(
                fields=[adm_level, var_trans], 
                aliases=[f'{adm_level}: ', f'{var_title}: ']
            ),
            control=False
        )
        m.add_child(nil)
        m.keep_in_front(nil)

        # Add layer control
        folium.LayerControl().add_to(m)

        return m
    
    
    def plot_raster(
        self,
        raster_name: str,
        title: str = None,
        subtitle: str = None, 
        legend_title: str = None,
        annotation: str = None,
        kwargs: dict = None,
        key = "raster"
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

        data = self.dm.data.copy()
        if data.empty:
            raise ValueError("Data is empty. Cannot plot raster.")
        if 'iso_code' not in data.columns:
            raise ValueError("'iso_code' column not found in data.")
            
        iso_code = data.iso_code.values[0]
        
        raster_file = os.path.join(self.data_dir, f"{iso_code}/{iso_code}_{raster_name.upper()}.tif")
        if not os.path.exists(raster_file):
            raise FileNotFoundError(f"Raster file not found: {raster_file}")

        # Create figure and axis
        fig, ax = plt.subplots(
            figsize=(config['figsize_x'], config['figsize_y']),  
            dpi=config['dpi']
        )

        # Open raster and mask no-data values
        with rio.open(raster_file) as src:
            out_image = src.read(1)
            plot_data = np.array(np.copy(out_image), dtype=np.float32)
            plot_data[plot_data == src.nodata] = np.nan

            if "heat_stress" in raster_name.lower():
                plot_data = plot_data / 100
            
            img = ax.imshow(
                plot_data,
                extent=[
                    src.bounds.left, src.bounds.right,
                    src.bounds.bottom, src.bounds.top
                ],
                cmap=config['cmap'],
                origin='upper'
            )

        # Setup colorbar axes and properties
        bbox_anchor = [
            config["cbar_bbox_x"], 
            config["cbar_bbox_y"], 
            config["cbar_bbox_width"],
            config["cbar_bbox_height"]
        ]

        axins = inset_axes(
            ax, 
            width=config["cbar_width"], 
            height=config["cbar_height"],
            loc=config["cbar_loc"],
            bbox_to_anchor=bbox_anchor,
            bbox_transform=ax.transAxes, 
            borderpad=0
        )
        
        cbar = fig.colorbar(img, cax=axins, orientation="vertical", pad=config["cbar_pad"])
        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=config["cbar_fontsize"])
        
        if legend_title is None:
            legend_title = self._get_title(raster_name, "legend_titles")

        # Add title to colorbar
        cbar.ax.set_title(
            legend_title, 
            fontsize=config['legend_title_fontsize'], 
            loc=config['legend_title_loc'], 
            x=config['legend_title_x'], 
            y=config['legend_title_y']
        )

        # Determine left position of legend for alignment
        tight_bbox = cbar.ax.get_tightbbox(fig.canvas.get_renderer())
        tight_bbox_fig = tight_bbox.transformed(fig.transFigure.inverted())
        xpos = tight_bbox_fig.x0

        # Add dissolved country outline
        dissolved = data.dissolve("iso_code")
        dissolved.geometry = dissolved.geometry.apply(data_utils._fill_holes)
        dissolved.plot(ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none");
        
        if title is None:
            #country = pycountry.countries.get(alpha_3=iso_code).name
            country = self.dm.country
            var_title = self._get_title(raster_name, "var_titles").title()
            title = config['title'].format(var_title, country)
        if annotation is None:
            annotation = self._get_annotation([raster_name], add_adm=False)
        else:
            annotation = self._get_annotation([raster_name], add_adm=False) + f"{annotation}\n"

        # Add titles and annotations with layout adjusted to legend
        self._add_titles_and_annotations(fig, ax, config, title, subtitle, annotation, x=xpos)  
        ax.axis("off")

        return ax


    def plot_points(
        self, 
        column: str = None, 
        dataset: str = "acled",
        ax: matplotlib.axes.Axes = None,
        xpos: float = None,
        clustering: bool = True,
        distance: int = 50,
        kwargs: dict = None,
        key: str = "points"
    ):
        self.refresh()
        if kwargs is not None:
            self.update(key, kwargs)
        config = self.map_config[key]
        
        data = self.data.copy()
        iso_code = data.iso_code.values[0]

        # Initialize figure
        if ax is None:   
            fig, ax = plt.subplots(
                figsize=(config['figsize_x'], config['figsize_y']),  
                dpi=config['dpi']
            )
            data_adm = data.dissolve(self.dm.adm_level).reset_index()
            data_adm.to_crs(config["crs"]).plot(
                ax=ax, 
                facecolor="none", 
                edgecolor=config["edgecolor"], 
                lw=config["linewidth"]
            )

        if 'legend_x' in config:
            xpos = config["legend_x"]
        if 'legend_y' in config:
            ypos = config["legend_y"]
        bbox_to_anchor = [xpos, ypos]

        if dataset == "acled":
            data = self.dm.acled
        elif dataset == "ucdp":
            data = self.dm.ucdp

        if column not in data.columns:
            warnings.warn(f"{column} is not in the {dataset.upper()} dataset.")
            return 

        markerscale = config["markerscale"]
        categories = sorted(data[column].unique())  
        cmap = plt.get_cmap(config["cmap"], len(categories))
        colors = cmap.colors
        
        if not clustering:            
            points = data.to_crs(config["crs"]).plot(
                ax=ax,
                cmap=cmap,
                column=column,
                legend=False,
                markersize=8,
                alpha=config["alpha"],
                lw=0
            )
                
            handles = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(index), markersize=10, label=label)
                for index, label in enumerate(categories)
            ]
    
            title = column.replace("_", " ").title()
            legend = ax.legend(
                handles=handles,
                title=title,
                loc="center left",
                markerscale=0.75,
                fontsize=config["legend_label_fontsize"],
                title_fontsize=config["legend_title_fontsize"],
                bbox_to_anchor=bbox_to_anchor,
                bbox_transform=ax.figure.transFigure  
            )
            ax.add_artist(legend)

        else:
            # Source: https://stackoverflow.com/a/53094495/4777141
            all_points, handles = [], []
            global_index = 0
            for category, color in zip(data[column].unique(), colors):
                color = matplotlib.colors.rgb2hex(color)
                subdata = data[data[column] == category].copy()
                subdata["lon"] = subdata.geometry.x
                subdata["lat"] = subdata.geometry.y
                subdata["group"] = None
                coords = set([tuple(x) for x in subdata[["lat", "lon"]].values])
                
                clusters = []
                while len(coords):
                    locus = coords.pop()
                    cluster = [x for x in coords if vincenty(locus, x) <= distance]
                    clusters.append(cluster + [locus])
                    for x in cluster:
                        coords.remove(x)

                lons, lats, groups = [], [], []
                for cluster in clusters:
                    centroid_x = MultiPoint(cluster).centroid.x
                    centroid_y = MultiPoint(cluster).centroid.y
                    centroid = (centroid_x, centroid_y)
                    center_point = min(cluster, key=lambda point: vincenty(point, centroid))

                    for point in cluster:
                        condition = (subdata["lon"] == point[1]) & (subdata["lat"] == point[0])
                        subdata.loc[condition, "group"] = global_index
                        
                    lons.append(center_point[1])
                    lats.append(center_point[0])
                    groups.append(global_index)
                    global_index += 1
          
                points = pd.DataFrame({'lon': lons, 'lat': lats, 'group': groups})
                points = gpd.GeoDataFrame(
                    points, 
                    geometry=gpd.points_from_xy(points["lon"], points['lat']), 
                    crs="EPSG:4326"
                )
                points["color"] = color

                handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=category)
                handles.append(handle)

                counts = pd.DataFrame(subdata["group"].value_counts()).reset_index()
                points = points.merge(counts, on="group")
                all_points.append(points)
            
            all_points = gpd.GeoDataFrame(pd.concat(all_points), geometry="geometry")
            all_points = all_points.sort_values(by='count', ascending=False)

            max_count = all_points["count"].max()
            all_points["count_scaled"] = all_points["count"] * markerscale

            all_points.to_crs(config["crs"]).plot(
                ax=ax,
                facecolor=all_points["color"],
                legend=False,
                marker = "o",
                markersize="count_scaled",
                alpha=config["alpha"],
                lw=0.1
            )
            title = column.replace("_", " ").title()
            legend1 = ax.legend(
                handles=handles,
                title=title,
                loc="center left",
                markerscale=0.75,
                fontsize=config["legend_label_fontsize"],
                title_fontsize=config["legend_title_fontsize"],
                bbox_to_anchor=bbox_to_anchor,
                bbox_transform=ax.figure.transFigure  
            )
            ax.add_artist(legend1)

            lw = 0
            legend_color = "silver"
            mec="silver"

            import math

            def nice_round(x):
                if x <= 10:
                    return 10
                elif x <= 50:
                    return math.ceil(x / 10) * 10
                elif x <= 100:
                    return math.ceil(x / 50) * 50
                elif x <= 500:
                    return math.ceil(x / 100) * 100
                elif x <= 1000:
                    return math.ceil(x / 500) * 500
                else:
                    return math.ceil(x / 1000) * 1000
            
            def make_legend_ticks(max_count):
                """Generate legend ticks starting at 5/10, ending at rounded max, with all intermediate nice multiples."""
                # min value
                min_val = 5 if max_count <= 20 else 10
                
                # max rounded nicely
                max_val = nice_round(max_count)
                                
                # pick only those <= max_val
                if max_val >= 1000:
                    nice_values = [10, 100, 500, 1000, 5000, 10000, 50000, 100000]
                    multiples = [v for v in nice_values if v < max_val]
                else:
                    nice_values = [10, 50, 100, 500, 1000]
                    multiples = [v for v in nice_values if v < max_val]
                
                
                # include min, all multiples, then max
                ticks = [min_val] + multiples + [max_val]
                
                # ensure at least 3 ticks
                if len(ticks) < 3:
                    mid = (min_val + max_val) // 2
                    ticks.insert(1, nice_round(mid))
                
                return ticks
            
            ticks = make_legend_ticks(max_count)
            
            legends = []
            for n in ticks:
                legend = mlines.Line2D(
                    [], [],
                    color=legend_color,
                    lw=lw,
                    marker="o",
                    mec=mec,
                    markeredgewidth=1,
                    markersize=np.sqrt(n * markerscale),
                    label=n
                )
                legends.append(legend)

            # Force draw to get sizes
            ax.figure.canvas.draw()
            renderer = ax.figure.canvas.get_renderer()
            
            # Get legend1 height in figure fraction
            bb1 = legend1.get_window_extent(renderer).transformed(ax.figure.transFigure.inverted())
            h1 = bb1.height
            center1 = bb1.y0 + h1/2  # center y of legend1

            from matplotlib.patches import Circle
            from matplotlib.legend_handler import HandlerPatch
            
            class HandlerStackedCircles(HandlerPatch):
                def __init__(self, sizes, labels, title="Number of events", color="silver", **kwargs):
                    super().__init__(**kwargs)
                    self.sizes = sizes    
                    self.labels = labels  
                    self.title = title
                    self.color = color
            
                def create_artists(
                    self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
                ):
                    artists = []
                    max_r = max(self.sizes) / 2
                    center_x = width / 2 - xdescent
                    bottom_y = height / 2 - ydescent - max_r  # align bottoms
            
                    label_x = center_x + max_r + 5
            
                    for s, lbl in sorted(zip(self.sizes, self.labels), reverse=True):
                        r = s / 2
                        c = Circle((center_x, bottom_y + r), radius=r,
                                   facecolor="none", edgecolor=self.color, lw=1)
                        c.set_transform(trans)
                        artists.append(c)
            
                        # Text label
                        t = plt.Text(x=label_x,
                                     y=bottom_y + 1.85*r,   # top of circle
                                     text=str(lbl),
                                     va="center_baseline", ha="left",
                                     fontsize=fontsize)
                        t.set_transform(trans)
                        artists.append(t)
            
                    # -Add title to the top
                    title_y = bottom_y + 2*max_r + fontsize  # a bit above largest circle
                    title = plt.Text(x=center_x,
                                     y=title_y,
                                     text=self.title,
                                     va="bottom", ha="center",
                                     fontsize=fontsize, fontweight="bold")
                    title.set_transform(trans)
                    artists.append(title)
            
                    return artists
            
            
            def get_legend2(legends):
                sizes = [h.get_markersize() for h in legends]
                labels = [h.get_label() for h in legends]
            
                dummy = Circle((0, 0), radius=1)
            
                return ax.legend(
                    [dummy],
                    [""],  # suppress normal text, we draw title ourselves
                    handler_map={Circle: HandlerStackedCircles(
                        sizes=sizes,
                        labels=labels,
                        title="Number of events",
                        color="silver"
                    )},
                    loc="center left",
                    frameon=False,
                    borderpad=1,
                    handletextpad=2,
                    labelspacing=config["labelspacing"],
                    fontsize=config["legend_label_fontsize"],
                    bbox_to_anchor=bbox_to_anchor,
                    bbox_transform=ax.figure.transFigure
                )
                
            # Create legend2 temporarily to measure its height
            legend2_temp = get_legend2(legends)
            ax.add_artist(legend2_temp)
            ax.figure.canvas.draw()
            bb2 = legend2_temp.get_window_extent(renderer).transformed(ax.figure.transFigure.inverted())
            h2 = bb2.height
            
            # Compute new y-coordinate for legend2 so it sits exactly below legend1
            new_y = center1 - (h1/2 + h2/2) - 0.05
            
            #bbox_to_anchor = [xpos, ypos - bb1.height - gap]  
            bbox_to_anchor = [xpos + 0.025, new_y]  
            legend2_temp.remove()
            legend2 = get_legend2(legends)
            ax.add_artist(legend2)
                
        return ax, xpos
    
            
    def plot_geoboundaries(
        self,
        adm_level: str, 
        title: str = None,
        subtitle: str = None, 
        legend_title: str = None,
        annotation: str = None,
        group: str = 'group',
        max_adms: int = 50,
        max_groups: int = 30,
        show_adm_names: bool = True,
        kwargs: dict = None,
        key = "geoboundaries"
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
        if self.data.empty:
            raise ValueError("Data is empty. Cannot plot geoboundaries.")
        if adm_level not in self.data.columns:
            raise ValueError(f"Column '{adm_level}' not found in data.")

        # Refresh config and apply any updates
        self.refresh()
        if kwargs is not None:
            self.update(key, kwargs)
        config = self.map_config[key]
        
        data = self.data.copy()
        iso_code = data.iso_code.values[0]

        # Initialize figure
        fig, ax = plt.subplots(
            figsize=(config['figsize_x'], config['figsize_y']),  
            dpi=config['dpi']
        )
        data_adm = data.dissolve(adm_level).reset_index()

        # Set default legend title if none provided
        if legend_title is None:
            legend_title = config["legend_title"]

        xpos = 0
        if group in data.columns and data[group].nunique() < max_groups:
            cmap = ListedColormap(config["cmap"])
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
                    'loc': config['group_legend_loc'], 
                    'fontsize': config['group_legend_fontsize'],
                    'title_fontsize': config['group_legend_title_fontsize']
                }
            )

            # Customize legend position and appearance
            legend = ax.get_legend()
            legend.set_bbox_to_anchor([
                config['group_legend_x'],
                config['group_legend_y'],
                config['group_legend_width'],
                config['group_legend_height'],
            ])
            legend.set_title(legend_title)
            legend._legend_box.align = config['group_legend_box_align']

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
        data_adm.to_crs(config["crs"]).plot(ax=ax, facecolor="none", edgecolor=edgecolor, lw=linewidth)

        # Add labels if number of units is below threshold
        if len(data_adm) < max_adms and show_adm_names is True:
            data_adm.to_crs(config["crs"]).apply(lambda x: ax.annotate(
                text=x[adm_level].replace("(", "\n("), 
                xy=x.geometry.centroid.coords[0], 
                ha='center', 
                fontsize=config["fontsize"],
                bbox=dict(
                    facecolor=config["label_facecolor"], 
                    edgecolor=config["label_edgecolor"], 
                    lw=config["label_linewidth"], 
                    alpha=config["label_alpha"], 
                    boxstyle=config["label_boxstyle"]
                )
            ), axis=1)

        # Add dissolved country outline
        dissolved = data.dissolve("iso_code")
        dissolved.geometry = dissolved.geometry.apply(data_utils._fill_holes)
        dissolved.to_crs(config["crs"]).plot(ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none");

        # Set default title and annotation if missing
        iso_code = data.iso_code.values[0]
        countr = self.dm.country
        #country = pycountry.countries.get(alpha_3=iso_code).name

        # Get title text
        if title is None:
            title = config['title'].format(country)

        # Get annotation text
        if annotation is None:
            annotation = self._get_annotation()
        else:
            annotation = self._get_annotation() + f"{annotation}\n"

        # Add titles and annotations with layout adjusted to legend
        self._add_titles_and_annotations(fig, ax, config, title, subtitle, annotation, x=xpos)
        ax.axis("off")
        
        return ax, xpos


    def plot_bivariate_choropleth(
        self,
        var1: str,
        var2: str,
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
        binning: str = "quantiles",
        nbins: int = 4,
        zoom_to: dict = None,
        kwargs: dict = None,
        key = "bivariate_choropleth"
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

        # Ensure data is not empty
        if self.data.empty:
            raise ValueError("Data is empty. Cannot plot bivariate choropleth.")

        # Ensure both variables exist
        for var in [var1, var2]:
            if var not in self.data.columns:
                raise ValueError(f"Variable '{var}' not found in self.data columns.")

        # Refresh config and apply any updates
        self.refresh()
        if kwargs is not None:
            self.update(key, kwargs)
        config = self.map_config[key]

        # Copy and reproject data
        data = self.data.copy().to_crs(config['crs'])
        iso_code = data.iso_code.values[0]

        # Create figure
        fig, ax = plt.subplots(
            figsize=(config['figsize_x'], config['figsize_y']),  
            dpi=config['dpi']
        )

        # Dissolve national geometry and fill geometry holes
        dissolved = data.dissolve("iso_code")
        dissolved.geometry = dissolved.geometry.apply(data_utils._fill_holes)

        # Apply zoom if requested
        dissolved_zoomed = None
        if zoom_to is not None:            
            data = []
            for key, value in zoom_to.items():
                selected = self.data[self.data[key].isin([value])].to_crs(config['crs'])
                data.append(selected)   
                
            data = gpd.GeoDataFrame(pd.concat(data), geometry="geometry")
            dissolved_zoomed = data.dissolve("iso_code")

        # Apply binning method for both variables
        if binning == "quantiles":
            var1_categories, var1_bins = pd.qcut(data[var1], nbins, labels=range(nbins), retbins=True)
            var2_categories, var2_bins = pd.qcut(data[var2], nbins, labels=range(nbins), retbins=True)
        elif binning == "equal_intervals":
            var1_categories, var1_bins = self._cut(data[var1], var1_bounds, nbins)
            var2_categories, var2_bins = self._cut(data[var2], var2_bounds, nbins)
             
        var1_edges = list(var1_bins)
        var2_edges = list(var2_bins)

        # Assign bivariate categories and colormap
        data_plot = data.copy()
        data_plot["bivariate"] = var1_categories.astype('str') + var2_categories.astype('str')
        cmap = config[f"cmap{nbins}"]

        # Build color lookup dictionary
        index = 0
        cmap_dict = dict()
        for i in range(nbins):
            for j in range(nbins):
                cmap_dict[f"{i}{j}"] = cmap[index]
                index += 1

        # Assign colors
        data_plot["cmap"] = data_plot['bivariate'].map(cmap_dict)
        data_missing = data_plot[data_plot["cmap"].isna()]
        data_plot["cmap"] = data_plot["cmap"].fillna(config['missing_color'])

        # Plot main choropleth
        data.to_crs(config["crs"]).plot(
            ax=ax,  
            color=data_plot["cmap"], 
            edgecolor=config['edgecolor'], 
            lw=config['linewidth'], 
        )
        if len(data_missing) > 0:
            ax = self._plot_missing(ax, data_missing, config)

        # Plot dissolved outline (national boundary)
        if dissolved_zoomed is not None:
            dissolved_zoomed.plot(ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none");  
        else:
            dissolved.plot(ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none");  
            
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
            legend_y = ax_pos.y0 + 2 * (ax_pos.height - legend_height) / 5  # vertically centered
        else:
            legend_y = ax_pos.y0 + 4 * (ax_pos.height - legend_height) / 5  

        # Create legend as inset axis
        ax2 = fig.add_axes([legend_x, legend_y, legend_width, legend_height])
        ax2.set_aspect('equal', adjustable='box')

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
                    color=cmap[color_index]
                )
                color_index += 1

        ax2.margins(x=0)
        ax2.spines[['right', 'top']].set_visible(False)

        # Legend tick labels
        var1_labels = [data_utils._humanize(x) for x in var1_edges]
        var2_labels = [data_utils._humanize(x) for x in var2_edges]
        tickpos = np.linspace(0, 1, nbins+1)
        ax2.set_xticks(tickpos, var1_labels, fontsize=config["legend_fontsize"])
        ax2.set_yticks(tickpos, var2_labels, fontsize=config["legend_fontsize"])

        # Legend axis titles
        if legend1_title is None:
            legend1_title = self._get_title(var1, "legend_titles")
        if legend2_title is None:
            legend2_title = self._get_title(var2, "legend_titles")

        ax2.set_xlabel(legend1_title, fontsize=6, ha='left')
        ax2.yaxis.set_label_coords(-0.35, 0)

        ax2.set_ylabel(legend2_title, fontsize=6, ha='left')
        ax2.xaxis.set_label_coords(0, -0.25)

        # Determine left position of legend for alignment
        tight_bbox = ax2.get_tightbbox(fig.canvas.get_renderer())
        tight_bbox_fig = tight_bbox.transformed(fig.transFigure.inverted())
        xpos = tight_bbox_fig.x0

        # Build titles and annotations
        if var1_title is None:
            var1_title = self._get_title(var1, "var_titles").title()
        if var2_title is None:
            var2_title = self._get_title(var2, "var_titles").title()
        if annotation is None:
            annotation = self._get_annotation([var1, var2])
        else:
            annotation = self._get_annotation([var1, var2]) + f"{annotation}\n"

        country = self.dm.country
        #country = pycountry.countries.get(alpha_3=iso_code).name

        # If zoomed, adjust titles and add tiny map
        if zoom_to is not None:
            subunit = ", ".join([value for value in zoom_to.values()])
            country = f"{subunit}, {country}"
            self._plot_tiny_map(zoom_to, country, subunit, data, dissolved, fig, ax, ax2, config, x=xpos)

        # Helper to clean up duplicate words in titles
        def get_names(var1_title, var2_title, name: str, remove_from_latter: bool = True):
            if name in var1_title.lower() and name in var2_title.lower():
                if remove_from_latter:
                    var2_title = var2_title.replace(name.title(), "").strip()
                else:
                    var1_title = var1_title.replace(name.title(), "").strip()
            return var1_title, var2_title

        var1_title, var2_title = get_names(var1_title, var2_title, "relative", remove_from_latter=True)
        var1_title, var2_title = get_names(var1_title, var2_title, "absolute", remove_from_latter=True)
        var1_title, var2_title = get_names(var1_title, var2_title, "exposure", remove_from_latter=False)

        # Get title text
        if title is None:
            title = config['title'].format(var1_title, var2_title, country)

        # Add titles and annotations with layout adjusted to legend
        self._add_titles_and_annotations(fig, ax, config, title, subtitle, annotation, x=xpos)
        
        return ax, xpos
                

    def plot_choropleth(
        self,
        var: str, 
        var_title: str = None, 
        title: str = None,
        subtitle: str = None, 
        legend_title: str = None,
        annotation: str = None,
        var_bounds: list = [None, None],
        zoom_to: dict = None,
        kwargs: dict = None,
        key = "choropleth"
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

        # Ensure data is not empty
        if self.data.empty:
            raise ValueError("self.data is empty. Cannot plot choropleth.")
    
        # Ensure the requested variable exists
        if var not in self.data.columns:
            raise ValueError(f"Variable '{var}' not found in self.data columns.")
        
        # Refresh config and apply any updates
        self.refresh()
        if kwargs is not None:
            self.update(key, kwargs)
        config = self.map_config[key]

        # Ensure CRS matches map config
        data = self.data.copy()
        data = data.to_crs(config['crs'])

        # ISO code for country labeling
        iso_code = data.iso_code.values[0]

        # Determine legend title
        if legend_title is None:
            legend_title = self._get_title(var, "legend_titles")

        # Create figure and axis
        fig, ax = plt.subplots(
            figsize=(config['figsize_x'], config['figsize_y']),  
            dpi=config['dpi']
        )

        # Choose colormap
        if config['create_cmap']:
            cmap = pypalettes.create_cmap(colors=config['colormap'], cmap_type=config['cmap_type'])
        else:
            cmap = pypalettes.load_cmap(config['cmap'], cmap_type=config['cmap_type'])

        # Dissolve geometries for plotting boundaries
        dissolved = data.dissolve("iso_code")
        dissolved.geometry = dissolved.geometry.apply(data_utils._fill_holes)  

        # Optionally zoom to subregions
        dissolved_zoomed = None
        if zoom_to is not None:            
            data = []
            for key, value in zoom_to.items():
                selected = self.data[self.data[key].isin([value])].to_crs(config['crs'])
                if selected.empty:
                    raise ValueError(f"{value} is not in {key}.")
                data.append(selected)   
                
            data = gpd.GeoDataFrame(pd.concat(data), geometry="geometry")
            dissolved_zoomed = data.dissolve("iso_code")

        # Setup legend parameters
        legend_kwds = dict()
        if config['legend_type'] == 'colorbar':
            legend_kwds = {'shrink': config['legend_shrink'], 'location': "left"}

        # Convert relative variables to percentages
        if 'relative' in var:
            data[var] = data[var] * 100

        # Determine min/max bounds
        vmin, vmax = var_bounds
        if vmin is None:
            vmin = data[var].min()
        if vmax is None:
            vmax = data[var].max()

        fig.canvas.draw()
        xpos = None

        # Handle case when all values are the same (single color map)
        if data[var].nunique() == 1:
            # Transform value and get color
            unique_value = data[var].dropna().unique()[0]
            cmap_value = unique_value / 100 if 'relative' in var else unique_value
            color = cmap(cmap_value) if 0 <= cmap_value <= 1 else cmap(0.5)

            # Plot single-color map
            data.plot(
                ax=ax,
                color=color,
                edgecolor=config["edgecolor"],
                linewidth=config["linewidth"]
            )

            # Add legend showing value
            label_text = data_utils._humanize(unique_value)
            
            # Create a single-color legend patch
            legend_patch = mpatches.Patch(
                facecolor=color,
                edgecolor=config["edgecolor"],
                label=label_text
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

        elif config['legend_type'] == 'colorbar':
            # Plot using a continuous colorbar legend
            data.plot(
                var, 
                ax=ax, 
                legend=True, 
                cmap=cmap, 
                edgecolor=config['edgecolor'], 
                lw=config['linewidth'], 
                legend_kwds=legend_kwds, 
                vmin=vmin,
                vmax=vmax
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
                cbar_y = ax.get_position().y0 + 0.5 * (ax.get_position().height - cbar_height) / 5
            else:
                cbar_y = ax.get_position().y0 + 2 * (ax.get_position().height - cbar_height) / 5
        
            cbar_x = pos.x0
            iax.set_position([cbar_x, cbar_y, cbar_width, cbar_height])

            iax.tick_params(labelsize=config['legend_label_fontsize'])
            iax.set_title(
                legend_title, 
                fontsize=config['legend_title_fontsize']
            )                
            iax.yaxis.set_major_formatter(mticker.FuncFormatter(data_utils._humanize))

            # Determine left position of legend for alignment
            tight_bbox = iax.get_tightbbox(fig.canvas.get_renderer())
            tight_bbox_fig = tight_bbox.transformed(fig.transFigure.inverted())
            xpos = tight_bbox_fig.x0
    
        elif config['legend_type'] == 'barplot':
            # Position inset axis for barplot relative to map axis
            ax_pos = ax.get_position()
            barplot_width = config["barplot_width"]
            barplot_height = config["barplot_height"]
            
            # Different x and y position depending on zoom mode
            if zoom_to is not None: 
                barplot_y = ax_pos.y0 + 2 * (ax_pos.height - barplot_height) / 5  
            else:
                barplot_y = ax_pos.y0 + 4 * (ax_pos.height - barplot_height) / 5

            barplot_x = ax_pos.x0 - 2 * barplot_width + config["barplot_x_offset"]
            barplot_y += config["barplot_y_offset"]

            # Create inset axis for histogram barplot
            iax = ax.inset_axes(bounds=[
                barplot_x,
                barplot_y,
                barplot_width,
                barplot_height
            ])   
            iax.set_xticks([])
            iax.spines[["top", "right", "bottom"]].set_visible(False)

            # Bin variable values into categories for histogram
            nbins = min(data[var].nunique(), config["barplot_nbins"])
            categories, bins = self._cut(data[var], [vmin, vmax], nbins)
            data["categories"] = categories.astype('Int64').fillna(-1)

            # Map bins to colors using cmap
            bin_width = bins[1] - bins[0]
            y_ticks = bins[:-1] + bin_width / 2

            # Map bins to colors using cmap
            colors = [cmap((val - min(bins)) / (max(bins) - min(bins))) for val in bins]
            color_mapping = {category: color for category, color in zip(range(nbins), colors)}
            color_mapping[-1] = config['missing_color']

            # Plot choropleth with bin-based colors
            data["colors"] = data["categories"].map(color_mapping)
            data.plot(
                ax=ax, 
                color=data["colors"],
                edgecolor=config['edgecolor'], 
                lw=config['linewidth'], 
                legend_kwds=legend_kwds, 
                vmin=vmin,
                vmax=vmax
            )  

            # Draw histogram bars in inset axis
            n = iax.hist(data[var], bins=bins, orientation='horizontal', alpha=0)[0]
            iax.barh(y_ticks, n, height=bin_width, color=colors)

            # Format y-axis ticks with bin ranges
            iax.set_yticks(
                y_ticks, 
                labels=[
                    f"{data_utils._humanize(edge)} to {data_utils._humanize(edge+bin_width)}" 
                    for edge in bins[:-1]
                ], 
                size=config["barplot_tick_size"]
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
                    config['legend_title_gap'], 
                    legend_title,
                    transform=iax.transAxes,
                    fontsize=config['legend_title_fontsize'],
                    va='bottom',
                    ha='left'
                )

            # Add bar labels showing counts + percentages
            for index, (x, y) in enumerate(zip(y_ticks, n)):
                percent = (y/sum(n))
                label = r"$\bf{" + str(
                    data_utils._humanize(y)
                ) + "}$" + f" ({percent * 100:.0f}%)"
                y_range = max(n) - min(n)
                y += 0.035 * y_range
                iax.text(
                    y, x, 
                    s=label, 
                    color=config["barplot_label_color"], 
                    size=config["barplot_label_size"],
                    va="center"
                )

            iax.tick_params(axis="y", length=2)

            # Determine left position of legend for alignment
            tight_bbox = iax.get_tightbbox(fig.canvas.get_renderer())
            tight_bbox_fig = tight_bbox.transformed(fig.transFigure.inverted())
            xpos = tight_bbox_fig.x0

        # Plot boundaries (zoomed or full)
        if dissolved_zoomed is not None:
            dissolved_zoomed.plot(ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none");  
        else:
            dissolved.plot(ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none");  

        # Plot missing data and legend
        if data[var].isnull().any():
            data_missing = data[data[var].isna()]
            ax = self._plot_missing(ax, data_missing, config)

        # Get variable legend title texts
        if var_title is None:
            var_title = self._get_title(var, "var_titles").title()

        # Get annotation text
        if annotation is None:
            annotation = self._get_annotation([var])
        else:
            annotation = self._get_annotation([var]) + f"{annotation}\n"

        # Plot tiny map
        country = self.dm.country
        #country = pycountry.countries.get(alpha_3=iso_code).name
        if zoom_to is not None:
            subunit = ", ".join([value for value in zoom_to.values()])
            country = f"{subunit}, {country}"
            self._plot_tiny_map(zoom_to, country, subunit, data, dissolved, fig, ax, iax, config, x=xpos)

        # Get title text
        if title is None:
            title = config['title'].format(var_title, country)

        # Add title, subtitle, and annotations
        self._add_titles_and_annotations(fig, ax, config, title, subtitle, annotation, x=xpos)
        ax.axis("off")
        
        return ax, xpos

    
    def _plot_missing(
        self, 
        ax: matplotlib.axes.Axes, 
        data_missing: gpd.GeoDataFrame, 
        config: dict
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
            
        Raises:
            TypeError: If `data_missing` is not a GeoDataFrame.
            ValueError: If `data_missing` is empty.
        """

        if not isinstance(data_missing, gpd.GeoDataFrame):
            raise TypeError("`data_missing` must be a GeoDataFrame.")
        if data_missing.empty:
            raise ValueError("`data_missing` GeoDataFrame is empty — cannot plot missing data.")

        # Set hatch linewidth globally (applies to all hatching in the plot)
        mpl.rcParams['hatch.linewidth'] = config["missing_hatch_linewidth"]

        # Plot missing data regions with hatching
        data_missing.to_crs(config["crs"]).plot(
            ax=ax, 
            facecolor=config['missing_color'],
            hatch=config['missing_hatch'],
            edgecolor=config['missing_edgecolor'],
            lw=config['missing_linewidth'],
            legend=True
        )

        # Create a custom legend patch for "No data"
        mpatch = [mpatches.Patch(
            facecolor=config['missing_color'], 
            hatch=config['missing_hatch'], 
            edgecolor=config['missing_edgecolor'], 
            linewidth=config['missing_linewidth']*1.5,
            label='No data'
        )]

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
            bbox_transform=ax.figure.transFigure           
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
        x: float
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

        Raises:
            ValueError: If `data` or `dissolved` is empty.
            KeyError: If `data` does not contain an 'iso_code' column.
            TypeError: If `x` is not numeric.
        """
        
        # Validation checks
        if data.empty:
            raise ValueError("`data` GeoDataFrame is empty — cannot plot tiny map.")
        if dissolved.empty:
            raise ValueError("`dissolved` GeoDataFrame is empty — cannot plot tiny map.")
        if "iso_code" not in data.columns:
            raise KeyError("`data` must contain an 'iso_code' column.")
        if not isinstance(x, (int, float)):
            raise TypeError("`x` must be a numeric value.")
        
        # Get main axes and legend axes positions (in figure coordinates)
        ax1_pos = ax1.get_position()       
        ax2_pos = ax2.get_position() 
    
        # Align tiny map horizontally with legend labels
        natural_height = ax1_pos.y1 - ax2_pos.y1
        
        # Max allowed height = 1/3 of ax height
        max_height = (ax1_pos.y1 - ax1_pos.y0) / 3
        
        # Choose the smaller of the natural height vs. max allowed
        total_gap = ax1_pos.y1 - ax2_pos.y1
        iax_height = min(natural_height, max_height)

        # Center tiny map vertically between main map and legend
        iax_y = ax2_pos.y1 + (total_gap - iax_height) / 2
    
        # Width (space between legend and ax) 
        iax_width = ax1_pos.x0 - x    
    
        # Create tiny map axes in figure coordinates
        iax = fig.add_axes([x, iax_y, iax_width, iax_height])
        iax.set_axis_off()

        # Plot background (dissolved country) and highlight the selected region
        dissolved.plot(ax=iax, facecolor="lightgray", edgecolor="lightgray", lw=1)
        data.dissolve("iso_code").plot(ax=iax, facecolor="bisque", edgecolor="sienna", lw=0.25)

        # Get bounding box for annotation placement
        xmin, ymin, xmax, ymax = data.dissolve("iso_code").total_bounds

        # Add subunit label centered above the selected area
        iax.annotate(text=subunit,
            xy=(xmin + abs(xmin - xmax)/2, ymax),
            xytext=(0, 5),  # offset upwards by 5 points
            textcoords='offset points',
            ha='center',     
            va='bottom',
            fontsize=config["fontsize"],
            bbox=dict(
                facecolor=config["label_facecolor"], 
                edgecolor=config["label_edgecolor"], 
                lw=config["label_linewidth"], 
                alpha=config["label_alpha"], 
                boxstyle=config["label_boxstyle"]
            )
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
        y0 = ax.get_position().y0  # bottom of the plot area
        y1 = ax.get_position().y1  # top of the plot area  

        # Title positioning defaults (can be overridden by config)
        title_x = config.get('title_x', x)
        title_y = config.get('title_y', y1)

        if subtitle is None:
            if 'conflict' in title.lower():
                start_date = datetime.strptime(self.dm.conflict_start_date, "%Y-%m-%d")
                end_date = datetime.strptime(self.dm.conflict_end_date, "%Y-%m-%d")
                subtitle = f"Conflict events from {start_date.year} to {end_date.year}"

        # Add subtitle (if provided)
        if subtitle is not None:
            subtitle_x = config.get('subtitle_x', x)
            subtitle_y = config.get('subtitle_y', y1)

            # Adjust title position to leave space for subtitle
            title_y += self._get_text_height(
                fig, 
                subtitle, 
                fontsize=config['subtitle_fontsize'], 
            ) + config['subtitle_gap']
              
            fig.text(
                x=subtitle_x, 
                y=subtitle_y, 
                s=subtitle, 
                size=config['subtitle_fontsize'], 
                font=self.regular_font
            )

        # Add title (if provided)
        if title is not None:                
            fig.text(
                x=title_x, 
                y=title_y, 
                s=title, 
                size=config['title_fontsize'], 
                font=self.bold_font
            )

        # Add annotation (if provided)
        if annotation is not None:
            annotation_x, annotation_y = x, y0
            annotation_x = config.get('annotation_x', title_x)

            text_height = self._get_text_height(
                fig, annotation, config['annotation_fontsize']
            )
            annotation_y = y0 - text_height - config['annotation_gap']
            annotation_y = config.get('annotation_y', annotation_y)
            
            fig.text(
                x=annotation_x, 
                y=annotation_y, 
                s=annotation, 
                size=config['annotation_fontsize'], 
                color=config['annotation_color'], 
                font=self.regular_font
            )

            
    def _get_title(self, var: str, config_key: str) -> str:
        """
        Generate a formatted legend title for a given variable based on configuration.
    
        Args:
            var (str): Variable name to match against legend title keys.
            config_key (str): Key in ``self.map_config`` containing legend title mappings.
    
        Returns:
            str: The formatted legend title string. Defaults to a title-cased
            version of the variable name with " Risk" appended if no match is found.
    
        Raises:
            AttributeError: If ``self.map_config`` does not contain the given 
                ``config_key``.
        """
        
        if config_key not in self.map_config:
            raise AttributeError(f"`map_config` must contain '{config_key}'.")

        # Retrieve legend title mappings from map_config
        legend_titles = self.map_config[config_key]

        # Try to match the variable with legend title keys
        for key, title in legend_titles.items():
            if key == var: 
                # Exact match
                return title
            elif key in var:
                # Partial match with special cases
                if "exposure" in var.lower():
                    var = var.replace("_" + self.dm.asset, "")
                    var = var.replace(self.dm.asset, "")
                if "acled" in var or 'ucdp' in var:
                    return title.format("conflict") 
                elif "mhs" in var:
                    return title.format("multi-hazard")
                else:
                    # Generic replacement for hazard names
                    inp = var.replace("_" + key, "").replace("_", " ")
                    return title.format(inp)

        # Fallback: return variable as a title-cased string with "Risk"
        return var.replace("_", " ").title() + " Risk"

    
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

        # Ensure map_config contains "annotations"
        if "annotations" not in self.map_config:
            raise AttributeError("`map_config` must contain an 'annotations' key.")
            
        annotations = self.map_config["annotations"]
        annotation = "Source: \n"

        # Optionally add administrative source to variable list
        if add_adm:
            var_list += [self.dm.adm_source.lower()]

        anns = [] # Track unique annotations to avoid duplicates
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
    
        Raises:
            ValueError: If ``nbins`` is not a positive integer.
            ValueError: If ``var_bounds`` is provided but has fewer than 2 elements.
        """
        
        def get_bins(var_bounds, nbins):
            """Helper to construct valid bin edges based on var_bounds and nbins."""
            if len(var_bounds) == nbins + 1:
                return var_bounds
            else:
                return np.linspace(var_bounds[0], var_bounds[-1], nbins+1)

        if nbins <= 0:
            raise ValueError("`nbins` must be a positive integer.")

        if var_bounds is not None:
            if len(var_bounds) < 2:
                raise ValueError("`var_bounds` must contain at least two elements.")
                
            # Generate bin edges based on user input or linear spacing
            var_bins = get_bins(var_bounds, nbins)  
            
            return pd.cut(
                series, 
                bins=var_bins, 
                labels=range(nbins), 
                retbins=True, 
                include_lowest=True
            )
        else:
            return pd.cut(
                series, 
                nbins, 
                labels=range(nbins), 
                retbins=True, 
                include_lowest=True
            )


    def _get_text_height(self, fig: plt.Figure, text: str, fontsize: float) -> float:
        """
        Computes the relative height of a text string within a Matplotlib figure.
    
        Args:
            fig (plt.Figure): The Matplotlib figure object.
            text (str): The text string to measure.
            fontsize (float): The font size of the text.
    
        Returns:
            float: The height of the text relative to the figure's height (0-1 scale).
    
        Raises:
            TypeError: If `fig` is not a Matplotlib Figure, `text` is not a string,
                       or `fontsize` is not a number.
        """
    
        # Input validation
        if not isinstance(fig, plt.Figure):
            raise TypeError(f"`fig` must be a matplotlib.figure.Figure, got {type(fig).__name__}")
        if not isinstance(text, str):
            raise TypeError(f"`text` must be a string, got {type(text).__name__}")
        if not isinstance(fontsize, (int, float)):
            raise TypeError(f"`fontsize` must be a number, got {type(fontsize).__name__}")
    
        # Get the renderer for the figure
        renderer = fig.canvas.get_renderer()
    
        # Create a temporary text object at (0, 0) to measure its size
        text = plt.text(0, 0, text, fontsize=fontsize)
    
        # Get bounding box
        bbox = text.get_window_extent(renderer=renderer)
    
        # Remove the temporary text
        text.remove()
    
        # Return text height relative to figure height
        return bbox.height / fig.bbox.height