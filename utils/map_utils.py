import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

import copy
import contextily as ctx
import rasterio as rio
import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import pycountry
import folium
import geojson_rewind
import json

import seaborn as sns
import pypalettes
import pyfonts

import pycountry
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap
from rasterio.plot import show
from utils import data_utils
import rasterio.mask


regular = pyfonts.load_google_font("Roboto")
bold = pyfonts.load_google_font("Roboto", weight="bold")


class GeoPlot:
    def __init__(
        self, 
        map_config_file: str = "configs/map_config.yaml",
        adm_config_file: str = 'configs/adm_config.yaml'
    ):
        self.map_config = data_utils.read_config(map_config_file)
        self.adm_config = data_utils.read_config(adm_config_file)
        

    def update_map_config(self, key: str, config: dict):
        self.map_config[key].update(config)
        

    def _get_title(self, var: str, config_key:str):
        legend_titles = self.map_config[config_key]
        for key, title in legend_titles.items():
            if key == var:
                return title
            elif key in var:
                if "conflict" in var:
                    return title.format("conflict")
                elif "mhs" in var:
                    return title.format("multi-hazard")
                else:
                    hazard = var.replace("_" + key, "").replace("_", " ")
                    return title.format(hazard)
                    
        return var.replace("_", " ").title() + " Risk"

    
    def _get_annotation(self, var_list: list = []):
        annotations = self.map_config["annotations"]
        annotation = "Source: \n"

        anns = []
        for var in var_list:
            for key, ann in annotations.items():
                if key in var:
                    if ann not in anns:
                        anns.append(ann)
                        annotation += ann + "\n"
        return annotation

    
    def plot_raster(
        self,
        data: gpd.GeoDataFrame, 
        raster_name: str,
        title: str = None,
        subtitle: str = None, 
        legend_title: str = None,
        annotation: str = None,
        data_dir = "./data/"
    ):
        iso_code = data.iso_code.values[0]
        config = self.map_config["raster"]
        raster_file = os.path.join(data_dir, f"{iso_code}/{iso_code}_{raster_name.upper()}.tif")
        fig, ax = plt.subplots(figsize=(config['figsize_x'], config['figsize_y']),  dpi=config['dpi'])
        
        with rio.open(raster_file) as src:
            out_image = src.read(1)
            plot_data = np.copy(out_image) 
            plot_data[plot_data == src.nodata] = np.nan
            img = ax.imshow(
                plot_data,
                extent=[
                    src.bounds.left, src.bounds.right,
                    src.bounds.bottom, src.bounds.top
                ],
                cmap=config['cmap'],
                origin='upper'
            )
        
        # Add colorbar
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

        cbar.ax.set_title(
            legend_title, 
            fontsize=config['legend_title_fontsize'], 
            loc=config['legend_title_loc'], 
            x=config['legend_title_x'], 
            y=config['legend_title_y']
        )
        
        if title is None:
            country = pycountry.countries.get(alpha_3=iso_code).name
            var_title = self._get_title(raster_name, "var_titles").title()
            title = config['title'].format(var_title, country)
        if annotation is None:
            annotation = self._get_annotation([raster_name])
        self._add_titles_and_annotations(fig, config, title, subtitle, annotation)
            
        ax.axis("off")
    
            
    def plot_geoboundaries(
        self,
        data: gpd.GeoDataFrame,
        adm_level: str, 
        title: str = None,
        subtitle: str = None, 
        legend_title: str = None,
        annotation: str = None,
        group: str = 'group'
    ):
        config = self.map_config["geoboundaries"]
    
        iso_code = data.iso_code.values[0]
        grouping = None
        if iso_code in self.adm_config:
            grouping = self.adm_config[iso_code]
    
        fig, ax = plt.subplots(figsize=(config['figsize_x'], config['figsize_y']),  dpi=config['dpi'])
        data_adm = data.dissolve(adm_level).reset_index()
    
        if (grouping is not None) or (group in data.columns):
            cmap = ListedColormap(config["cmap"])
            edgecolor = config["edgecolor_with_group"]
            linewidth = config["linewidth_with_group"]
            
            if group not in data.columns:
                data[group] = data[adm_level].map(grouping)
                
            data.dissolve(group).reset_index().plot(
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
            legend = ax.get_legend()
            legend.set_bbox_to_anchor([
                config['group_legend_x'],
                config['group_legend_y'],
                config['group_legend_width'],
                config['group_legend_height'],
            ])
            legend.set_title(legend_title)
            legend._legend_box.align = config['group_legend_box_align']
        else:
            linewidth = config["linewidth_no_group"]
            edgecolor = config["edgecolor_no_group"]
        
        data_adm.plot(ax=ax, facecolor="none", edgecolor=edgecolor, lw=linewidth)
        data_adm.apply(lambda x: ax.annotate(
            text=x[adm_level].replace("(", "\n("), 
            xy=x.geometry.centroid.coords[0], ha='center', 
            fontsize=config["fontsize"],
            bbox=dict(
                facecolor=config["label_facecolor"], 
                edgecolor=config["label_edgecolor"], 
                lw=config["label_linewidth"], 
                alpha=config["label_alpha"], 
                boxstyle=config["label_boxstyle"]
            )
        ), axis=1);
        dissolved = data.dissolve("iso_code")
        dissolved.geometry = dissolved.geometry.apply(data_utils._fill_holes)
        dissolved.plot(ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none");
        
        self._add_titles_and_annotations(fig, config, title, subtitle, annotation)
        ax.axis("off")


    def plot_bivariate_choropleth(
        self,
        data: gpd.GeoDataFrame,
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
        nbins: int = 4
    ):
        config = self.map_config["bivariate_choropleth"]
        data = data.to_crs(config['crs'])
        iso_code = data.iso_code.values[0]

        if binning == "quantiles":
            var1_categories, var1_bins = pd.qcut(data[var1], nbins, labels=range(nbins), retbins=True)
            var2_categories, var2_bins = pd.qcut(data[var2], nbins, labels=range(nbins), retbins=True)
            
        elif binning == "equal_interval":
            def cut(series, var_bounds, nbins):
                def get_bins(var_bounds, nbins):
                    if len(var_bounds) == nbins + 1:
                        return var_bounds
                    else:
                        return np.linspace(var_bounds[0], var_bounds[-1], nbins+1)

                if var_bounds is not None:
                    var_bins = get_bins(var_bounds, nbins)  
                    return pd.cut(series, bins=var_bins, labels=range(nbins), retbins=True, include_lowest=True)
                else:
                    return pd.cut(series, nbins, labels=range(nbins), retbins=True, include_lowest=True)

            var1_categories, var1_bins = cut(data[var1], var1_bounds, nbins)
            var2_categories, var2_bins = cut(data[var2], var2_bounds, nbins)
             
        var1_edges = list(var1_bins)
        var2_edges = list(var2_bins)

        data_plot = data.copy()
        data_plot["bivariate"] = var1_categories.astype('str') + var2_categories.astype('str')
        cmap = config[f"cmap{nbins}"]

        index = 0
        cmap_dict = dict()
        for i in range(nbins):
            for j in range(nbins):
                cmap_dict[f"{i}{j}"] = cmap[index]
                index += 1

        data_plot["cmap"] = data_plot['bivariate'].map(cmap_dict)
        data_missing = data_plot[data_plot["cmap"].isna()]
        data_plot["cmap"] = data_plot["cmap"].fillna(config['missing_color'])

        mpl.rcParams['hatch.linewidth'] = 0.1
        fig, ax = plt.subplots(figsize=(config['figsize_x'], config['figsize_y']),  dpi=config['dpi'])
        data.to_crs(config["crs"]).plot(
            ax=ax,  
            color=data_plot["cmap"], 
            edgecolor=config['edgecolor'], 
            lw=config['linewidth'], 
        )
        if len(data_missing) > 0:
            data_missing.to_crs(config["crs"]).plot(
                ax=ax, 
                facecolor=config['missing_color'],
                hatch=config['missing_hatch'],
                edgecolor=config['missing_edgecolor'],
                lw=config['missing_linewidth'],
                legend=True
            )
            mpatch = [mpatches.Patch(
                facecolor=config['missing_color'], 
                hatch=config['missing_hatch'], 
                edgecolor=config['missing_edgecolor'], 
                linewidth=config['missing_linewidth']*1.5,
                label='No data'
            )]
            ax.legend(handles=mpatch, loc='lower right', frameon=False, fontsize=8)
        ax.axis("off")

        ncols = nbins
        nrows = nbins
        alpha = 1

        ax2 = fig.add_axes([0.2, 0.7, 0.1, 0.1])
        ax2.set_aspect('equal', adjustable='box')

        # Compute cell width/height
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

        var1_labels = [data_utils._humanize(x) for x in var1_edges]
        var2_labels = [data_utils._humanize(x) for x in var2_edges]

        tickpos = np.linspace(0, 1, nbins+1)
        ax2.set_xticks(tickpos, var1_labels, fontsize=5)
        ax2.set_yticks(tickpos, var2_labels, fontsize=5)

        if legend1_title is None:
            legend1_title = self._get_title(var1, "legend_titles")
        if legend2_title is None:
            legend2_title = self._get_title(var2, "legend_titles")

        ax2.set_xlabel(legend1_title, fontsize=6, ha='left')
        ax2.yaxis.set_label_coords(-0.35, 0)

        ax2.set_ylabel(legend2_title, fontsize=6, ha='left')
        ax2.xaxis.set_label_coords(0, -0.25)

        dissolved = data.dissolve("iso_code")
        dissolved.geometry = dissolved.geometry.apply(data_utils._fill_holes)
        dissolved.plot(ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none");

        if var1_title is None:
            var1_title = self._get_title(var1, "var_titles").title()
        if var2_title is None:
            var2_title = self._get_title(var2, "var_titles").title()
        if annotation is None:
            annotation = self._get_annotation([var1, var2])
        country = pycountry.countries.get(alpha_3=iso_code).name

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
        
        title = config['title'].format(var1_title, var2_title, country)
        self._add_titles_and_annotations(fig, config, title, subtitle, annotation)
                

    def plot_choropleth(
        self,
        data: gpd.GeoDataFrame,
        var: str, 
        var_title: str = None, 
        title: str = None,
        subtitle: str = None, 
        legend_title: str = None,
        annotation: str = None,
        vmin: float = None,
        vmax: float = None,
        legend: bool = False,
        adm_level: str="ADM3"
    ):
        config = self.map_config["choropleth"]
        
        data = data.to_crs(config['crs'])
        iso_code = data.iso_code.values[0]

        if legend_title is None:
            legend_title = self._get_title(var, "legend_titles")
        
        fig, ax = plt.subplots(figsize=(config['figsize_x'], config['figsize_y']),  dpi=config['dpi'])
        if config['create_cmap']:
            cmap = pypalettes.create_cmap(colors=config['colormap'], cmap_type=config['cmap_type'])
        else:
            cmap = pypalettes.load_cmap(config['palette_name'], cmap_type=config['cmap_type'])

        legend_kwds = dict()
        if config['legend_type'] == 'colorbar':
            legend = True
            legend_kwds = {
                'shrink': config['legend_shrink'], 
                'location': config['legend_location']
            }
            
        mpl.rcParams['hatch.linewidth'] = 0.1
        missing_kwds={
            "color": config['missing_color'],
            "edgecolor": config['missing_edgecolor'],
            "linewidth": config['missing_linewidth'],
            "hatch": config['missing_hatch'],
            "label": config['missing_label']
        }
    
        if 'relative' in var:
            data[var] = data[var] * 100
        if vmin is None and vmax is None:
            #if "relative" in var:
            #    vmin = 0
            #    vmax = 100
            #else:
            vmin = data[var].min()
            vmax = data[var].max()
            
        data.plot(
            var, 
            ax=ax, 
            legend=legend, 
            cmap=cmap, 
            edgecolor=config['edgecolor'], 
            scheme=None,
            lw=config['linewidth'], 
            legend_kwds=legend_kwds, 
            missing_kwds=missing_kwds,
            vmin=vmin,
            vmax=vmax
        )   
    
        if config['legend_type'] == 'colorbar':
            cbar = fig.axes[1]
            cbar.tick_params(labelsize=config['legend_label_fontsize'])
            cbar.set_title(
                legend_title, 
                fontsize=config['legend_title_fontsize'], 
                loc=config['legend_title_loc'], 
                x=config['legend_title_x'], 
                y=config['legend_title_y']
            )
            def custom_format(value, tick_number):
                if value > 1:
                    return f"{value:,.1f}"
                else:
                    return f"{value:,.2f}"
            cbar.yaxis.set_major_formatter(mticker.FuncFormatter(custom_format))
    
        elif config['legend_type'] == 'barplot':
            iax = ax.inset_axes(bounds=[
                config["barplot_x"],
                config["barplot_y"],
                config["barplot_width"],
                config["barplot_height"]
            ])   
            iax.set_xticks([])
            iax.spines[["top", "right", "bottom"]].set_visible(False)

            
            nbins = min(data[var].nunique(), config["barplot_nbins"])
            bins = np.linspace(vmin, vmax, nbins + 1)
            bin_width = bins[1] - bins[0]
            y_ticks = bins[:-1] + bin_width / 2
    
            colors = [cmap((val - min(bins)) / (max(bins) - min(bins))) for val in bins]
            n = iax.hist(data[var], bins=bins, orientation='horizontal', alpha=0)[0]
            iax.barh(y_ticks, n, height=bin_width, color=colors)
            
            iax.set_yticks(
                y_ticks, 
                labels=[
                    f"{data_utils._humanize(edge)} to {data_utils._humanize(edge+bin_width)}" 
                    for edge in bins[:-1]
                ], 
                size=config["barplot_tick_size"]
            ) 
    
            # Get the current y-axis tick label positions
            iax.figure.canvas.draw()
    
            # Get the bounding box of y-axis tick labels in display coords
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
            
        if data[var].isnull().any():
            missing_patch = mpatches.Patch(
                facecolor=config['missing_color'], 
                edgecolor=config['missing_edgecolor'], 
                hatch=config['missing_hatch'],
                label=config['missing_label'],
                linewidth=config['linewidth'] 
            )
            ax.legend(
                handles=[missing_patch], 
                fancybox=False, 
                frameon=False, 
                fontsize=config["missing_legend_fontsize"],
                loc=config["missing_legend_loc"],  
                bbox_transform=fig.transFigure
            )
    
        dissolved = data.dissolve("iso_code")
        dissolved.geometry = dissolved.geometry.apply(data_utils._fill_holes)
        dissolved.plot(ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none");        

        if var_title is None:
            var_title = self._get_title(var, "var_titles").title()
        if annotation is None:
            annotation = self._get_annotation([var])

        country = pycountry.countries.get(alpha_3=iso_code).name
        title = config['title'].format(var_title, country)

        self._add_titles_and_annotations(fig, config, title, subtitle, annotation)
        ax.axis("off")
    
    
    def plot_folium(
        self,
        data: gpd.GeoDataFrame,
        acled: gpd.GeoDataFrame,
        var: str, 
        var_title: str = None, 
        adm_level: str = "ADM3",
        acled_group_name: str = None
    ):
        config = self.map_config["folium"]

        if var_title is None:
            var_title = self._get_title(var, "var_titles").title()
    
        original_crs = data.crs
        centroid = data.dissolve("iso_code").to_crs(config["meter_crs"]).centroid
        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS(config["meter_crs"]), pyproj.CRS(original_crs), always_xy=True
        )
        x, y = transformer.transform(centroid.x.iloc[0], centroid.y.iloc[0])
    
        m = folium.Map(location=[y, x], tiles=config["tiles"], zoom_start=config["zoom_start"])
        key_on = f"feature.properties.{adm_level}_ID"
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
        
        style_function = lambda x: {
            'fillColor': '#ffffff', 
            'color': '#000000', 
            'fillOpacity': 0.1, 
            'weight': 0.1,
            'lineColor': '#ffffff', 
        }
        highlight_function = lambda x: {
            'fillColor': '#000000', 
            'color':'#000000', 
            'fillOpacity': 0.50, 
            'weight': 0.1
        }
        nil = folium.features.GeoJson(
            data,
            style_function=style_function, 
            highlight_function=highlight_function, 
            tooltip=folium.features.GeoJsonTooltip(
                fields=[adm_level, var], 
                aliases=[f'{adm_level}: ', f'{var_title}: ']
            ),
            control=False
        )
        m.add_child(nil)
        m.keep_in_front(nil)
    
    
        if acled is not None:
            points_group = folium.FeatureGroup(name=acled_group_name)
            fill_colors = ["#a8225e", "#bc5090", "#ff6361", "#ffa600"]
            points_var = "disorder_type"
        
            categories = acled[points_var].unique()
            acled["fill_color"] = None
            for fill_color, category in zip(fill_colors[:len(categories)], categories):
                acled.loc[acled[points_var] == category, "fill_color"] = fill_color
            
            for index, row in acled.iterrows():
                popup = """
                <b>Disorder type:</b> %s<br>
                <b>Event type:</b> %s<br>
                <b>Subevent type:</b> %s<br>
                """ % (
                    row['disorder_type'],
                    row['event_type'],
                    row['sub_event_type']
                )
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x], 
                    stroke=False,
                    fill=True, 
                    fill_color=row['fill_color'],
                    radius=4,
                    tooltip=popup
                ).add_to(points_group)
            
            
                points_group.add_to(m)
                m.keep_in_front(points_group)
                
        folium.LayerControl().add_to(m)
        return m
    
    
    def _add_titles_and_annotations(self, fig, config, title, subtitle, annotation):
        fig.text(
            x=config['title_x'], 
            y=config['title_y'], 
            s=title, 
            size=config['title_fontsize'], 
            font=bold
        )
        if subtitle is not None:
            fig.text(
                x=config['subtitle_x'], 
                y=config['subtitle_y'], 
                s=subtitle, 
                size=config['subtitle_fontsize'], 
                font=regular
            )
        if annotation is not None:
            fig.text(
                x=config['annotation_x'], 
                y=config['annotation_y'], 
                s=annotation, 
                size=config['annotation_fontsize'], 
                color=config['annotation_color'], 
                font=regular
            )


  