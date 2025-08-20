import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

import copy
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
from dfcv_colocation_mapping import data_utils
import rasterio.mask

import importlib_resources

resources = importlib_resources.files("dfcv_colocation_mapping")
_map_config_file = resources.joinpath("configs", "map_config.yaml")


regular = pyfonts.load_google_font("Roboto")
bold = pyfonts.load_google_font("Roboto", weight="bold")


class GeoPlot:
    def __init__(
        self, 
        dm,
        map_config_file: str = None
    ):
        self.dm = dm
        self.data = dm.data  
        if map_config_file is None:
            map_config_file = _map_config_file
        self.map_config_file = map_config_file
        self.refresh_config()

    
    def refresh_config(self):
        self.map_config = data_utils.read_config(self.map_config_file)
        return self.map_config
        

    def update_config(self, key: str, config: dict):
        self.map_config[key].update(config)


    def plot_folium(
        self,
        var: str, 
        var_title: str = None, 
        adm_level: str = "ADM3",
        config: dict = None,
        precision: int = 4,
        config_key = "folium"
    ):
        self.refresh_config()
        if config is not None:
            self.update_config(key=config_key, config=config)
        config = self.map_config[config_key]

        if var_title is None:
            var_title = self._get_title(var, "var_titles").title()

        data = self.data.copy()
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
        
        style_function = lambda x: config["style_function"]
        highlight_function = lambda x: config["highlight_function"]

        var_trans = var+"_transformed"
        data[var_trans] = data[var].apply(lambda x: round(x, precision))   
        
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
                
        folium.LayerControl().add_to(m)
        return m
    
    
    def plot_raster(
        self,
        raster_name: str,
        title: str = None,
        subtitle: str = None, 
        legend_title: str = None,
        annotation: str = None,
        data_dir = "./data/",
        config: dict = None,
        config_key = "raster"
    ):
        self.refresh_config()
        if config is not None:
            self.update_config(key=config_key, config=config)
        config = self.map_config[config_key]

        data = self.dm.data.copy()
        iso_code = data.iso_code.values[0]
        
        raster_file = os.path.join(data_dir, f"{iso_code}/{iso_code}_{raster_name.upper()}.tif")
        fig, ax = plt.subplots(
            figsize=(config['figsize_x'], config['figsize_y']),  
            dpi=config['dpi']
        )
        
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

        tight_bbox = cbar.ax.get_tightbbox(fig.canvas.get_renderer())
        tight_bbox_fig = tight_bbox.transformed(fig.transFigure.inverted())
        legend_left = tight_bbox_fig.x0
        
        if title is None:
            country = pycountry.countries.get(alpha_3=iso_code).name
            var_title = self._get_title(raster_name, "var_titles").title()
            title = config['title'].format(var_title, country)
        if annotation is None:
            annotation = self._get_annotation([raster_name], add_adm=False)
        else:
            annotation = self._get_annotation([raster_name], add_adm=False) + f"{annotation}\n"
        self._add_titles_and_annotations(fig, ax, config, title, subtitle, annotation, x=legend_left)
            
        ax.axis("off")
    
            
    def plot_geoboundaries(
        self,
        adm_level: str, 
        title: str = None,
        subtitle: str = None, 
        legend_title: str = None,
        annotation: str = None,
        group: str = 'group',
        config: dict = None,
        config_key = "geoboundaries"
    ):
        self.refresh_config()
        if config is not None:
            self.update_config(key=config_key, config=config)
            
        config = self.map_config[config_key]
        data = self.data.copy()
        iso_code = data.iso_code.values[0]
    
        fig, ax = plt.subplots(figsize=(config['figsize_x'], config['figsize_y']),  dpi=config['dpi'])
        data_adm = data.dissolve(adm_level).reset_index()

        if legend_title is None:
            legend_title = config["legend_title"]

        legend_left = None
        if group in data.columns:
            cmap = ListedColormap(config["cmap"])
            edgecolor = config["edgecolor_with_group"]
            linewidth = config["linewidth_with_group"]
                
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

            fig.canvas.draw()
            bbox = legend.get_window_extent(fig.canvas.get_renderer())
            bbox_fig = bbox.transformed(fig.transFigure.inverted())
            legend_left = bbox_fig.x0
        else:
            linewidth = config["linewidth_no_group"]
            edgecolor = config["edgecolor_no_group"]
        
        data_adm.plot(ax=ax, facecolor="none", edgecolor=edgecolor, lw=linewidth)
        data_adm.apply(lambda x: ax.annotate(
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
        ), axis=1);
        dissolved = data.dissolve("iso_code")
        dissolved.geometry = dissolved.geometry.apply(data_utils._fill_holes)
        dissolved.plot(ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none");

        iso_code = data.iso_code.values[0]
        country = pycountry.countries.get(alpha_3=iso_code).name
        if title is None:
            title = config['title'].format(country)
        if annotation is None:
            annotation = self._get_annotation()
        else:
            annotation = self._get_annotation() + f"{annotation}\n"
            
        self._add_titles_and_annotations(fig, ax, config, title, subtitle, annotation, x=legend_left)
        ax.axis("off")
        return ax


    def _plot_missing(self, ax, data_missing: gpd.GeoDataFrame, config: dict):
        mpl.rcParams['hatch.linewidth'] = 0.1
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
        return ax


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
        update_config: dict = None
    ):
        config_key = "bivariate_choropleth"
        self.refresh_config()
        if update_config is not None:
            self.update_config(key=config_key, config=update_config)
            
        config = self.map_config[config_key]
        data = self.data.copy()
        data = data.to_crs(config['crs'])
        iso_code = data.iso_code.values[0]

        fig, ax = plt.subplots(figsize=(config['figsize_x'], config['figsize_y']),  dpi=config['dpi'])
        dissolved = data.dissolve("iso_code")
        dissolved.geometry = dissolved.geometry.apply(data_utils._fill_holes)

        dissolved_zoomed = None
        if zoom_to is not None:            
            data = []
            for key, value in zoom_to.items():
                selected = self.data[self.data[key].isin([value])].to_crs(config['crs'])
                data.append(selected)   
                
            data = gpd.GeoDataFrame(pd.concat(data), geometry="geometry")
            dissolved_zoomed = data.dissolve("iso_code")

        if binning == "quantiles":
            var1_categories, var1_bins = pd.qcut(data[var1], nbins, labels=range(nbins), retbins=True)
            var2_categories, var2_bins = pd.qcut(data[var2], nbins, labels=range(nbins), retbins=True)
            
        elif binning == "equal_intervals":
            var1_categories, var1_bins = self.cut(data[var1], var1_bounds, nbins)
            var2_categories, var2_bins = self.cut(data[var2], var2_bounds, nbins)
             
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
        
        data.to_crs(config["crs"]).plot(
            ax=ax,  
            color=data_plot["cmap"], 
            edgecolor=config['edgecolor'], 
            lw=config['linewidth'], 
        )
        if len(data_missing) > 0:
            ax = self._plot_missing(ax, data_missing, config)

        if dissolved_zoomed is not None:
            dissolved_zoomed.plot(ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none");  
        else:
            dissolved.plot(ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none");  
            
        ax.axis("off")

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
        
        ax2 = fig.add_axes([legend_x, legend_y, legend_width, legend_height])
        ax2.set_aspect('equal', adjustable='box')
        
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
        ax2.set_xticks(tickpos, var1_labels, fontsize=config["legend_fontsize"])
        ax2.set_yticks(tickpos, var2_labels, fontsize=config["legend_fontsize"])

        if legend1_title is None:
            legend1_title = self._get_title(var1, "legend_titles")
        if legend2_title is None:
            legend2_title = self._get_title(var2, "legend_titles")

        ax2.set_xlabel(legend1_title, fontsize=6, ha='left')
        ax2.yaxis.set_label_coords(-0.35, 0)

        ax2.set_ylabel(legend2_title, fontsize=6, ha='left')
        ax2.xaxis.set_label_coords(0, -0.25)

        tight_bbox = ax2.get_tightbbox(fig.canvas.get_renderer())
        tight_bbox_fig = tight_bbox.transformed(fig.transFigure.inverted())
        legend_left = tight_bbox_fig.x0

        if var1_title is None:
            var1_title = self._get_title(var1, "var_titles").title()
        if var2_title is None:
            var2_title = self._get_title(var2, "var_titles").title()
        if annotation is None:
            annotation = self._get_annotation([var1, var2])
        else:
            annotation = self._get_annotation([var1, var2]) + f"{annotation}\n"
        country = pycountry.countries.get(alpha_3=iso_code).name

        if zoom_to is not None:
            subunit = ", ".join([value for value in zoom_to.values()])
            country = f"{subunit}, {country}"
            self._plot_tiny_map(zoom_to, country, subunit, data, dissolved, fig, ax, ax2, config, x=legend_left)

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

        if title is None:
            title = config['title'].format(var1_title, var2_title, country)
        self._add_titles_and_annotations(fig, ax, config, title, subtitle, annotation, x=legend_left)
        return ax
                

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
        update_config: dict = None
    ):
        config_key = "choropleth"
        self.refresh_config()
        if update_config is not None:
            self.update_config(key=config_key, config=update_config)
        config = self.map_config[config_key]

        vmin, vmax = var_bounds
        data = self.data.copy()
        data = data.to_crs(config['crs'])
        iso_code = data.iso_code.values[0]

        if legend_title is None:
            legend_title = self._get_title(var, "legend_titles")
        
        fig, ax = plt.subplots(figsize=(config['figsize_x'], config['figsize_y']),  dpi=config['dpi'])
        if config['create_cmap']:
            cmap = pypalettes.create_cmap(colors=config['colormap'], cmap_type=config['cmap_type'])
        else:
            cmap = pypalettes.load_cmap(config['palette_name'], cmap_type=config['cmap_type'])

        dissolved = data.dissolve("iso_code")
        dissolved.geometry = dissolved.geometry.apply(data_utils._fill_holes)  

        dissolved_zoomed = None
        if zoom_to is not None:            
            data = []
            for key, value in zoom_to.items():
                selected = self.data[self.data[key].isin([value])].to_crs(config['crs'])
                data.append(selected)   
                
            data = gpd.GeoDataFrame(pd.concat(data), geometry="geometry")
            dissolved_zoomed = data.dissolve("iso_code")
            
        legend_kwds = dict()
        if config['legend_type'] == 'colorbar':
            legend = True
            legend_kwds = {
                'shrink': config['legend_shrink'], 
                'location': "left"
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
            
        if vmin is None:
            vmin = data[var].min()
        if vmax is None:
            vmax = data[var].max()

        fig.canvas.draw()
        pos = ax.get_position()
        legend_left = None
        
        if config['legend_type'] == 'colorbar':
            data.plot(
                var, 
                ax=ax, 
                legend=True, 
                cmap=cmap, 
                edgecolor=config['edgecolor'], 
                lw=config['linewidth'], 
                legend_kwds=legend_kwds, 
                missing_kwds=missing_kwds,
                vmin=vmin,
                vmax=vmax
            )  
            iax = fig.axes[1]
            iax.tick_params(labelsize=config['legend_label_fontsize'])
            iax.set_title(
                legend_title, 
                fontsize=config['legend_title_fontsize']
            )                
            iax.yaxis.set_major_formatter(mticker.FuncFormatter(data_utils._humanize))

            tight_bbox = iax.get_tightbbox(fig.canvas.get_renderer())
            tight_bbox_fig = tight_bbox.transformed(fig.transFigure.inverted())
            legend_left = tight_bbox_fig.x0
    
        elif config['legend_type'] == 'barplot':
            ax_pos = ax.get_position()
            barplot_width = 0.2
            barplot_height = 0.2
            
            barplot_x = ax_pos.x0 - 2 * barplot_width 
    
            if zoom_to is not None: 
                barplot_y = ax_pos.y0 + 2 * (ax_pos.height - barplot_height) / 5  
            else:
                barplot_y = ax_pos.y0 + 4 * (ax_pos.height - barplot_height) / 5
            
            iax = ax.inset_axes(bounds=[
                barplot_x,
                barplot_y,
                barplot_width,
                barplot_height
            ])   
            iax.set_xticks([])
            iax.spines[["top", "right", "bottom"]].set_visible(False)
            
            nbins = min(data[var].nunique(), config["barplot_nbins"])
            categories, bins = self.cut(data[var], [vmin, vmax], nbins)
            data["categories"] = categories.astype('Int64')
            data["categories"] = data["categories"].fillna(-1)
            
            bin_width = bins[1] - bins[0]
            y_ticks = bins[:-1] + bin_width / 2
    
            colors = [cmap((val - min(bins)) / (max(bins) - min(bins))) for val in bins]
            color_mapping = {category: color for category, color in zip(range(nbins), colors)}
            color_mapping[-1] = config['missing_color']

            data_missing = data[data["categories"] == -1]
            data["colors"] = data["categories"].map(color_mapping)

            data.plot(
                ax=ax, 
                color=data["colors"],
                edgecolor=config['edgecolor'], 
                lw=config['linewidth'], 
                legend_kwds=legend_kwds, 
                missing_kwds=missing_kwds,
                vmin=vmin,
                vmax=vmax
            )  
            if len(data_missing) > 0:
                ax = self._plot_missing(ax, data_missing, config)
            
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
            tight_bbox = iax.get_tightbbox(fig.canvas.get_renderer())
            tight_bbox_fig = tight_bbox.transformed(fig.transFigure.inverted())
            legend_left = tight_bbox_fig.x0

        if dissolved_zoomed is not None:
            dissolved_zoomed.plot(ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none");  
        else:
            dissolved.plot(ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none");  
            
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

        if var_title is None:
            var_title = self._get_title(var, "var_titles").title()
        if annotation is None:
            annotation = self._get_annotation([var])
        else:
            annotation = self._get_annotation([var]) + f"{annotation}\n"

        country = pycountry.countries.get(alpha_3=iso_code).name
        if zoom_to is not None:
            subunit = ", ".join([value for value in zoom_to.values()])
            country = f"{subunit}, {country}"
            self._plot_tiny_map(zoom_to, country, subunit, data, dissolved, fig, ax, iax, config, x=legend_left)

        if title is None:
            title = config['title'].format(var_title, country)
        self._add_titles_and_annotations(fig, ax, config, title, subtitle, annotation, x=legend_left)
        ax.axis("off")
        return ax


    def _plot_tiny_map(self, zoom_to, country, subunit, data, dissolved, fig, ax, ax2, config, x):
        # Get legend (ax2) position and main ax position in figure coords
        ax_pos = ax.get_position()       
        ax2_pos = ax2.get_position() 
    
        # Horizontal alignment (align left with legend labels)
        iax_x = x
        natural_height = ax_pos.y1 - ax2_pos.y1
        
        # Max allowed height = 1/3 of ax height
        max_height = (ax_pos.y1 - ax_pos.y0) / 3
        
        # Choose the smaller of the two
        total_gap = ax_pos.y1 - ax2_pos.y1
        iax_height = min(natural_height, max_height)
        iax_y = ax2_pos.y1 + (total_gap - iax_height) / 2
    
        # Width (space between legend and ax) 
        iax_width = ax_pos.x0 - x    
    
        # Create tiny map axes in figure coordinates
        iax = fig.add_axes([iax_x, iax_y, iax_width, iax_height])
        iax.set_axis_off()

        iax.set_axis_off()
        dissolved.plot(ax=iax, facecolor="lightgray", edgecolor="lightgray", lw=1)
        data.dissolve("iso_code").plot(ax=iax, facecolor="bisque", edgecolor="sienna", lw=0.25)
        
        xmin, ymin, xmax, ymax = data.dissolve("iso_code").total_bounds
        iax.annotate(text=subunit,
            xy=(xmin + abs(xmin - xmax)/2, ymax),
            xytext=(0, 5),  
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
        ax, 
        config, 
        title, 
        subtitle, 
        annotation, 
        x: float = None,
        subtitle_gap: float = 0.02
    ):
        y0 = ax.get_position().y0
        y1 = ax.get_position().y1  
        
        title_x, title_y = x, y1
        if 'title_x' in config:
            title_x = config['title_x']
        if 'title_y' in config:
            title_y = config['title_y']

        if title is not None:
            fig.text(
                x=title_x, 
                y=title_y, 
                s=title, 
                size=config['title_fontsize'], 
                font=bold
            )
            
        if subtitle is not None:
            subtitle_x, subtitle_y = x, y1
            if 'subtitle_x' in config:
                subtitle_x = config['subtitle_x']  

            if subtitle_y is not None:
                subtitle_y -= subtitle_gap
            elif 'subtitle_y' in config:
                subtitle_y = config['subtitle_y']
                
            fig.text(
                x=subtitle_x, 
                y=subtitle_y, 
                s=subtitle, 
                size=config['subtitle_fontsize'], 
                font=regular
            )
            
        if annotation is not None:
            annotation_x, annotation_y = x, y0
            if 'annotation_x' in config:
                annotation_x = config['annotation_x']
            elif annotation_x is None:
                annotation_x = title_x
                
            if 'annotation_y' in config:
                annotation_y = config['annotation_y']
            
            fig.text(
                x=annotation_x, 
                y=annotation_y, 
                s=annotation, 
                size=config['annotation_fontsize'], 
                color=config['annotation_color'], 
                font=regular
            )

            
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

    
    def _get_annotation(self, var_list: list = [], add_adm: bool = True):
        annotations = self.map_config["annotations"]
        annotation = "Source: \n"

        if add_adm:
            var_list += [self.dm.adm_source.lower()]

        anns = []
        for var in var_list:
            for key, ann in annotations.items():
                if key in var:
                    if ann not in anns:
                        anns.append(ann)
                        annotation += ann + "\n"
        return annotation


    def cut(self, series, var_bounds, nbins):
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


  