import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
import pyproj
import pycountry
import folium
import geojson_rewind
import json

import pypalettes
import pyfonts

def plot_folium(
    data: gpd.GeoDataFrame,
    acled: gpd.GeoDataFrame,
    var: str, 
    var_title: str, 
    adm_level: str = "ADM3",
    tiles: str = 'cartodbpositron',
    zoom_start: int = 8,
    fill_opacity: float = 0.8,
    fill_color: str = "YlGn",
    line_color: str = "gray",
    acled_group_name: str = None,
    meter_crs: str = "EPSG:3857"
):
    original_crs = data.crs
    centroid = data.dissolve("iso_code").to_crs(meter_crs).centroid
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS(meter_crs), pyproj.CRS(original_crs), always_xy=True
    )
    x, y = transformer.transform(centroid.x.iloc[0], centroid.y.iloc[0])

    m = folium.Map(location=[y, x], tiles=tiles, zoom_start=zoom_start)
    key_on = f"feature.properties.{adm_level}_ID"
    folium.Choropleth(
        data=data,
        geo_data=data.to_json(),
        columns=[f"{adm_level}_ID", var],
        key_on=key_on,
        fill_opacity=fill_opacity,
        fill_color=fill_color,
        line_color=line_color,
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


def plot_choropleth(
    data: gpd.GeoDataFrame,
    var: str, 
    var_title: str, 
    figsize_x: int,
    figsize_y: int,
    dpi: int,
    crs: str,
    title: str,   
    title_x: float,
    title_y: float,
    title_fontsize: int, 
    subtitle_x: float,
    subtitle_y: float,
    subtitle_fontsize: int, 
    edgecolor: str,
    lw: float,
    font_type: str,
    legend: bool,
    vmin: float = 0,
    vmax: float = 1,
    create_cmap: bool = False,
    cmap_type: str = "continuous",
    palette_name: str = "YlGnBu",
    colormap: list = [],
    subtitle: str = None, 
    annotation: str = None,
):
    regular = pyfonts.load_google_font(font_type)
    bold = pyfonts.load_google_font(font_type, weight="bold")
    
    data = data.to_crs(crs)
    iso_code = data.iso_code.values[0]
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y),  dpi=dpi)

    if create_cmap:
        cmap = pypalettes.create_cmap(colors=colormap, cmap_type=cmap_type)
    else:
        cmap = pypalettes.load_cmap(palette_name, cmap_type=cmap_type)
        
    data.plot(var, ax=ax, legend=legend, cmap=cmap, edgecolor=edgecolor, lw=lw, vmin=vmin, vmax=vmax);
    data.dissolve("iso_code").plot(ax=ax, lw=0.5, edgecolor="dimgrey", facecolor="none");
    country = pycountry.countries.get(alpha_3=iso_code).name

    if title is not None:
        title = title.format(var_title, country)
        fig.text(x=title_x, y=title_y, s=title, size=title_fontsize, font=bold)
    if subtitle is not None:
        fig.text(x=subtitle_x, y=subtitle_y, s=subtitle, size=subtitle_fontsize, font=regular)
    if annotation is not None:
        fig.text(x=0.15, y=0.13, s=annotation, size=6, color="#909090", font=regular)

    ax.axis("off")
    