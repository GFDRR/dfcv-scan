import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
import scienceplots
import pycountry
import folium

plt.style.use("science")
plt.style.use(["no-latex"])

def plot_choropleth_folium(
    data: gpd.GeoDataFrame,
    var: str, 
    var_title: str, 
    adm_level: str = "ADM3",
    tiles: str = 'cartodbpositron',
    zoom_start: int = 8,
    fill_opacity: float = 0.8,
    fill_color: str = "YlGn",
    line_color: str = "gray",
    meter_crs: str = "EPSG:3857"
):
    original_crs = data.crs
    x = data.dissolve("iso_code").to_crs(meter_crs).centroid.to_crs(original_crs).x.iloc[0]
    y = data.dissolve("iso_code").to_crs(meter_crs).centroid.to_crs(original_crs).y.iloc[0]

    m = folium.Map(location=[y, x], tiles=tiles, zoom_start=zoom_start)
    key_on = f"feature.properties.{adm_level}"
    folium.Choropleth(
        data=data,
        geo_data=data,
        columns=[adm_level, var],
        key_on=key_on,
        fill_opacity=fill_opacity,
        fill_color=fill_color,
        line_color=line_color
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
    folium.LayerControl().add_to(m)
    return m


def plot_choropleth(
    data: gpd.GeoDataFrame,
    iso_code: str,
    var: str, 
    var_title: str, 
    vmin: float = 0,
    vmax: float = 1,
    title: str = "{} in {}",   
    title_loc: str = "left",
    figsize_x: int = 8,
    figsize_y: int = 8,
    dpi: int = 300,
    crs: str = "EPSG:3857"
):
    data = data.to_crs(crs)
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y), dpi=dpi)
    data.plot(var, ax=ax, legend=True, vmin=vmin, vmax=vmax);

    country = pycountry.countries.get(alpha_3=iso_code).name
    title = title.format(var_title, country)
    ax.set_title(title, loc=title_loc)

    ctx.add_basemap(
        ax=ax, source=ctx.providers.CartoDB.Positron
    )
    plt.tight_layout() 
    plt.axis('off')
    