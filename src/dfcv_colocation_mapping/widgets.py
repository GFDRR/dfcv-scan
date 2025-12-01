import os
import logging
import matplotlib.pyplot as plt
from IPython.display import clear_output
from dfcv_colocation_mapping import data_utils

import ipywidgets as widgets
from IPython.display import display


class HierarchicalCheckboxes:
    def __init__(
        self,
        asset_name,
        hierarchy,
        indent_per_level=20,
        selected_hierarchy=None,
        save_callback=None,
        save_label="Save Selection",
    ):
        self.controls_container = widgets.VBox([])
        self.asset_name = asset_name
        self.hierarchy = hierarchy
        self.selected_hierarchy = selected_hierarchy or {}
        self.indent_per_level = indent_per_level
        self.save_callback = save_callback
        self.save_label = save_label

        self.category_checkboxes = {}
        self.subtype_checkboxes = {}
        self.option_checkboxes = {}
        self.updating = {"lock": False}
        self.widgets_tree = []

        self._build_ui()
        self._apply_selection(self.selected_hierarchy)
        self._add_save_button()

        title_label = widgets.HTML(
            f"<b style='font-size:16px'>{self.asset_name}</b>"
        )

        self.widget = widgets.VBox(
            [title_label] + self.widgets_tree,
            layout=widgets.Layout(margin="0", padding="0", width="auto"),
        )

    def _make_checkbox(self, description, indent=0):
        """Helper to create checkboxes with consistent styling and wrapping text."""
        return widgets.Checkbox(
            value=False,
            description=description,
            layout=widgets.Layout(
                width="auto",
                margin=f"0 0 0 {indent}px",
                padding="0",
            ),
            style={"description_width": "initial"},
        )

    def _build_ui(self):
        """Build the nested checkbox hierarchy with consistent indentation."""
        for cat, subtypes in self.hierarchy.items():
            cat_cb = self._make_checkbox(cat)
            self.category_checkboxes[cat] = cat_cb
            subtype_boxes = []

            for sub, options in subtypes.items():
                sub_cb = self._make_checkbox(sub, indent=self.indent_per_level)
                self.subtype_checkboxes.setdefault(cat, {})[sub] = sub_cb

                opt_widgets = [
                    self._make_checkbox(o, indent=2 * self.indent_per_level)
                    for o in options
                ]
                self.option_checkboxes.setdefault(cat, {})[sub] = opt_widgets

                sub_box = widgets.VBox(
                    [sub_cb] + opt_widgets,
                    layout=widgets.Layout(margin="0", padding="0"),
                )
                subtype_boxes.append(sub_box)

                sub_cb.observe(
                    lambda ch, c=cat, s=sub: self._on_subtype_change(ch, c, s),
                    names="value",
                )
                for opt in opt_widgets:
                    opt.observe(
                        lambda ch, c=cat, s=sub, o=opt: self._on_option_change(
                            ch, c, s, o
                        ),
                        names="value",
                    )

            cat_box = widgets.VBox(
                [cat_cb] + subtype_boxes,
                layout=widgets.Layout(margin="0", padding="0"),
            )
            cat_cb.observe(
                lambda ch, c=cat: self._on_category_change(ch, c),
                names="value",
            )
            self.widgets_tree.append(cat_box)

    def _add_save_button(self):
        """Add a Save button at the bottom of the widget."""
        button = widgets.Button(
            description=self.save_label,
            button_style="success",
            icon="save",
            layout=widgets.Layout(width="200px", margin="10px 0 0 0"),
        )
        output = widgets.Output()

        def on_click(b):
            with output:
                output.clear_output()
                result = self.get_selected()
                if callable(self.save_callback):
                    self.save_callback(result)
                    print("✅ Selection saved.")
                else:
                    print("ℹ️ No save callback provided.")

        button.on_click(on_click)
        self.widgets_tree.append(widgets.VBox([button]))
        self.widgets_tree.append(output)

    def _on_category_change(self, change, category):
        """When category toggled, set all subtypes and options."""
        if self.updating["lock"]:
            return
        if change["name"] == "value":
            self.updating["lock"] = True
            for sub, sub_cb in self.subtype_checkboxes[category].items():
                sub_cb.value = change["new"]
                for opt in self.option_checkboxes[category][sub]:
                    opt.value = change["new"]
            self.updating["lock"] = False

    def _on_subtype_change(self, change, category, subtype):
        """When subtype toggled, set all options under it (but not the category)."""
        if self.updating["lock"]:
            return
        if change["name"] == "value":
            self.updating["lock"] = True
            for opt in self.option_checkboxes[category][subtype]:
                opt.value = change["new"]
            self.updating["lock"] = False

    def _on_option_change(self, change, category, subtype, option):
        """Don't affect parents — just the individual checkbox."""
        if self.updating["lock"]:
            return
        return

    def _apply_selection(self, selected_hierarchy):
        """Set checkbox states based on a provided filtered hierarchy."""
        self.updating["lock"] = True
        for cat, subtypes in self.hierarchy.items():
            if cat not in selected_hierarchy:
                continue
            self.category_checkboxes[cat].value = True
            for sub, opts in subtypes.items():
                if sub not in selected_hierarchy[cat]:
                    continue
                self.subtype_checkboxes[cat][sub].value = True
                for opt in self.option_checkboxes[cat][sub]:
                    if opt.description in selected_hierarchy[cat][sub]:
                        opt.value = True

        self.updating["lock"] = False

    def get_selected(self):
        """Return a dict of selected options."""
        selected = {}
        for cat, subtypes in self.option_checkboxes.items():
            selected[cat] = {}
            for sub, opts in subtypes.items():
                sel_opts = [o.description for o in opts if o.value]
                if sel_opts:
                    selected[cat][sub] = sel_opts
        selected = {c: s for c, s in selected.items() if s}
        return selected

    def show(self):
        display(self.widget)


class MultiSelectorWidget:
    def __init__(self, dataset, data_all, data_selected, save_callback=None):
        self.dataset = dataset
        self.data_all = data_all
        self.data_selected = data_selected
        self.save_callback = save_callback

        self.data_all = (
            data_utils.get_deepest_values(self.data_all)
            if "hazard" in dataset
            else data_all
        )
        self.checkboxes = self._build_checkboxes(
            self.data_all, self.data_selected
        )
        self.section = self._build_section(
            dataset,
            self.checkboxes,
            self.data_selected,
            f"Save {dataset} selection",
        )
        self.widget = widgets.VBox([self.section])

    def _build_checkboxes(self, items, selected):
        """Create a list of checkboxes for a list of strings."""
        checkboxes = []
        for item in items:
            cb = widgets.Checkbox(
                value=item in selected,
                description=item,
                layout=widgets.Layout(width="auto"),
                style={"description_width": "initial"},
            )
            checkboxes.append(cb)
        return checkboxes

    def _build_section(self, title, checkboxes, data_selected, button_label):
        """Build a titled section with checkboxes and a save button."""
        title_label = widgets.HTML(
            f"<b style='font-size:16px'>{title.upper()}</b>"
        )

        save_button = widgets.Button(
            description=button_label,
            button_style="success",
            icon="save",
            layout=widgets.Layout(width="200px", margin="10px 0 0 0"),
        )

        output = widgets.Output()

        def on_click(b):
            with output:
                output.clear_output()
                selected = self.get_selected()
                if callable(self.save_callback):
                    self.save_callback(selected)
                    print(f"✅ {title.upper()} selection saved.")

        save_button.on_click(on_click)

        section = widgets.VBox(
            [
                title_label,
                widgets.VBox(checkboxes, layout=widgets.Layout(margin="0")),
                widgets.HBox([save_button]),
                output,
            ],
            layout=widgets.Layout(margin="0 0 10px 0"),
        )
        return section

    def get_selected(self):
        return [cb.description for cb in self.checkboxes if cb.value]

    def show(self):
        display(self.widget)


class MapWidget:
    def __init__(
        self,
        geoplot,
        map_mode: str = "choropleth",
        var_list: list = None,
        default_var: str = None,
        var_label: str = "Variable:",
        enable_conflict: bool = False,
        enable_conflict_exposure: bool = False,
        enable_hazard_exposure: bool = False,
        enable_mhs_exposure: bool = False,
        out_dir: str = "",
    ):
        self.geoplot = geoplot
        self.var_list = var_list
        self.default_var = default_var
        self.var_label = var_label
        self.output = widgets.Output()
        self.last_vars = []
        self.save_suffixes = ""
        self.out_dir = self.geoplot.dm.iso_code + "_" + out_dir

        self.enable_conflict = enable_conflict
        self.enable_conflict_exposure = enable_conflict_exposure
        self.enable_hazard_exposure = enable_hazard_exposure
        self.enable_mhs_exposure = enable_mhs_exposure
        self.map_mode = map_mode

        self.var_list = var_list or self.geoplot.dm.data.columns
        self.all_var_list = self.geoplot.dm.data.columns

        self.map_title = widgets.Text(
            value=None,
            description="Title:",  # Label for the text box
        )
        self.map_subtitle = widgets.Text(
            value=None,
            description="Subtitle:",  # Label for the text box
        )

        # Choropleth
        self.variable_dropdown = widgets.Dropdown(
            options=self.var_list,
            value=self.default_var,
            description=self.var_label,
        )
        self.all_variable_dropdown = widgets.Dropdown(
            options=self.all_var_list,
            value=self.default_var,
            description=self.var_label,
        )

        self.legend_type = widgets.Dropdown(
            options=["default", "colorbar", "barplot"],
            value="colorbar",
            description="Legend:",
        )

        self.binning = widgets.Dropdown(
            options=["equal_intervals", "quantiles"],
            value="equal_intervals",
            description="Binning:",
        )

        self.var_bounds_selector = widgets.Dropdown(
            options=["[0, 1]", "[min, max]"],
            value="[min, max]",
            description="Bounds:",
        )

        self.asset = widgets.Dropdown(
            options=geoplot.dm.asset_names,
            value="worldpop",
            description="Asset:",
        )

        hazard_exposure_options = self.geoplot.dm.config["suffixes"]
        self.hazard_exposure_type = widgets.Dropdown(
            options=hazard_exposure_options,
            value="exposure_relative",
            description="Hazard exposure:",
        )

        conflict_exposure_options = hazard_exposure_options[:-1]
        self.conflict_exposure_type = widgets.Dropdown(
            options=conflict_exposure_options,
            value="exposure_relative",
            description="Conflict exposure:",
        )
        self.conflict_data_source = widgets.Dropdown(
            options=["acled", "ucdp"],
            value="acled",
            description="Conflict data:",
        )
        self.conflict_column = widgets.Dropdown(
            options=[
                "conflict_count",
                "fatalities",
                "fatalities_per_conflict",
            ],
            value="conflict_count",
            description="Column:",
        )
        conflict_exposure_sources = self.geoplot.dm.config["conflict_columns"]
        self.conflict_exposure_source = widgets.Dropdown(
            options=conflict_exposure_sources,
            value=conflict_exposure_sources[0],
            description="Conflict data:",
        )

        self.hazard_exposure_source = widgets.Dropdown(
            options=geoplot.dm.hazard_names,
            value=geoplot.dm.hazard_names[0],
            description="Hazard:",
        )

        self.mhs_aggregation = widgets.Dropdown(
            options=[
                "arithmetic_mean",
                "power_mean",
                "geometric_mean",
            ],
            value="arithmetic_mean",
            description="MHS aggregation:",
        )
        self.hazard_category = widgets.Dropdown(
            options=["all"] + list(geoplot.dm.config["hazards_all"].keys()),
            value="all",
            description="MHS category:",
        )

        self.zoom_to_region = widgets.Checkbox(
            value=False,
            description="Zoom to region",
        )

        adm_options = [col for col in geoplot.dm.data.columns if "ADM" in col]
        self.adm_level = widgets.Dropdown(
            options=adm_options,
            value=adm_options[0],
            description="ADM Level:",
        )

        self.adm_string = widgets.Dropdown(
            options=self._get_adm_options(adm_options[0]),
            description="Region:",
        )

        self.overlay_conflict_points = widgets.Checkbox(
            value=False,
            description="Overlay conflict points",
        )
        self.conflict_points = widgets.Dropdown(
            options=["ACLED", "UCDP"],
            value="ACLED",
            description="Conflict data:",
        )
        self.conflict_points_column = widgets.Dropdown(
            options=[
                None,
                "disorder_type",
                "event_type",
                "type_of_violence",
                "sub_event_type",
            ],
            value="disorder_type",
            description="Category:",
        )

        self.conflict_point_columns_by_source = {
            "ACLED": ["disorder_type", "event_type", "sub_event_type"],
            "UCDP": ["type_of_violence"],
        }

        self.conflict_points.observe(
            self._on_conflict_points_source_change, names="value"
        )
        self._on_conflict_points_source_change(
            {"new": self.conflict_points.value}
        )

        self.conflict_markerscale = widgets.FloatSlider(
            value=10,
            min=1,
            max=500,
            step=1,
            description="Marker size:",
            continuous_update=False,
        )
        self.conflict_alpha = widgets.FloatSlider(
            value=0.7,
            min=0.1,
            max=1.0,
            step=0.05,
            description="Transparency:",
            continuous_update=False,
        )
        self.conflict_legend1_y = widgets.FloatSlider(
            value=0.30,
            min=0.0,
            max=1.0,
            step=0.025,
            description="Legend 1 Y:",
            continuous_update=False,
        )
        self.conflict_legend2_y = widgets.FloatSlider(
            value=0.20,
            min=0.0,
            max=1.0,
            step=0.025,
            description="Legend 2 Y:",
            continuous_update=False,
        )

        self.overlay_idp_points = widgets.Checkbox(
            value=False,
            description="Overlay IDP data",
        )
        self.idp_points = widgets.Dropdown(
            options=[
                "idmc_gidd_combined",
                "idmc_gidd_disaster",
                "idmc_gidd_conflict",
            ],
            value="idmc_gidd_combined",
            description="IDP data:",
        )

        self.idp_points_column = widgets.Dropdown(
            options=[
                None,
                "Event cause",
                "Hazard category",
                "Hazard sub category",
                "Hazard type",
                "Hazard sub type",
                "Violence type",
            ],
            value="Event cause",
            description="Category:",
        )

        self.idp_point_columns_by_source = {
            "idmc_gidd_combined": ["Event cause"],
            "idmc_gidd_disaster": [
                "Hazard category",
                "Hazard sub category",
                "Hazard type",
                "Hazard sub type",
            ],
            "idmc_gidd_conflict": ["Violence type"],
        }

        self.idp_points.observe(
            self._on_idp_points_source_change, names="value"
        )
        self._on_idp_points_source_change({"new": self.idp_points.value})

        self.idp_markerscale = widgets.FloatSlider(
            value=10,
            min=1,
            max=500,
            step=1,
            description="Marker size:",
            continuous_update=False,
        )
        self.idp_alpha = widgets.FloatSlider(
            value=0.7,
            min=0.1,
            max=1.0,
            step=0.05,
            description="Transparency:",
            continuous_update=False,
        )
        self.idp_legend1_y = widgets.FloatSlider(
            value=0.30,
            min=0.0,
            max=1.0,
            step=0.025,
            description="Legend 1 Y:",
            continuous_update=False,
        )
        self.idp_legend2_y = widgets.FloatSlider(
            value=0.20,
            min=0.0,
            max=1.0,
            step=0.025,
            description="Legend 2 Y:",
            continuous_update=False,
        )

        if len(self.geoplot.dm.config["osm_selected"]) > 0:
            self.overlay_osm_points = widgets.Checkbox(
                value=False, description="Overlay OSM points"
            )
            osm_pois = [x for x in geoplot.dm.osm_pois]
            self.osm_poi_selector = widgets.SelectMultiple(
                options=osm_pois,
                value=[
                    osm_pois[0],
                ],
                description="OSM Points:",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="300px", height="90px"),
            )

            self.overlay_osm_networks = widgets.Checkbox(
                value=False, description="Overlay OSM networks"
            )

            self.osm_pois_markerscale = widgets.FloatSlider(
                value=5,
                min=1,
                max=100,
                step=1,
                description="Marker size:",
                continuous_update=False,
            )
            self.osm_pois_alpha = widgets.FloatSlider(
                value=0.6,
                min=0.1,
                max=1.0,
                step=0.05,
                description="Transparency:",
                continuous_update=False,
            )
            self.osm_pois_legend1_y = widgets.FloatSlider(
                value=0.30,
                min=0.0,
                max=1.0,
                step=0.025,
                description="Legend Y:",
                continuous_update=False,
            )

            osm_networks = [x for x in geoplot.dm.osm_networks]
            self.osm_network_selector = widgets.SelectMultiple(
                options=osm_networks,
                value=[
                    osm_networks[0],
                ],
                description="OSM Networks:",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="300px", height="90px"),
            )
            self.osm_networks_alpha = widgets.FloatSlider(
                value=0.6,
                min=0.1,
                max=1.0,
                step=0.05,
                description="Transparency:",
                continuous_update=False,
            )
            self.osm_networks_legend1_y = widgets.FloatSlider(
                value=0.20,
                min=0.0,
                max=1.0,
                step=0.025,
                description="Legend Y:",
                continuous_update=False,
            )

        self.overlay_hatches = widgets.Checkbox(
            value=False, description="Overlay top n by column"
        )
        self.hatches_title = widgets.Text(
            value="Top 5 most vulnerable townships",  # Initial value
            description="Enter title:",  # Label for the text box
        )
        self.hatches_legend_x = widgets.FloatSlider(
            value=0.10,
            min=0.0,
            max=1.0,
            step=0.025,
            description="Legend X:",
            continuous_update=False,
        )
        self.hatches_legend_y = widgets.FloatSlider(
            value=0.20,
            min=0.0,
            max=1.0,
            step=0.025,
            description="Legend Y:",
            continuous_update=False,
        )

        self.run_button = widgets.Button(
            description="Plot", button_style="primary", icon="map"
        )

        self.adm_level.observe(self._on_adm_level_change, names="value")
        self.run_button.on_click(self._on_plot_click)

        self.save_button = widgets.Button(
            description="Save", button_style="success", icon="save"
        )
        self.save_button.on_click(self._on_save_click)

        self._build_layout()

    def _on_idp_points_source_change(self, change):
        """Update available columns when the points source changes."""
        source = change["new"]
        valid_columns = self.idp_point_columns_by_source.get(source, [])
        self.idp_points_column.options = valid_columns

        if self.idp_points_column.value not in valid_columns:
            self.idp_points_column.value = (
                valid_columns[0] if valid_columns else None
            )

    def _on_conflict_points_source_change(self, change):
        """Update available columns when the points source changes."""
        source = change["new"]
        valid_columns = self.conflict_point_columns_by_source.get(source, [])
        self.conflict_points_column.options = valid_columns

        if self.conflict_points_column.value not in valid_columns:
            self.conflict_points_column.value = (
                valid_columns[0] if valid_columns else None
            )

    def _get_adm_options(self, level):
        """Return available region names for a given ADM level."""
        return sorted(list(set(self.geoplot.dm.data.get(level, []))))

    def _on_adm_level_change(self, change):
        """Update region dropdown when ADM level changes."""
        level = change["new"]
        self.adm_string.options = self._get_adm_options(level)
        if self.adm_string.options:
            self.adm_string.value = self.adm_string.options[0]

    def _on_plot_click(self, _):
        """Handle Plot button click."""
        with self.output:
            clear_output(wait=True)
            existing_figs = set(plt.get_fignums())
            self.save_suffixes = ""

            zoom_to = None
            if self.zoom_to_region.value:
                zoom_to = {self.adm_level.value: self.adm_string.value}

            title = (
                self.map_title.value if len(self.map_title.value) > 0 else None
            )
            subtitle = (
                self.map_subtitle.value
                if len(self.map_subtitle.value) > 0
                else None
            )

            var, ax, xpos = None, None, None
            zorder = 1
            if self.map_mode == "choropleth":
                if self.enable_conflict:
                    if self.conflict_data_source.value.lower() == "acled":
                        var = (
                            f"{self.conflict_data_source.value.lower()}_"
                            f"{self.asset.value}_{self.conflict_column.value}"
                        )
                    else:
                        var = (
                            f"{self.conflict_data_source.value.lower()}_"
                            f"total_{self.conflict_column.value}"
                        )
                if self.enable_conflict_exposure:
                    var = f"{self.conflict_exposure_source.value}_{self.asset.value}_{self.conflict_exposure_type.value}"

                if self.enable_hazard_exposure:
                    var = f"{self.hazard_exposure_source.value}_{self.asset.value}_{self.hazard_exposure_type.value}"

                if self.enable_mhs_exposure:
                    self.geoplot.dm.mhs_aggregation = (
                        self.mhs_aggregation.value
                    )
                    self.geoplot.dm.data = (
                        self.geoplot.dm._calculate_multihazard_score(
                            self.geoplot.dm.data,
                            aggregation=self.mhs_aggregation.value,
                        )
                    )
                    var = f"mhs_{self.hazard_category.value}_{self.asset.value}_{self.hazard_exposure_type.value}"

                if self.enable_conflict_exposure and self.enable_mhs_exposure:
                    var = f"mhs_{self.hazard_category.value}_{self.conflict_exposure_source.value}_{self.asset.value}_{self.hazard_exposure_type.value}"

                if self.variable_dropdown.value is not None:
                    var = self.variable_dropdown.value

                if var is not None:
                    logging.info(f"Plotting variable: {var}")
                    self.last_vars = [var]

                    var_bounds = [None, None]
                    if self.var_bounds_selector.value == "[0, 1]":
                        var_bounds = [0, 1]

                    # Plot the choropleth on the axes
                    ax, xpos = self.geoplot.plot_choropleth(
                        var=var,
                        kwargs={"legend_type": self.legend_type.value},
                        zoom_to=zoom_to,
                        var_bounds=var_bounds,
                        title=title,
                        subtitle=subtitle,
                        binning=self.binning.value,
                        zorder=zorder,
                    )
                    zorder += 1

            else:
                if self.enable_conflict:
                    if self.conflict_data_source.value.lower() == "acled":
                        var1 = (
                            f"{self.conflict_data_source.value.lower()}_"
                            f"{self.asset.value}_{self.conflict_column.value}"
                        )
                    else:
                        var1 = (
                            f"{self.conflict_data_source.value.lower()}_"
                            f"total_{self.conflict_column.value}"
                        )
                elif self.enable_conflict_exposure:
                    var1 = f"{self.conflict_exposure_source.value}_{self.asset.value}_{self.conflict_exposure_type.value}"

                if self.enable_hazard_exposure or self.enable_mhs_exposure:
                    if self.enable_hazard_exposure:
                        var2 = f"{self.hazard_exposure_source.value}_{self.asset.value}_{self.hazard_exposure_type.value}"

                    elif self.enable_mhs_exposure:
                        self.geoplot.dm.data = (
                            self.geoplot.dm._calculate_multihazard_score(
                                self.geoplot.dm.data,
                                aggregation=self.mhs_aggregation.value,
                            )
                        )
                        var2 = f"mhs_{self.hazard_category.value}_{self.asset.value}_{self.hazard_exposure_type.value}"

                logging.info(f"Plotting variable 1: {var1}")
                logging.info(f"Plotting variable 2: {var2}")
                self.last_vars = [var2, var1]

                # Plot the bivariate choropleth
                ax, xpos = self.geoplot.plot_bivariate_choropleth(
                    var1=var1,
                    var2=var2,
                    var1_bounds=[0, 1],
                    var2_bounds=[0, 1],
                    title=title,
                    subtitle=subtitle,
                    binning=self.binning.value,
                    zoom_to=zoom_to,
                    zorder=zorder,
                )
                zorder += 1

            if self.overlay_conflict_points.value:
                ax, xpos = self.geoplot.plot_points(
                    self.conflict_points_column.value,
                    asset=self.asset.value,
                    dataset=self.conflict_points.value.lower(),
                    zoom_to=zoom_to,
                    title=title,
                    subtitle=subtitle,
                    ax=ax,
                    xpos=xpos,
                    zorder=zorder,
                    kwargs={
                        "alpha": self.conflict_alpha.value,
                        "legend1_y": self.conflict_legend1_y.value,
                        "legend2_y": self.conflict_legend2_y.value,
                        "markerscale": self.conflict_markerscale.value,
                    },
                )
                self.last_vars.append(self.conflict_points_column.value)
                zorder += 1

            if self.overlay_idp_points.value:
                self.save_suffixes += "-displacement"
                ax, xpos = self.geoplot.plot_points(
                    self.idp_points_column.value,
                    dataset=self.idp_points.value.lower(),
                    value_col="total_idps",
                    title=title,
                    subtitle=subtitle,
                    zoom_to=zoom_to,
                    ax=ax,
                    xpos=xpos,
                    zorder=zorder,
                    kwargs={
                        "alpha": self.idp_alpha.value,
                        "legend1_y": self.idp_legend1_y.value,
                        "legend2_y": self.idp_legend2_y.value,
                        "markerscale": self.idp_markerscale.value,
                    },
                )
                self.last_vars.append(self.idp_points_column.value)
                zorder += 1

            if len(self.geoplot.dm.config["osm_selected"]) > 0:
                if self.overlay_osm_networks.value:
                    ax, xpos = self.geoplot.plot_lines(
                        "tag",
                        dataset="osm",
                        osm_tags=self.osm_network_selector.value,
                        zoom_to=zoom_to,
                        ax=ax,
                        xpos=xpos,
                        zorder=zorder,
                        kwargs={
                            "alpha": self.osm_networks_alpha.value,
                            "legend_y": self.osm_networks_legend1_y.value,
                        },
                    )
                    self.last_vars.extend(self.osm_network_selector.value)
                    zorder += 1

                if self.overlay_osm_points.value:
                    ax, xpos = self.geoplot.plot_points(
                        "tag",
                        dataset="osm",
                        title=title,
                        subtitle=subtitle,
                        osm_tags=self.osm_poi_selector.value,
                        zoom_to=zoom_to,
                        ax=ax,
                        xpos=xpos,
                        zorder=zorder,
                        kwargs={
                            "alpha": self.osm_pois_alpha.value,
                            "legend1_y": self.osm_pois_legend1_y.value,
                            "markerscale": self.osm_pois_markerscale.value,
                        },
                    )
                    self.last_vars.extend(self.osm_poi_selector.value)
                    zorder += 1

            if self.overlay_hatches.value:
                ax, xpos = self.geoplot.plot_hatches(
                    adm_level=self.geoplot.dm.adm_level,
                    column=self.variable_dropdown.value,
                    zoom_to=zoom_to,
                    ax=ax,
                    xpos=xpos,
                    zorder=zorder,
                    title=self.hatches_title.value,
                    kwargs={
                        "legend_x": self.hatches_legend_x.value,
                        "legend_y": self.hatches_legend_y.value,
                    },
                )
                self.last_vars.append(self.variable_dropdown.value)
                zorder += 1

            # Render the figure in the notebook
            new_figs = set(plt.get_fignums()) - existing_figs
            if new_figs:
                self.last_fig = plt.figure(list(new_figs)[-1])
            else:
                self.last_fig = plt.gcf()
            plt.show()

    def _on_save_click(self, _, base_folder: str = "outputs"):
        """Save last plotted data subset by ADM and plotted variables."""
        with self.output:
            if not self.last_vars:
                print(
                    "⚠️ No variables have been plotted yet. Please plot the map first."
                )
                return

            data = self.geoplot.dm.data.copy()
            adm_cols = [
                col
                for col in self.geoplot.dm.geoboundary.columns
                if col != "geometry"
            ]

            region_name = self.geoplot.dm.iso_code
            if self.zoom_to_region.value is True:
                adm_col = self.adm_level.value
                region_name = self.adm_string.value

                # Filter by ADM region if column exists
                if adm_col in data.columns:
                    data = data[data[adm_col] == region_name]
                    if adm_col not in adm_cols:
                        adm_cols.append(adm_col)

            # Keep only ADM columns + last plotted variables
            cols = [col for col in adm_cols if col in data.columns]
            for var in self.last_vars:
                if var in data.columns and var not in cols:
                    cols.append(var)

            subset = data[cols].copy()

            # Create folder structure
            safe_region = region_name.replace(" ", "_")
            safe_vars = "-".join([v.replace(" ", "_") for v in self.last_vars])
            filename_base = f"{safe_region}-{safe_vars}"
            sub_folder = os.path.join(
                base_folder, self.geoplot.dm.iso_code, self.out_dir
            )
            os.makedirs(sub_folder, exist_ok=True)

            # Save CSV (no geometry)
            csv_dir = os.path.join(sub_folder, "csv")
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, f"{filename_base}.csv")
            subset.to_csv(csv_path, index=False)

            print(f"✅ Data subset saved to: {csv_path}")
            print(f"   Variables: {', '.join(self.last_vars)}")

            if self.last_fig is not None:
                img_path = os.path.join(sub_folder, f"{filename_base}.png")
                self.last_fig.savefig(img_path, dpi=300, bbox_inches="tight")
                print(f"🗺️ Plot saved to: {img_path}")

            if self.map_mode == "choropleth":
                fmap = self.geoplot.plot_folium(
                    adm_level=self.geoplot.dm.adm_level,
                    var=self.last_vars[0],
                    data=data,
                )
                html_dir = os.path.join(sub_folder, "html")
                os.makedirs(html_dir, exist_ok=True)
                html_path = os.path.join(html_dir, f"{filename_base}.html")
                fmap.save(html_path)
                print(f"🗺️ HTML saved to: {html_path}")

            display(subset.head())

    def _build_layout(self):
        """Assemble widget layout."""

        titles = widgets.VBox(
            [
                self.map_title,
                self.map_subtitle,
            ]
        )
        controls = [widgets.HTML("<hr style='margin:10px 0'>"), titles]

        zoom_box = widgets.VBox(
            [self.zoom_to_region, self.adm_level, self.adm_string]
        )
        controls.extend([widgets.HTML("<hr style='margin:10px 0'>"), zoom_box])

        conflict_points_box = widgets.VBox(
            [
                self.overlay_conflict_points,
                self.conflict_points,
                self.conflict_points_column,
            ]
        )
        conflict_style_box = widgets.VBox(
            [
                self.conflict_markerscale,
                self.conflict_alpha,
                self.conflict_legend1_y,
                self.conflict_legend2_y,
            ]
        )
        controls.extend(
            [
                widgets.HTML("<hr style='margin:10px 0'>"),
                widgets.HBox(
                    [
                        conflict_points_box,
                        conflict_style_box,
                    ]
                ),
            ]
        )

        idp_points_box = widgets.VBox(
            [
                self.overlay_idp_points,
                self.idp_points,
                self.idp_points_column,
            ]
        )
        idp_style_box = widgets.VBox(
            [
                self.idp_markerscale,
                self.idp_alpha,
                self.idp_legend1_y,
                self.idp_legend2_y,
            ]
        )
        controls.extend(
            [
                widgets.HTML("<hr style='margin:10px 0'>"),
                widgets.HBox(
                    [
                        idp_points_box,
                        idp_style_box,
                    ]
                ),
            ]
        )

        if len(self.geoplot.dm.config["osm_selected"]) > 0:
            osm_pois_box = widgets.VBox(
                [self.overlay_osm_points, self.osm_poi_selector]
            )
            osm_pois_style_box = widgets.VBox(
                [
                    self.osm_pois_markerscale,
                    self.osm_pois_alpha,
                    self.osm_pois_legend1_y,
                ]
            )
            osm_networks_box = widgets.VBox(
                [self.overlay_osm_networks, self.osm_network_selector]
            )
            osm_networks_style_box = widgets.VBox(
                [self.osm_networks_alpha, self.osm_networks_legend1_y]
            )
            controls.extend(
                [
                    widgets.HTML("<hr style='margin:10px 0'>"),
                    widgets.HBox(
                        [
                            osm_pois_box,
                            osm_pois_style_box,
                        ]
                    ),
                    widgets.HTML("<hr style='margin:10px 0'>"),
                    widgets.HBox(
                        [
                            osm_networks_box,
                            osm_networks_style_box,
                        ]
                    ),
                ]
            )

        hatches_box = widgets.VBox(
            [
                self.overlay_hatches,
                self.all_variable_dropdown,
                self.hatches_title,
            ]
        )
        hatches_style_box = widgets.VBox(
            [
                self.hatches_legend_x,
                self.hatches_legend_y,
            ]
        )
        controls.extend(
            [
                widgets.HTML("<hr style='margin:10px 0'>"),
                widgets.HBox([hatches_box, hatches_style_box]),
            ]
        )

        controls.extend(
            [
                widgets.HTML("<hr style='margin:10px 0'>"),
                self.run_button,
                self.save_button,
            ]
        )

        if self.map_mode == "choropleth":
            controls = [
                widgets.HBox(
                    [
                        self.legend_type,
                        self.binning,
                        self.var_bounds_selector,
                    ]
                )
            ] + controls

            if self.enable_conflict:
                box = widgets.HBox(
                    [
                        self.conflict_data_source,
                        self.asset,
                        self.conflict_column,
                    ]
                )
                controls.insert(0, box)

            elif (
                self.enable_conflict_exposure
                or self.enable_hazard_exposure
                or self.enable_mhs_exposure
            ):
                controls.insert(0, self.asset)
                if self.enable_hazard_exposure:
                    box = [self.hazard_exposure_source]
                    box.append(self.hazard_exposure_type)
                    box = widgets.HBox(box)
                    controls.insert(1, box)
                if self.enable_mhs_exposure:
                    box = [self.hazard_category]
                    box.append(self.hazard_exposure_type)
                    box.append(self.mhs_aggregation)
                    box = widgets.HBox(box)
                    controls.insert(1, box)
                if self.enable_conflict_exposure:
                    box = [self.conflict_exposure_source]
                    box.append(self.conflict_exposure_type)
                    box = widgets.HBox(box)
                    controls.insert(1, box)
            else:
                controls.insert(0, self.variable_dropdown)

        else:
            conflict_box = widgets.HBox(
                [self.conflict_exposure_source, self.conflict_exposure_type]
            )
            if self.enable_hazard_exposure:
                hazard_box = widgets.HBox(
                    [self.hazard_exposure_source, self.hazard_exposure_type]
                )
            elif self.enable_mhs_exposure:
                hazard_box = widgets.HBox(
                    [
                        self.hazard_category,
                        self.hazard_exposure_type,
                        self.mhs_aggregation,
                    ]
                )

            controls = [
                conflict_box,
                hazard_box,
                self.asset,
                self.binning,
            ] + controls

        self.controls = widgets.VBox(controls)

    def show(self):
        """Display the interactive widget."""
        display(self.controls, self.output)
