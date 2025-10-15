import logging
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from keplergl import KeplerGl

# Local package import
from dfcv_colocation_mapping import data_utils


class HierarchicalCheckboxes:
    def __init__(
        self,
        hierarchy,
        indent_per_level=20,
        selected_hierarchy=None,
        save_callback=None,
        save_label="Save Selection",
    ):
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
        self.widget = widgets.VBox(
            self.widgets_tree,
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
            style={
                "description_width": "initial"
            },  # ensures full label is visible
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
        self.widgets_tree.append(widgets.HBox([button]))
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


class ChoroplethWidget:
    def __init__(
        self,
        dm,
        geoplot,
        var_list: list = [],
        default_var: str = None,
        var_label: str = "Variable:",
        enable_conflict: bool = False,
        enable_exposure: bool = False,
        enable_conflict_exposure: bool = False,
        enable_hazard_exposure: bool = False,
        enable_mhs_exposure: bool = False,
    ):
        self.dm = dm
        self.geoplot = geoplot
        self.var_list = var_list
        self.default_var = default_var
        self.var_label = var_label
        self.output = widgets.Output()

        self.enable_conflict = enable_conflict
        self.enable_exposure = enable_exposure
        self.enable_conflict_exposure = enable_conflict_exposure
        self.enable_hazard_exposure = enable_hazard_exposure
        self.enable_mhs_exposure = enable_mhs_exposure

        if self.enable_conflict or self.enable_exposure:
            self.asset = widgets.Dropdown(
                options=dm.asset_names, value="worldpop", description="Asset:"
            )
            exposure_options = ["relative", "absolute"]
            if not enable_conflict_exposure:
                exposure_options.append("intensity_weighted_relative")
            self.exposure_type = widgets.Dropdown(
                options=exposure_options,
                value="relative",
                description="Exposure:",
            )

        if self.enable_conflict:
            self.conflict_data_source = widgets.Dropdown(
                options=["ACLED", "UCDP"],
                value="ACLED",
                description="Conflict Source:",
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

        if self.enable_conflict_exposure:
            self.conflict_exposure_source = widgets.Dropdown(
                options=[
                    "ACLED (WBG calculation)",
                    "ACLED (population_best)",
                    "UCDP",
                ],
                value="ACLED (WBG calculation)",
                description="Exposure DS:",
            )

        if self.enable_hazard_exposure:
            hazard_options = [
                x.replace("global_", "") for x in dm.config["hazard_data"]
            ]
            self.hazard_exposure_source = widgets.Dropdown(
                options=hazard_options,
                value="earthquake",
                description="Hazard:",
            )

        if enable_mhs_exposure:
            self.mhs_aggregation = widgets.Dropdown(
                options=["power_mean", "geometric_mean", "arithmetic_mean"],
                value="power_mean",
                description="MHS Aggregation:",
            )
            self.hazard_category = widgets.Dropdown(
                options=["all"] + list(dm.config["hazards"].keys()),
                value="all",
                description="Category:",
            )

        self.variable_dropdown = None
        if len(var_list) > 0:
            self.variable_dropdown = widgets.Dropdown(
                options=self.var_list,
                value=self.default_var,
                description=self.var_label,
            )

        self.legend_type = widgets.Dropdown(
            options=["default", "colorbar", "barplot"],
            value="default",
            description="Legend:",
        )

        # --- Region selection ---
        self.zoom_to_region = widgets.Checkbox(
            value=False, description="Zoom to region"
        )

        adm_options = ["ADM1", "ADM2"]
        if dm.group is not None:
            adm_options = [dm.group] + adm_options
        self.adm_level = widgets.Dropdown(
            options=adm_options,
            value=adm_options[0],
            description="ADM Level:",
        )

        self.adm_string = widgets.Dropdown(
            options=self._get_adm_options(adm_options[0]),
            description="Region:",
        )

        # --- Overlay points ---
        self.overlay_points = widgets.Checkbox(
            value=False, description="Overlay points"
        )

        self.points = widgets.Dropdown(
            options=["ACLED", "UCDP", "OSM"],
            value="ACLED",
            description="Points:",
        )

        self.points_column = widgets.Dropdown(
            options=[
                "disorder_type",
                "event_type",
                "type_of_violence",
                "sub_event_type",
                "osm_category",
            ],
            value="disorder_type",
            description="Points column:",
        )

        self.point_columns_by_source = {
            "ACLED": ["disorder_type", "event_type", "sub_event_type"],
            "UCDP": ["type_of_violence"],
            "OSM": ["osm_category"],
        }

        self.points.observe(self._on_points_source_change, names="value")
        self._on_points_source_change({"new": self.points.value})

        self.markerscale = widgets.FloatSlider(
            value=20,
            min=5,
            max=100,
            step=1,
            description="Marker scale:",
            continuous_update=False,
        )
        self.alpha = widgets.FloatSlider(
            value=0.7,
            min=0.1,
            max=1.0,
            step=0.05,
            description="Alpha:",
            continuous_update=False,
        )
        self.legend1_y = widgets.FloatSlider(
            value=0.30,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Legend Y:",
            continuous_update=False,
        )

        self.run_button = widgets.Button(
            description="Plot", button_style="primary", icon="map"
        )

        self.adm_level.observe(self._on_adm_level_change, names="value")
        self.run_button.on_click(self._on_plot_click)

        self._build_layout()

    def _on_points_source_change(self, change):
        """Update available columns when the points source changes."""
        source = change["new"]
        valid_columns = self.point_columns_by_source.get(source, [])
        self.points_column.options = valid_columns

        # Keep value valid if possible
        if self.points_column.value not in valid_columns:
            self.points_column.value = (
                valid_columns[0] if valid_columns else None
            )

    def _get_adm_options(self, level):
        """Return available region names for a given ADM level."""
        return sorted(list(set(self.dm.data.get(level, []))))

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

            if self.enable_conflict:
                var = (
                    f"{self.conflict_data_source.value.lower()}_"
                    f"{self.asset.value}_{self.conflict_column.value}"
                )
            if self.enable_conflict_exposure:
                conflict_source = data_utils.get_conflict_source(
                    self.conflict_exposure_source.value
                )
                exposure = data_utils.get_exposure(self.exposure_type.value)
                var = f"{conflict_source}_{self.asset.value}_{exposure}"
            if self.enable_hazard_exposure:
                exposure = data_utils.get_exposure(self.exposure_type.value)
                var = f"{self.hazard_exposure_source.value}_{self.asset.value}_{exposure}"
            if self.enable_mhs_exposure:
                self.dm.mhs_aggregation = self.mhs_aggregation.value
                self.dm.data = self.dm._calculate_multihazard_score(
                    self.dm.data, aggregation=self.dm.mhs_aggregation
                )
                exposure = data_utils.get_exposure(self.exposure_type.value)
                var = f"mhs_{self.hazard_category.value}_{self.asset.value}_{exposure}"
            if self.enable_conflict_exposure and self.enable_mhs_exposure:
                conflict_source = data_utils.get_conflict_source(
                    self.conflict_exposure_source.value
                )
                exposure = data_utils.get_exposure(self.exposure_type.value)
                var = f"mhs_{self.hazard_category.value}_{conflict_source}_{self.asset.value}_{exposure}"
            elif self.variable_dropdown:
                var = self.variable_dropdown.value

            logging.info(f"Plotting variable: {var}")

            zoom_to = None
            if self.zoom_to_region.value:
                zoom_to = {self.adm_level.value: self.adm_string.value}

            # Plot the choropleth on the axes
            ax, xpos = self.geoplot.plot_choropleth(
                var=var,
                kwargs={"legend_type": self.legend_type.value},
                zoom_to=zoom_to,
            )

            # Optional: overlay points
            if self.overlay_points.value:
                self.geoplot.plot_points(
                    self.points_column.value,
                    dataset=self.points.value.lower(),
                    kwargs={
                        "alpha": self.alpha.value,
                        "legend1_y": self.legend1_y.value,
                        "markerscale": self.markerscale.value,
                        "cmap": "tab10",
                    },
                    zoom_to=zoom_to,
                    ax=ax,
                    xpos=xpos,
                )

            # Render the figure in the notebook
            plt.show()

    def _build_layout(self):
        """Assemble widget layout."""
        zoom_box = widgets.VBox(
            [self.zoom_to_region, self.adm_level, self.adm_string]
        )
        points_box = widgets.VBox(
            [self.overlay_points, self.points, self.points_column]
        )
        style_box = widgets.VBox(
            [self.markerscale, self.alpha, self.legend1_y]
        )

        controls_list = [
            self.legend_type,
            zoom_box,
            points_box,
            style_box,
            self.run_button,
        ]
        if self.enable_conflict:
            box = widgets.HBox(
                [self.conflict_data_source, self.asset, self.conflict_column]
            )
            controls_list.insert(0, box)
        elif self.enable_exposure:
            box = [self.asset, self.exposure_type]
            if self.enable_conflict_exposure:
                box.insert(0, self.conflict_exposure_source)
            if self.enable_hazard_exposure:
                box.insert(0, self.hazard_exposure_source)
            if self.enable_mhs_exposure:
                box.append(self.mhs_aggregation)
                box.append(self.hazard_category)
            box = widgets.HBox(box)
            controls_list.insert(0, box)
        else:
            controls_list.insert(0, self.variable_dropdown)

        self.controls = widgets.VBox(controls_list)

    # -------------------
    # Public method
    # -------------------
    def show(self):
        """Display the interactive widget."""
        display(self.controls, self.output)


class BivariateChoroplethWidget:
    def __init__(self, dm, geoplot, data_utils):
        """
        Interactive widget for plotting bivariate choropleth maps
        (e.g., conflict exposure vs multi-hazard exposure).

        Parameters
        ----------
        dm : object
            Data manager providing ADM-level data and asset names.
        geoplot : module/object
            Exposes `plot_bivariate_choropleth()` and optionally `plot_points()`.
        data_utils : module/object
            Provides helper functions like `get_conflict_source()` and `get_exposure()`.
        """
        self.dm = dm
        self.geoplot = geoplot
        self.data_utils = data_utils
        self.output = widgets.Output()

        self.conflict_exposure_source = widgets.Dropdown(
            options=["ACLED (WBG calculation)", "UCDP"],
            value="ACLED (WBG calculation)",
            description="Conflict DS:",
        )
        self.conflict_exposure_type = widgets.Dropdown(
            options=["absolute", "relative"],
            value="relative",
            description="Conflict Exp:",
        )

        self.hazard_category = widgets.Dropdown(
            options=["all"] + list(dm.config["hazards"].keys()),
            value="all",
            description="Hazard Cat:",
        )
        self.hazard_exposure_type = widgets.Dropdown(
            options=["relative", "intensity_weighted_relative"],
            value="relative",
            description="Hazard Exp:",
        )

        self.asset = widgets.Dropdown(
            options=dm.asset_names,
            value="worldpop",
            description="Asset:",
        )
        self.binning = widgets.Dropdown(
            options=["equal_intervals", "quantiles"],
            value="equal_intervals",
            description="Binning:",
        )

        self.zoom_to_region = widgets.Checkbox(
            value=False, description="Zoom to region"
        )
        adm_options = ["ADM1", "ADM2"]
        if dm.group is not None:
            adm_options = [dm.group] + adm_options
        self.adm_level = widgets.Dropdown(
            options=adm_options,
            value=adm_options[0],
            description="ADM Level:",
        )
        self.adm_string = widgets.Dropdown(
            options=self._get_adm_options(adm_options[0]),
            description="Region:",
        )

        self.overlay_points = widgets.Checkbox(
            value=False, description="Overlay points"
        )
        self.points = widgets.Dropdown(
            options=["ACLED", "UCDP", "OSM"],
            value="ACLED",
            description="Points:",
        )

        self.points_column = widgets.Dropdown(
            options=[
                "disorder_type",
                "event_type",
                "type_of_violence",
                "sub_event_type",
                "osm_category",
            ],
            value="disorder_type",
            description="Points column:",
        )

        self.point_columns_by_source = {
            "ACLED": ["disorder_type", "event_type", "sub_event_type"],
            "UCDP": ["type_of_violence"],
            "OSM": ["osm_category"],
        }

        self.points.observe(self._on_points_source_change, names="value")
        self._on_points_source_change({"new": self.points.value})

        self.markerscale = widgets.FloatSlider(
            value=20,
            min=5,
            max=100,
            step=1,
            description="Marker scale:",
            continuous_update=False,
        )
        self.alpha = widgets.FloatSlider(
            value=0.7,
            min=0.1,
            max=1.0,
            step=0.05,
            description="Alpha:",
            continuous_update=False,
        )
        self.legend1_y = widgets.FloatSlider(
            value=0.30,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Legend Y:",
            continuous_update=False,
        )

        self.run_button = widgets.Button(
            description="Plot", button_style="primary", icon="map"
        )

        # Observers
        self.adm_level.observe(self._on_adm_level_change, names="value")
        self.run_button.on_click(self._on_plot_click)

        # Layout
        self._build_layout()

    def _on_points_source_change(self, change):
        """Update available columns when the points source changes."""
        source = change["new"]
        valid_columns = self.point_columns_by_source.get(source, [])
        self.points_column.options = valid_columns

        # Keep value valid if possible
        if self.points_column.value not in valid_columns:
            self.points_column.value = (
                valid_columns[0] if valid_columns else None
            )

    def _get_adm_options(self, level):
        """Return available region names for the ADM level."""
        return sorted(list(set(self.dm.data.get(level, []))))

    def _on_adm_level_change(self, change):
        level = change["new"]
        self.adm_string.options = self._get_adm_options(level)
        if self.adm_string.options:
            self.adm_string.value = self.adm_string.options[0]

    def _on_plot_click(self, _):
        with self.output:
            clear_output(wait=True)

            # Determine zoom region
            zoom_to = None
            if self.zoom_to_region.value:
                zoom_to = {self.adm_level.value: self.adm_string.value}

            # Resolve variables
            conflict_source = self.data_utils.get_conflict_source(
                self.conflict_exposure_source.value
            )
            conflict_exposure = self.data_utils.get_exposure(
                self.conflict_exposure_type.value
            )
            hazard_exposure = self.data_utils.get_exposure(
                self.hazard_exposure_type.value
            )

            var1 = f"{conflict_source}_{self.asset.value}_{conflict_exposure}"
            var2 = f"mhs_{self.hazard_category.value}_{self.asset.value}_{hazard_exposure}"

            # Plot the bivariate choropleth
            ax, xpos = self.geoplot.plot_bivariate_choropleth(
                var1=var1,
                var2=var2,
                var1_bounds=[0, 1],
                var2_bounds=[0, 1],
                binning=self.binning.value,
                kwargs={
                    "legend_fontsize": 4,
                    "edgecolor": "dimgray",
                    "linewidth": 0.2,
                },
                zoom_to=zoom_to,
            )

            # Optional: overlay points
            if self.overlay_points.value:
                self.geoplot.plot_points(
                    self.points_column.value,
                    dataset=self.points.value.lower(),
                    kwargs={
                        "alpha": self.alpha.value,
                        "legend1_y": self.legend1_y.value,
                        "markerscale": self.markerscale.value,
                        "cmap": "tab10",
                    },
                    zoom_to=zoom_to,
                    ax=ax,
                    xpos=xpos,
                )

            plt.show()

    def _build_layout(self):
        """Assemble widget layout."""
        conflict_box = widgets.HBox(
            [self.conflict_exposure_source, self.conflict_exposure_type]
        )
        hazard_box = widgets.HBox(
            [self.hazard_category, self.hazard_exposure_type]
        )
        region_box = widgets.VBox(
            [self.zoom_to_region, self.adm_level, self.adm_string]
        )
        points_box = widgets.VBox(
            [self.overlay_points, self.points, self.points_column]
        )
        style_box = widgets.VBox(
            [self.markerscale, self.alpha, self.legend1_y]
        )

        controls = widgets.VBox(
            [
                conflict_box,
                hazard_box,
                self.asset,
                self.binning,
                region_box,
                points_box,
                style_box,
                self.run_button,
            ]
        )

        display(controls)

    def show(self):
        """Public display method."""
        display(self.output)


class KeplerMapUI:
    """
    Interactive KeplerGL map widget for visualizing datasets from a DataManager.
    Datasets are added in a specific order for consistent layer stacking.
    """

    def __init__(self, dm):
        self.dm = dm
        self._init_datasets()
        self._init_widgets()
        self._connect_events()

    def _init_datasets(self):
        """Define available datasets in the correct visualization order."""
        self.dataset_order = [
            "IDMC Conflict",
            "IDMC Disaster",
            "ACLED",
            "UCDP",
            "OSM",
            "Main Data",
        ]

        self.dataset_sources = {
            "IDMC Conflict": lambda: self.dm.idmc_gidd_conflict.fillna(0),
            "IDMC Disaster": lambda: self.dm.idmc_gidd_disaster.fillna(0),
            "ACLED": lambda: self.dm.acled["worldpop"].fillna(0),
            "UCDP": lambda: self.dm.ucdp.drop(
                ["latitude", "longitude"], axis=1
            ),
            "OSM": lambda: self.dm.osm.fillna(0),
            "Main Data": lambda: self.dm.data.fillna(0),
        }

    def _init_widgets(self):
        """Initialize the interactive widgets."""
        self.dataset_selector = widgets.SelectMultiple(
            options=self.dataset_order,
            value=["Main Data"],
            description="Datasets:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px", height="180px"),
        )

        self.auto_update = widgets.Checkbox(
            value=False, description="Auto-update map on selection change"
        )

        self.render_button = widgets.Button(
            description="Render KeplerGL Map",
            button_style="success",
            icon="map",
        )

        self.output = widgets.Output()

        self.save_button = widgets.Button(
            description="💾 Save Map as HTML",
            button_style="info",
            icon="download",
        )

    def _connect_events(self):
        """Attach event listeners to widgets."""
        self.render_button.on_click(self._render_map)
        self.save_button.on_click(self._save_map)
        self.dataset_selector.observe(self._auto_render, names="value")

    def _auto_render(self, change):
        """Re-render map automatically when selection changes, if enabled."""
        if self.auto_update.value:
            self._render_map()

    def _save_map(self, _=None):
        if hasattr(self, "last_map"):
            file_name = f"{self.dm.country}_map.html"
            self.last_map.save_to_html(file_name=file_name)
            with self.output:
                print(f"✅ Map saved to {file_name}")
        else:
            with self.output:
                print("⚠️ No map to save yet. Please render first.")

    def _render_map(self, _=None):
        self.output.clear_output()
        with self.output:
            self.last_map = KeplerGl(height=800)
            for name in self.dataset_order:
                if name in self.dataset_selector.value:
                    self.last_map.add_data(
                        data=self.dataset_sources[name](),
                        name=f"{self.dm.country} {name} Data",
                    )
            display(self.last_map)

    def show(self):
        """Display the full interactive UI."""
        display(
            widgets.VBox(
                [
                    widgets.HBox(
                        [
                            self.dataset_selector,
                            widgets.VBox(
                                [
                                    self.auto_update,
                                    self.render_button,
                                    self.save_button,
                                ]
                            ),
                        ]
                    ),
                    self.output,
                ]
            )
        )
