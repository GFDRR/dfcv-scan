import os
import logging
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output


class BaseSelector:
    """Base class providing common UI helpers for checkbox-based widgets."""

    def _make_checkbox(self, description, value=False, indent=0):
        """
        Create a styled checkbox widget.

        Args:
            description (str): Text label displayed next to the checkbox.
            value (bool, optional): Initial checked state. Defaults to False.
            indent (int, optional): Left margin in pixels used to visually
                indent the checkbox. Defaults to 0.

        Returns:
            widgets.Checkbox: An ipywidgets Checkbox instance with
            predefined layout and styling.
        """

        return widgets.Checkbox(
            value=value,
            description=description,
            layout=widgets.Layout(width="auto", margin=f"0 0 0 {indent}px"),
            style={"description_width": "initial"},
        )

    def _add_save_button(self, label, callback):
        """
        Create a save button that triggers a callback with the current selection.

        Args:
            label (str): Text displayed on the save button.
            callback (callable): Function to be called with the current
                selection. If not callable, a message is shown instead.

        Returns:
            widgets.VBox: A vertical container holding the save button
            and an output area for status messages.
        """

        button = widgets.Button(
            description=label,
            button_style="success",
            icon="save",
            layout=widgets.Layout(width="200px", margin="10px 0 0 0"),
        )
        output = widgets.Output()

        def on_click(b):
            """Handle save button click events."""
            with output:
                output.clear_output()
                selection = self.get_selection()
                if callable(callback):
                    callback(selection)
                    print("✅ Selection saved.")
                else:
                    print("ℹ️ No save callback provided.")

        button.on_click(on_click)

        return widgets.VBox([button, output])


class HierarchicalSelector(BaseSelector):
    """Widget for displaying and managing hierarchical checkbox selections."""

    def __init__(
        self,
        asset_category,
        hierarchy,
        selected=None,
        save_callback=None,
        indent_per_level=20,
        save_label="Save Selection",
    ):
        """
        Initialize the hierarchical checkbox widget.

        Args:
            asset_category (str): High-level label displayed as the widget title.
            hierarchy (dict): Nested dictionary defining the checkbox hierarchy
                in the form {category: {subcat: [options]}}.
            indent_per_level (int, optional): Pixel indentation per hierarchy
                level. Defaults to 20.
            selected_hierarchy (dict, optional): Preselected values structured
                like the hierarchy input. Defaults to None.
            save_callback (callable, optional): Function called with the current
                selection when the save button is clicked.
            save_label (str, optional): Label for the save button.
        """

        self.asset_category = asset_category
        self.hierarchy = hierarchy
        self.selected = selected or {}
        self.indent_per_level = indent_per_level
        self.save_callback = save_callback
        self.save_label = save_label

        # Store checkbox references for state management
        self.cat_checkboxes = {}
        self.subcat_checkboxes = {}
        self.option_checkboxes = {}

        # Prevent recursive updates when syncing checkbox states
        self._lock = False

        # Build UI and apply initial selection
        body = self._build_ui()
        self._apply_selection(self.selected)
        save_box = self._add_save_button(save_label, save_callback)

        # Widget title
        title = widgets.HTML(
            f"<b style='font-size:16px'>{asset_category.title()}</b>"
        )
        self.widget = widgets.VBox([title, body, save_box])

    def _build_ui(self):
        """
        Construct the hierarchical checkbox UI.

        Returns:
            widgets.VBox: Container holding all category, subcat,
            and option checkboxes.
        """

        rows = []

        for cat, subcats in self.hierarchy.items():
            # Top-level category checkbox
            cat_checkbox = self._make_checkbox(cat)
            self.cat_checkboxes[cat] = cat_checkbox
            self.subcat_checkboxes.setdefault(cat, {})
            self.option_checkboxes.setdefault(cat, {})

            subcat_boxes = []
            for subcat, options in subcats.items():
                # subcat checkbox
                subcat_checkbox = self._make_checkbox(
                    subcat, indent=self.indent_per_level
                )
                self.subcat_checkboxes[cat][subcat] = subcat_checkbox

                # Option-level checkboxes
                option_checkboxes = [
                    self._make_checkbox(
                        option, indent=2 * self.indent_per_level
                    )
                    for option in options
                ]
                self.option_checkboxes[cat][subcat] = option_checkboxes

                # Attach subcat and option callbacks
                subcat_checkbox.observe(
                    lambda change, cat=cat, subcat=subcat: self._on_subcategory_change(
                        change, cat, subcat
                    ),
                    names="value",
                )
                for checkbox in option_checkboxes:
                    checkbox.observe(
                        lambda change, cat=cat, subcat=subcat, checkbox=checkbox: self._on_option_change(
                            change, cat, subcat, checkbox
                        ),
                        names="value",
                    )

                subcat_boxes.append(
                    widgets.VBox([subcat_checkbox] + option_checkboxes)
                )

            # Attach category-level callback
            cat_checkbox.observe(
                lambda change, cat=cat: self._on_category_change(change, cat),
                names="value",
            )

            rows.append(widgets.VBox([cat_checkbox] + subcat_boxes))

        return widgets.VBox(rows)

    def _apply_selection(self, selection):
        """
        Apply a preselected hierarchy to the widget.

        Args:
            selection (dict): Nested dictionary matching the hierarchy
                structure, specifying which options should be checked.
        """
        # Disable callbacks during bulk updates
        self._lock = True

        for cat, subcats in selection.items():
            if cat in self.cat_checkboxes:
                self.cat_checkboxes[cat].value = True

            for subcat, options in subcats.items():
                if subcat in self.subcat_checkboxes.get(cat, {}):
                    self.subcat_checkboxes[cat][subcat].value = True

                for option_checkbox in self.option_checkboxes.get(cat, {}).get(
                    subcat, []
                ):
                    if option_checkbox.description in options:
                        option_checkbox.value = True

        self._lock = False

    def _on_category_change(self, change, cat):
        """
        Handle category checkbox changes.

        Selecting or deselecting a category propagates the state
        to all its subcats and options.
        """

        if self._lock or change["name"] != "value":
            return

        self._lock = True

        new = change["new"]
        for subcat, subcat_checkbox in self.subcat_checkboxes[cat].items():
            subcat_checkbox.value = new
            for option in self.option_checkboxes[cat][subcat]:
                option.value = new

        self._lock = False

    def _on_subcategory_change(self, change, cat, subcat):
        """
        Handle subcategory checkbox changes.

        Selecting or deselecting a subcat propagates the state
        to all its options.
        """

        if self._lock or change["name"] != "value":
            return

        self._lock = True

        new = change["new"]
        for opt in self.option_checkboxes[cat][subcat]:
            opt.value = new
        self._lock = False

    def _on_option_change(self, change, cat, subcat, option):
        """
        Handle option checkbox changes.

        This method currently acts as a placeholder for future
        upward-propagation logic.
        """
        if self._lock:
            return

    def get_selection(self):
        """
        Retrieve the current checkbox selection.

        Returns:
            dict: Nested dictionary of selected options structured
            as {category: {subcategory: [options]}}.
        """

        selected = {}
        for cat, subcats in self.option_checkboxes.items():
            selected[cat] = {}
            for subcat, options in subcats.items():
                chosen = [cb.description for cb in options if cb.value]
                if chosen:
                    selected[cat][subcat] = chosen

        return {cat: subcat for cat, subcat in selected.items() if subcat}

    def show(self):
        """
        Display the widget in a Jupyter notebook.
        """
        display(self.widget)


class MultiSelector(BaseSelector):
    """A simple widget for selecting multiple items from a list or nested dictionary."""

    def __init__(self, dataset, data_all, data_selected, save_callback=None):
        """
        Initialize the MultiSelectorWidget.

        Args:
            dataset (str): Name of the dataset (used in widget title).
            data_all (list or dict): All possible selectable items.
                If a nested dictionary (for hazards), the widget will extract
                the deepest values as the selectable options.
            data_selected (list): List of items to pre-select.
            save_callback (callable, optional): Function to call with the selected
                items when the save button is clicked.
        """

        self.dataset = dataset

        # If hazard dataset, flatten nested dict to a list of deepest values
        self.data_all = (
            self._get_deepest_values(data_all)
            if "hazard" in dataset
            else data_all
        )
        self.data_selected = data_selected
        self.save_callback = save_callback

        # Build UI and save button
        body = self._build_ui()
        save_box = self._add_save_button(
            f"Save {dataset} selection", save_callback
        )

        # Widget title
        title = widgets.HTML(
            f"<b style='font-size:16px'>{dataset.upper()}</b>"
        )
        self.widget = widgets.VBox([title, body, save_box])

    def _build_ui(self):
        """
        Build the main checkbox list for the widget.

        Returns:
            ipywidgets.VBox: A vertical box containing all checkboxes.
        """

        self.checkboxes = [
            self._make_checkbox(item, value=item in self.data_selected)
            for item in self.data_all
        ]
        return widgets.VBox(self.checkboxes, layout=widgets.Layout(margin="0"))

    def _get_deepest_values(self, dictionary):
        """
        Recursively extract the deepest values from a nested dictionary.

        Args:
            dictionary (dict): Nested dictionary.

        Returns:
            list: Flattened list of the deepest values.
        """

        deepest_values = []
        for value in dictionary.values():
            if isinstance(value, dict):
                deepest_values.extend(self._get_deepest_values(value))
            else:
                deepest_values.extend(value)
        return deepest_values

    def get_selection(self):
        """
        Get the currently selected items.

        Returns:
            list: Descriptions of the checkboxes that are checked.
        """

        return [cb.description for cb in self.checkboxes if cb.value]

    def show(self):
        """
        Display the widget in a Jupyter notebook.
        """
        display(self.widget)


class MapWidget:
    def __init__(
        self,
        geoplot,
        map_mode: str = "choropleth",
        var_list: list = None,
        var_label: str = "Variable:",
        zoom_to_region: bool = False,
        overwrite_titles: bool = False,
        plot_displacement: bool = False,
        plot_displacement_points: bool = False,
        plot_conflict: bool = False,
        plot_conflict_points: bool = False,
        plot_conflict_exposure: bool = False,
        plot_hazard_exposure: bool = False,
        plot_mhs_exposure: bool = False,
        plot_osm_points: bool = False,
        plot_osm_networks: bool = False,
        out_dir: str = None,
    ):
        """
        Initialize a MapWidget for interactive mapping and data visualization.

        Args:
            geoplot: Geoplot object containing the data manager and datasets.
            map_mode (str): Map display mode (default: "choropleth").
            var_list (list, optional): List of variables to choose from.
            var_label (str): Label for the variable dropdown (default: "Variable:").
            zoom_to_region (bool): Whether to zoom map to selected region (default: False).
            overwrite_titles (bool): Allow overwriting default titles (default: False).
            plot_displacement (bool): Include displacement data (default: False).
            plot_displacement_points (bool): Include IDP points (default: False).
            plot_conflict (bool): Include conflict data (default: False).
            plot_conflict_points (bool): Include conflict points (default: False).
            plot_conflict_exposure (bool): Include conflict exposure map (default: False).
            plot_hazard_exposure (bool): Include hazard exposure map (default: False).
            plot_mhs_exposure (bool): Include multi-hazard exposure map (default: False).
            plot_osm_points (bool): Include OpenStreetMap point data (default: False).
            plot_osm_networks (bool): Include OpenStreetMap network data (default: False).
            out_dir (str, optional): Output directory for saving plots.
        """

        self.geoplot = geoplot
        self.map_mode = map_mode
        self.var_list = var_list or self.geoplot.dm.data.columns
        self.var_label = var_label
        self.overwrite_titles = overwrite_titles
        self.zoom_to_region = zoom_to_region
        self.plot_displacement = plot_displacement
        self.plot_displacement_points = plot_displacement_points
        self.plot_conflict = plot_conflict
        self.plot_conflict_points = plot_conflict_points
        self.plot_conflict_exposure = plot_conflict_exposure
        self.plot_hazard_exposure = plot_hazard_exposure
        self.plot_mhs_exposure = plot_mhs_exposure
        self.plot_osm_points = plot_osm_points
        self.plot_osm_networks = plot_osm_networks
        self.out_dir = self._get_out_dir(out_dir)

        self.last_vars = []
        self.output = widgets.Output()

        # Initialize all the internal state
        self._setup_var_list()
        self._setup_adm_options()
        self._setup_exposure_options()

        # Create widgets
        self._create_dropdowns()
        self._create_sliders()
        self._create_buttons()

        if self.geoplot.dm.config["osm_selected"]:
            self._create_osm_selectors()

        # Build final UI
        self._build_ui()

    def _setup_var_list(self):
        """Filter and set the default variable list based on the map mode."""
        if self.plot_displacement:
            self.var_list = [
                var for var in self.var_list if "dtm" in var or "idmc" in var
            ]
        self.default_var = (
            "worldpop" if "worldpop" in self.var_list else self.var_list[0]
        )

    def _setup_adm_options(self):
        """Prepare ADM levels and region options for dropdowns."""
        self.adm_options = [
            col
            for col in self.geoplot.dm.data.columns
            if "ADM" in col and "ID" not in col
        ][::-1]
        self.adm_region_options = self._get_adm_options(self.adm_options[0])

    def _setup_exposure_options(self):
        """Set hazard and conflict exposure options."""
        self.hazard_exposure_options = self.geoplot.dm.config["suffixes"]
        self.conflict_exposure_options = self.hazard_exposure_options[:-1]
        self.conflict_exposure_sources = self.geoplot.dm.config[
            "conflict_columns"
        ]

    def _create_dropdowns(self):
        """Create all dropdown and text widgets."""

        self.var_dropdown = widgets.Dropdown(
            options=self.var_list,
            value=self.default_var,
            description=self.var_label,
        )
        self.asset = widgets.Dropdown(
            options=self.geoplot.dm.asset_names,
            value="worldpop",
            description="Asset:",
        )

        self.map_title = widgets.Text(
            value=None,
            description="Title:",
        )
        self.map_subtitle = widgets.Text(
            value=None,
            description="Subtitle:",
        )

        self.legend_type = widgets.Dropdown(
            options=["bins", "colorbar", "barplot"],
            value="bins",
            description="Legend:",
        )

        self.binning = widgets.Dropdown(
            options=["equal_intervals", "quantiles"],
            value="equal_intervals",
            description="Binning:",
        )

        self.var_bounds = widgets.Dropdown(
            options=["[0, 1]", "[min, max]"],
            value="[min, max]",
            description="Bounds:",
        )

        self.hazard_exposure_type = widgets.Dropdown(
            options=self.hazard_exposure_options,
            value="exposure_relative",
            description="Hazard exposure:",
        )

        self.conflict_exposure_type = widgets.Dropdown(
            options=self.conflict_exposure_options,
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

        self.conflict_exposure_source = widgets.Dropdown(
            options=self.conflict_exposure_sources,
            value=self.conflict_exposure_sources[0],
            description="Conflict data:",
        )

        self.hazard_exposure_source = widgets.Dropdown(
            options=self.geoplot.dm.hazard_names,
            value=self.geoplot.dm.hazard_names[0],
            description="Hazard:",
        )

        self.hazard_category = widgets.Dropdown(
            options=["all"]
            + list(self.geoplot.dm.config["hazards_all"].keys()),
            value="all",
            description="MHS category:",
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

        self.adm_level = widgets.Dropdown(
            options=self.adm_options,
            value=self.adm_options[0],
            description="ADM Level:",
        )

        self.adm_string = widgets.Dropdown(
            options=self.adm_region_options,
            description="Region:",
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

    def _create_sliders(self):
        """Create sliders for markers, transparency, and legend positions."""

        # Conflicts
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

        # IDPs
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

        # OSM
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

    def _create_osm_selectors(self):
        """Create OSM multiple selectors"""

        osm_pois = [x for x in self.geoplot.dm.osm_pois]
        self.osm_poi_selector = widgets.SelectMultiple(
            options=osm_pois,
            value=[
                osm_pois[0],
            ],
            description="OSM Points:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px", height="90px"),
        )

        osm_networks = [x for x in self.geoplot.dm.osm_networks]
        self.osm_network_selector = widgets.SelectMultiple(
            options=osm_networks,
            value=[
                osm_networks[0],
            ],
            description="OSM Networks:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px", height="90px"),
        )

    def _create_buttons(self):
        """Create action buttons and attach callbacks."""

        self.run_button = widgets.Button(
            description="Plot", button_style="primary", icon="map"
        )

        self.adm_level.observe(self._on_adm_level_change, names="value")
        self.run_button.on_click(self._on_plot_click)

        self.save_button = widgets.Button(
            description="Save", button_style="success", icon="save"
        )
        self.save_button.on_click(self._on_save_click)

    def _get_out_dir(self, name):
        out_dir = self.geoplot.dm.iso_code
        if name is not None:
            out_dir = os.path.join(out_dir, name)
        return out_dir

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
        """Handle button click."""
        with self.output:
            clear_output(wait=True)
            existing_figs = set(plt.get_fignums())

            zoom_to = None
            if self.zoom_to_region:
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

                if self.plot_conflict:
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

                if self.plot_conflict_exposure:
                    var = f"{self.conflict_exposure_source.value}_{self.asset.value}_{self.conflict_exposure_type.value}"

                if self.plot_hazard_exposure:
                    var = f"{self.hazard_exposure_source.value}_{self.asset.value}_{self.hazard_exposure_type.value}"

                if self.plot_mhs_exposure:
                    self.geoplot.dm.mhs_aggregation = (
                        self.mhs_aggregation.value
                    )
                    self.geoplot.dm.data = (
                        self.geoplot.dm.calculate_multihazard_score(
                            self.geoplot.dm.data,
                            aggregation=self.mhs_aggregation.value,
                        )
                    )
                    var = f"mhs_{self.hazard_category.value}_{self.asset.value}_{self.hazard_exposure_type.value}"

                if self.plot_conflict_exposure and self.plot_mhs_exposure:
                    var = f"mhs_{self.hazard_category.value}_{self.conflict_exposure_source.value}_{self.asset.value}_{self.hazard_exposure_type.value}"

                if var is None and self.var_dropdown.value is not None:
                    var = self.var_dropdown.value

                if var is not None:
                    logging.info(f"Plotting variable: {var}")
                    self.last_vars = [var]

                    var_bounds = [None, None]
                    if self.var_bounds.value == "[0, 1]":
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

            elif self.map_mode == "bivariate_choropleth":
                if self.plot_conflict:
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

                elif self.plot_conflict_exposure:
                    var1 = f"{self.conflict_exposure_source.value}_{self.asset.value}_{self.conflict_exposure_type.value}"

                if self.plot_hazard_exposure or self.plot_mhs_exposure:
                    if self.plot_hazard_exposure:
                        var2 = f"{self.hazard_exposure_source.value}_{self.asset.value}_{self.hazard_exposure_type.value}"

                    elif self.plot_mhs_exposure:
                        self.geoplot.dm.data = (
                            self.geoplot.dm.calculate_multihazard_score(
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
                    # var1_bounds=[0, 1],
                    # var2_bounds=[0, 1],
                    title=title,
                    subtitle=subtitle,
                    binning=self.binning.value,
                    zoom_to=zoom_to,
                    zorder=zorder,
                )
                zorder += 1

            if self.plot_conflict_points:
                ax, xpos = self.geoplot.plot_points(
                    column=self.conflict_points_column.value,
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

            if self.plot_displacement_points:
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

            if self.geoplot.dm.config["osm_selected"]:
                if self.plot_osm_networks:
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

                if self.plot_osm_points:
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
                    "No variables have been plotted yet. Please plot the map first."
                )
                return

            data = self.geoplot.dm.data.copy()
            adm_cols = [
                col
                for col in self.geoplot.dm.geoboundary.columns
                if col != "geometry"
            ]

            region_name = self.geoplot.dm.iso_code
            if self.zoom_to_region:
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

    def _build_ui(self):
        """Assemble widget layout."""

        controls = []
        if self.map_mode == "choropleth":
            controls = [
                widgets.HBox(
                    [
                        self.legend_type,
                        self.binning,
                        self.var_bounds,
                    ]
                )
            ]

            if self.plot_conflict:
                conflict_box = widgets.HBox(
                    [
                        self.conflict_data_source,
                        self.asset,
                        self.conflict_column,
                    ]
                )
                controls.insert(0, conflict_box)

            elif (
                self.plot_conflict_exposure
                or self.plot_hazard_exposure
                or self.plot_mhs_exposure
            ):
                controls.insert(0, self.asset)

                if self.plot_hazard_exposure:
                    box = [self.hazard_exposure_source]
                    box.append(self.hazard_exposure_type)
                    box = widgets.HBox(box)
                    controls.insert(1, box)

                if self.plot_mhs_exposure:
                    box = [self.hazard_category]
                    box.append(self.hazard_exposure_type)
                    box.append(self.mhs_aggregation)
                    box = widgets.HBox(box)
                    controls.insert(1, box)

                if self.plot_conflict_exposure:
                    box = [self.conflict_exposure_source]
                    box.append(self.conflict_exposure_type)
                    box = widgets.HBox(box)
                    controls.insert(1, box)
            else:
                controls.insert(0, self.var_dropdown)

        elif self.map_mode == "bivariate_choropleth":
            conflict_box = widgets.HBox(
                [self.conflict_exposure_source, self.conflict_exposure_type]
            )
            if self.plot_hazard_exposure:
                hazard_box = widgets.HBox(
                    [self.hazard_exposure_source, self.hazard_exposure_type]
                )
            elif self.plot_mhs_exposure:
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

        if self.overwrite_titles:
            titles = widgets.VBox(
                [
                    self.map_title,
                    self.map_subtitle,
                ]
            )
            controls.extend(
                [widgets.HTML("<hr style='margin:10px 0'>"), titles]
            )

        if self.zoom_to_region:
            zoom_box = widgets.VBox([self.adm_level, self.adm_string])
            controls.extend(
                [widgets.HTML("<hr style='margin:10px 0'>"), zoom_box]
            )

        if self.plot_conflict_points:
            conflict_points_box = widgets.VBox(
                [
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

        if self.plot_displacement_points:
            idp_points_box = widgets.VBox(
                [
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

        if self.geoplot.dm.config["osm_selected"]:
            if self.plot_osm_points:
                osm_pois_box = widgets.VBox([self.osm_poi_selector])
                osm_pois_style_box = widgets.VBox(
                    [
                        self.osm_pois_markerscale,
                        self.osm_pois_alpha,
                        self.osm_pois_legend1_y,
                    ]
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
                    ]
                )
            if self.plot_osm_networks:
                osm_networks_box = widgets.VBox([self.osm_network_selector])
                osm_networks_style_box = widgets.VBox(
                    [self.osm_networks_alpha, self.osm_networks_legend1_y]
                )
                controls.extend(
                    [
                        widgets.HTML("<hr style='margin:10px 0'>"),
                        widgets.HBox(
                            [
                                osm_networks_box,
                                osm_networks_style_box,
                            ]
                        ),
                    ]
                )

        controls.extend(
            [
                widgets.HTML("<hr style='margin:10px 0'>"),
                self.run_button,
                self.save_button,
            ]
        )

        self.controls = widgets.VBox(controls)

    def show(self):
        """Display the interactive widget."""
        display(self.controls, self.output)
