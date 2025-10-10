import ipywidgets as widgets
from IPython.display import display


class HierarchicalCheckboxes:
    def __init__(
        self, hierarchy, indent_per_level=20, selected_hierarchy=None
    ):
        """
        hierarchy: dict - full hierarchy {level1: {level2: [level3s]}}
        selected_hierarchy: dict - subset of hierarchy representing checked items
        """
        self.hierarchy = hierarchy
        self.selected_hierarchy = selected_hierarchy or {}
        self.indent_per_level = indent_per_level

        self.category_checkboxes = {}
        self.subtype_checkboxes = {}
        self.option_checkboxes = {}
        self.updating = {"lock": False}
        self.widgets_tree = []

        self._build_ui()
        self._apply_selection(self.selected_hierarchy)
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
            # Level 1: category
            cat_cb = self._make_checkbox(cat)
            self.category_checkboxes[cat] = cat_cb
            subtype_boxes = []

            for sub, options in subtypes.items():
                # Level 2: subtype
                sub_cb = self._make_checkbox(sub, indent=self.indent_per_level)
                self.subtype_checkboxes.setdefault(cat, {})[sub] = sub_cb

                # Level 3: options
                opt_widgets = [
                    self._make_checkbox(o, indent=2 * self.indent_per_level)
                    for o in options
                ]
                self.option_checkboxes.setdefault(cat, {})[sub] = opt_widgets

                # Nest options under subtype
                sub_box = widgets.VBox(
                    [sub_cb] + opt_widgets,
                    layout=widgets.Layout(margin="0", padding="0"),
                )
                subtype_boxes.append(sub_box)

                # Observe changes
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

            # Nest subtypes under category
            cat_box = widgets.VBox(
                [cat_cb] + subtype_boxes,
                layout=widgets.Layout(margin="0", padding="0"),
            )
            cat_cb.observe(
                lambda ch, c=cat: self._on_category_change(ch, c),
                names="value",
            )
            self.widgets_tree.append(cat_box)

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
        # Remove empty branches
        selected = {c: s for c, s in selected.items() if s}
        return selected

    def show(self):
        display(self.widget)
