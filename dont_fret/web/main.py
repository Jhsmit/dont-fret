import argparse
import copy
import sys
from pathlib import Path

import altair as alt
import solara
import solara.lab
import solara.server.settings
import yaml

import dont_fret.web.state as state
from dont_fret.config import cfg
from dont_fret.web.bursts import BurstPage
from dont_fret.web.components import Snackbar
from dont_fret.web.home import HomePage
from dont_fret.web.models import BurstColorList
from dont_fret.web.state import disable_burst_page, disable_trace_page
from dont_fret.web.trace import TracePage

# config option?
alt.data_transformers.enable("vegafusion")


parser = argparse.ArgumentParser(description="Process config argument")
parser.add_argument("--config", help="Configuration file")

if "--" in sys.argv:
    extra_args = sys.argv[sys.argv.index("--") + 1 :]
    parsed = parser.parse_args(extra_args)

    data = yaml.safe_load(Path(parsed.config).read_text())
    cfg.update(data)


SCRIPT_PATH = Path(__file__).parent
PAGES = [
    {
        "name": "home",
        "main": HomePage,
        "sidebar": None,
        "disabled": solara.Reactive(False),  # always false
        "show": lambda: True,
    },
    {
        "name": "bursts",
        "main": BurstPage,
        "sidebar": None,
        "disabled": disable_burst_page,
        "show": lambda: True,
    },
    {
        "name": "trace",
        "main": TracePage,
        "sidebar": None,
        "disabled": disable_trace_page,
        "show": lambda: True,
    },
]


@solara.component
def Page():
    tab_selection = solara.use_reactive(0)
    authorized = solara.use_reactive(cfg.web.password is None)
    login_failed = solara.use_reactive(False)
    password = solara.use_reactive("")

    solara.Style(SCRIPT_PATH / "style.css")

    def initialize():
        # TODO burst settings as listStore
        state.burst_settings.set({k: BurstColorList(v) for k, v in cfg.burst_search.items()})  # type: ignore

        default_filters = copy.deepcopy(cfg.web.burst_filters)
        state.filters.set(default_filters)

        state.filebrowser_folder.set(cfg.web.default_dir)
        if cfg.web.protect_filebrowser:

            def check_parent(new_value: Path):
                if new_value in cfg.web.default_dir.parents:
                    state.filebrowser_folder.set(cfg.web.default_dir)

            return state.filebrowser_folder.subscribe(check_parent)

    solara.use_effect(initialize, [])

    def authorize(value: str):
        if value == cfg.web.password:
            authorized.set(True)
        else:
            login_failed.set(True)

    # has_photons = [node for node in state.fr--

    # it is important we do not interrupt the height 100% chain
    # to ensure independent scrolling for both columns
    with solara.Column(style={"height": "100%"}):
        Snackbar()
        with solara.AppBar():
            solara.lab.ThemeToggle()
            with solara.lab.Tabs(
                value=tab_selection.value, on_value=tab_selection.set, align="center"
            ):
                for page in PAGES:
                    if page["show"]():
                        solara.lab.Tab(page["name"], disabled=page["disabled"].value)
        solara.Title(state.APP_TITLE)

        if authorized.value:
            PAGES[tab_selection.value]["main"]()
            sidebar = PAGES[tab_selection.value]["sidebar"]
            if sidebar is not None:
                sidebar()
        else:
            with solara.Row(style="align-items: center;"):
                solara.InputText(
                    label="Password", password=True, value=password.value, on_value=authorize
                )
                solara.Button(label="Submit", on_click=lambda: authorize(password.value))
            if login_failed.value:
                solara.Error("Incorrect password")
            else:
                solara.Info("Please enter password")


@solara.component
def Layout(children):
    dark_effective = solara.lab.use_dark_effective()
    return solara.AppLayout(
        children=children,
        toolbar_dark=dark_effective,
        color="None",  # if dark_effective else "primary",
    )
