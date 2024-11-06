"""Solara components for the home page."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

import polars as pl
import solara
import solara.lab
import yaml
from solara.alias import rv

import dont_fret.web.state as state
from dont_fret.config.config import BurstColor
from dont_fret.web.components import EditableTitle
from dont_fret.web.home.burst_settings import BurstSettingsDialog
from dont_fret.web.home.methods import task_burst_search
from dont_fret.web.methods import format_size
from dont_fret.web.models import (
    BurstNode,
    BurstSettingsStore,
    ListStore,
    PhotonNode,
)
from dont_fret.web.utils import has_photons


@solara.component
def FRETNodeInfoCard(
    name: solara.Reactive[str],
    description: solara.Reactive[str],
    on_delete: Callable[[], None],
):
    edit = solara.use_reactive(False)

    with solara.Card(EditableTitle(name, edit)):  # type: ignore
        rv.Textarea(label="Description", v_model=description.value, on_v_model=description.set)
        with solara.CardActions():
            rv.Spacer()
            solara.IconButton("edit", on_click=lambda: edit.set(not edit.value))
            solara.IconButton("delete", on_click=lambda *args: on_delete())


@solara.component
def PhotonInfoCard(
    name: str,
    photon_store: ListStore[PhotonNode],
    burst_store: ListStore[BurstNode],
    filebrowser_folder: solara.Reactive[Path],
    burst_settings: BurstSettingsStore,
    # open_: solara.Reactive[list[str]],
):
    # todo reactives
    bs_name, set_bs_name = solara.use_state(next(iter(burst_settings.value.keys())))
    add_new, set_add_new = solara.use_state(False)
    show_settings_dialog, set_show_settings_dialog = solara.use_state(False)

    open_confirmation_dialog = solara.use_reactive(False)

    # open the current node so that users can see the files they're adding
    def open_node():
        # Todo attach to callable
        pass
        # ph_node_id = f"{node_idx}:photons"

        # if ph_node_id not in open_.value:
        #     new_open = open_.value.copy()
        #     new_open.append(ph_node_id)
        #     open_.value = new_open

    def filebrowser_filter(pth: Path) -> bool:
        """Return True if the path is a ptu file or a folder."""
        if pth.is_dir():
            return True
        elif pth.suffix == ".ptu":
            return True
        else:
            return False

    def add_single_file(ptu_pth: Path):
        if ptu_pth.is_dir():
            return
        elif ptu_pth.suffix != ".ptu":
            return

        add_files([ptu_pth])

    def add_all_files(*ignore):
        filebrowser_folder.value
        ptu_files = list(filebrowser_folder.value.glob("*.ptu"))  # TODO allow other file types
        add_files(ptu_files)

    def add_files(files: list[Path]):
        open_node()
        current_names = [ph.name for ph in photon_store.items]
        to_add = [PhotonNode(file_path=pth) for pth in files if pth.name not in current_names]
        # TODO sort ?

        photon_store.extend(to_add)
        state.disable_trace_page.set(not has_photons(state.fret_nodes.items))

    def remove_all_files(*ignore):
        photon_store.set([])
        state.disable_trace_page.set(not has_photons(state.fret_nodes.items))

    def confirm_burst_search():
        # TODO needs to be passed as callable as well

        if not photon_store.items:
            state.snackbar.warning("No photon files selected", timeout=0)
            return
        if bs_name in {node.name for node in burst_store.items}:
            open_confirmation_dialog.value = True
        else:
            do_burst_search()

    def do_burst_search():
        # open the bursts node if not open already
        # bursts_node_id = f"{measurement_id}:bursts:"
        # if bursts_node_id not in open_.value:
        #     new_open = open_.value.copy()
        #     new_open.append(bursts_node_id)
        #     open_.value = new_open

        # remove the burst node if its already in the store
        try:
            names = [node.name for node in burst_store.items]
            idx = names.index(bs_name)
            burst_store.pop(idx)
        except ValueError:
            pass

        task_burst_search(bs_name, photon_store.items, burst_store)

    def on_new_settings(new_name):
        if new_name in burst_settings.settings_names:
            return
        set_add_new(False)
        burst_settings.add_settings(new_name)
        set_bs_name(new_name)

    with solara.Card(f"{name} / Photons"):
        with solara.Columns([3, 1], gutters=False, gutters_dense=True):
            with solara.Card(margin=0):
                solara.Text("Click files to add")
                solara.FileBrowser(
                    directory=filebrowser_folder,
                    on_file_open=add_single_file,
                    filter=filebrowser_filter,
                    directory_first=True,
                )

            with solara.Card("Burst Search", margin=0):
                # TODO the settings need to move to a component
                with solara.Column():
                    if add_new:
                        solara.InputText(
                            label="New settings name", value="", on_value=on_new_settings
                        )
                    else:
                        solara.Select(
                            label="Burst search settings",
                            value=bs_name,
                            on_value=set_bs_name,
                            values=list(
                                burst_settings.value.keys()
                            ),  # color names? colorset names?
                        )
                    with solara.Row(gap="0px", justify="space-evenly"):
                        with solara.Tooltip("Edit current settings"):  # type: ignore
                            solara.IconButton(
                                icon_name="mdi-pencil",
                                on_click=lambda *args: set_show_settings_dialog(True),
                            )

                        with solara.Tooltip("Add new settings"):  # type: ignore
                            solara.IconButton(
                                icon_name="mdi-plus", on_click=lambda *args: set_add_new(True)
                            )

                        def remove_settings():
                            new_names = burst_settings.settings_names
                            if len(new_names) == 1:
                                return

                            new_names.remove(bs_name)
                            set_bs_name(new_names[0])
                            burst_settings.remove_settings(bs_name)

                        with solara.Tooltip("Delete current settings"):  # type: ignore
                            solara.IconButton(
                                icon_name="mdi-delete", on_click=lambda *args: remove_settings()
                            )

                        def reset():
                            burst_settings.reset()
                            # possibly we enter an illegal state, so set name to the first entry.
                            set_bs_name(next(iter(burst_settings.value.keys())))

                        with solara.Tooltip("Restore default settings"):  # type: ignore
                            solara.IconButton(
                                icon_name="mdi-restore",
                                on_click=lambda *args: reset(),
                            )

                    solara.Button(
                        "Search!",
                        on_click=confirm_burst_search,
                        disabled=task_burst_search.pending,
                        block=True,
                    )

                    solara.lab.ConfirmationDialog(
                        open_confirmation_dialog,
                        content=f"Burst search result {bs_name!r} already exists. Overwrite?",
                        on_ok=do_burst_search,
                    )

                    solara.ProgressLinear(
                        task_burst_search.progress if task_burst_search.pending else False  # type: ignore
                    )

        with solara.CardActions():
            solara.Button("Add all files", text=True, on_click=add_all_files)
            solara.Button("Remove all files", text=True, on_click=remove_all_files)

    with rv.Dialog(
        v_model=show_settings_dialog,
        on_v_model=set_show_settings_dialog,
        persistent=False,
        max_width=800,
    ):

        def on_value(value: list[BurstColor], name=bs_name):
            burst_settings[name] = value

        BurstSettingsDialog(
            burst_settings[bs_name], bs_name, on_value=on_value, on_close=set_show_settings_dialog
        )


@solara.component  # type: ignore
def BurstInfoCard(
    name: str,
    burst_store: ListStore[BurstNode],
    filebrowser_folder: solara.Reactive[Path],
    # open_: solara.Reactive[list[str]],
):
    # TODO different file types for bursts
    def filebrowser_filter(pth: Path) -> bool:
        """Return True if the path is a .csv file or a folder."""
        if pth.is_dir():
            return True
        elif pth.suffix == ".csv":
            return True
        else:
            return False

    def add_single_file(burst_pth: Path):
        if burst_pth.is_dir():
            return

        state.snackbar.warning("Not implemented")
        # open the leaf we are adding to
        # if bursts_node_id not in open_.value:
        #     new_open = open_.value.copy()
        #     new_open.append(bursts_node_id)
        #     open_.value = new_open

        # check if the item already exists:
        if burst_pth.stem in [b.name for b in burst_store.items]:
            state.snackbar.warning(f"Burst result name {burst_pth.stem!r} already exists.")
            return

        # TODO via dataloader object
        raise NotImplementedError("TODO via dataloader object")
        # burst_store.extend([BurstNode.from_path(burst_pth)])

    with solara.Card(f"{name} / Bursts"):
        solara.Text("Click files to add")
        solara.FileBrowser(
            directory=filebrowser_folder.value,
            on_directory_change=filebrowser_folder.set,
            on_file_open=add_single_file,
            filter=filebrowser_filter,
            directory_first=True,
        )


@solara.component
def PhotonNodeInfoCard(node_name: str, photon_node: PhotonNode, on_delete: Callable[[], None]):
    # prevent flicker by setting `loaded` to `True` if we find ourselves in this component
    # with the photon loading future already completed
    loaded, set_loaded = solara.use_state(False)

    async def load_info():
        if photon_node.id in state.data_manager.photon_cache:
            set_loaded(True)
        else:
            set_loaded(False)
        info = await state.data_manager.get_info(photon_node)
        return info

    result = solara.lab.use_task(load_info, dependencies=[photon_node], prefer_threaded=False)  # type: ignore

    with solara.Card(f"{node_name} / Photon file / {photon_node.name}"):
        solara.ProgressLinear(result.pending and not loaded)

        if result.pending and not loaded:
            solara.Div(style="height: 20px;")
            solara.Info("Loading photons...")

        elif result.finished:
            info: dict = result.value  # type: ignore
            solara.Text("General info:")
            dt = datetime.strptime(info["creation_time"], "%Y-%m-%d %H:%M:%S")

            headers = [
                {"text": "Property", "value": "property", "width": "50%"},
                {"text": "Value", "value": "value", "width": "50%"},
            ]

            items = [
                {
                    "property": "Creation date",
                    "value": f"{dt:%d %B %Y}",
                },
                {
                    "property": "Creation time",
                    "value": f"{dt:%H:%M %Ss}",
                },
                # TODO humanize file size
                {
                    "property": "File size",
                    "value": format_size(photon_node.size),
                },
                {
                    "property": "Number of photons",
                    "value": info["number_of_photons"],
                },
                {
                    "property": "Acquisition duration",
                    "value": f"{info['acquisition_duration']} s",
                },
                {
                    "property": "Power diode",
                    "value": f"{info['power_diode']:.2f} au",
                },
            ]

            rv.DataTable(
                headers=headers,
                items=items,
                hide_default_footer=True,
                hide_default_header=True,
                calculate_widths=True,
            )

            solara.Div(style="height: 20px;")
            solara.Text("Count rates:")
            headers = [
                {"text": "Stream", "value": "stream"},
                {"text": "Count rate (Hz)", "value": "count_rate", "width": "50%"},
            ]

            items = [
                {
                    "stream": "All",
                    "count_rate": f"{info['cps']:.2f}",
                }
            ]
            stream_items = [
                {"stream": stream_name, "count_rate": f"{cps:.2f}"}
                for stream_name, cps in info["stream_cps"].items()
            ]

            rv.DataTable(
                headers=headers,
                items=items + stream_items,
                hide_default_footer=True,
                # hide_default_header=True,
                calculate_widths=True,
            )

            solara.Div(style="height: 20px;")

            if comment := info.get("comment", None):
                solara.Text(f"User comment: {comment}")

        with solara.CardActions():
            rv.Spacer()
            solara.IconButton("delete", on_click=lambda *args: on_delete())


@solara.component  # type: ignore
def BurstItemInfoCard(node_name: str, burst_node: BurstNode, on_delete: Callable[[], None]):
    headers = [
        {"text": "Filename", "value": "filename"},
        {"text": "Number of bursts", "value": "bursts"},
        {"text": "Photons per burst", "value": "n_photons"},
        {"text": "Duration (ms)", "value": "time_length"},
        {"text": "Interburst time (s)", "value": "ibt"},
        {"text": "Bursts per second (Hz)", "value": "bps"},
        {"text": "In-burst countrate (kHz)", "value": "burst_cps"},
    ]

    filenames = burst_node.df["filename"].unique().sort()
    items = []
    for fname in filenames:
        item = {"filename": fname}
        df = burst_node.df.filter(pl.col("filename") == fname)

        item["bursts"] = len(df)

        inter_burst_times = df["time_mean"].diff().drop_nulls()
        ibt_mean, ibt_std = inter_burst_times.mean(), inter_burst_times.std()
        item["ibt"] = f"{ibt_mean:.2f} ± {ibt_std:.2f}"
        items.append(item)

        bps_mean = 1 / ibt_mean  # type: ignore
        bps_std = (1 / (ibt_mean**2)) * ibt_std  # type: ignore

        item["bps"] = f"{bps_mean:.2f} ± {bps_std:.2f}"

        item["n_photons"] = f"{df['n_photons'].mean():.1f} ± {df['n_photons'].std():.1f}"
        item[
            "time_length"
        ] = f"{df['time_length'].mean()*1e3:.2f} ± {df['time_length'].std()*1e3:.2f}"  # type: ignore

        burst_cps = df["n_photons"] / df["time_length"]
        item["burst_cps"] = f"{burst_cps.mean()*1e-3:.2f} ± {burst_cps.std()*1e-3:.2f}"  # type: ignore

    with solara.Card(f"{node_name} / Bursts / {burst_node.name}"):
        solara.Text(f"Total number of bursts: {len(burst_node.df)}", style="font-weight: bold;")
        solara.HTML(tag="br")
        solara.Text("Bursts statistics per file:")

        rv.DataTable(
            headers=headers,
            items=items,
            calculate_widths=True,
            hide_default_footer=len(items) <= 10,
        )

        with solara.CardActions():
            solara.FileDownload(
                burst_node.df.write_csv,
                filename=f"{burst_node.name}_bursts.csv",
                children=[
                    solara.Button(
                        "Download burst data.csv",
                        text=True,
                    )
                ],
            )
            solara.FileDownload(
                lambda: yaml.dump(yaml.dump([asdict(c) for c in burst_node.colors])),
                filename=f"{burst_node.name}_setttings.yaml",
                children=[
                    solara.Button(
                        "Download settings",
                        text=True,
                    )
                ],
            )
            rv.Spacer()
            solara.IconButton("delete", on_click=lambda *args: on_delete())
