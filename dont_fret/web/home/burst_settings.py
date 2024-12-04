from functools import partial

import humanize
import solara
from solara.alias import rv

from dont_fret.config import cfg
from dont_fret.config.config import BurstColor
from dont_fret.web.models import ListStore, use_liststore


@solara.component
def BurstColorSettingForm(color_store: ListStore[BurstColor], color_idx: int):
    burst_color = color_store[color_idx]
    # burst_color = burst_settings.get_color(settings_name, color_idx)

    setter = partial(color_store.update, color_idx)
    # setter = partial(burst_settings.update_color, settings_name, color_idx)
    with solara.ColumnsResponsive([8, 4]):
        with solara.Card("Search Thresholds"):
            with solara.Column():
                # TODO figure out why
                with solara.Tooltip("Minimal number of 'burst' photons."):  # type: ignore
                    # TODO doesnt keep its value when losing focus; ONLY when pressing enter (-> continous_update)
                    solara.InputInt(
                        label="L",
                        value=burst_color.L,
                        on_value=lambda val: setter(L=val),
                        continuous_update=True,
                    )
                with solara.Tooltip("Number of neighbours in interval."):  # type: ignore
                    solara.InputInt(
                        label="M",
                        value=burst_color.M,
                        on_value=lambda val: setter(M=val),
                        continuous_update=True,
                    )
                with solara.Tooltip("Interval size in seconds."):  # type: ignore
                    solara.InputFloat(
                        label="T",
                        value=burst_color.T,
                        on_value=lambda val: setter(T=val),
                        continuous_update=True,
                    )

        with solara.Card("Photon Streams"):
            with rv.List():
                with solara.Tooltip("Select the streams to include in this color."):  # type: ignore
                    for stream in cfg.streams:

                        def on_value(val, stream: str = stream):
                            new_streams = burst_color.streams.copy()
                            if val:
                                new_streams.append(stream)
                            else:
                                new_streams.remove(stream)
                            setter(streams=new_streams)

                        with rv.ListItem():
                            rv.ListItemTitle(children=[stream])
                            with rv.ListItemAction():
                                solara.Checkbox(
                                    value=stream in burst_color.streams, on_value=on_value
                                )


@solara.component
def BurstSettingsDialog(colors: list[BurstColor], settings_name, on_value, on_close):
    tab, set_tab = solara.use_state(0)  # active tab number
    color_store = use_liststore(colors)
    title = f"Editing burst search settings: '{settings_name}'"

    with solara.Card(title):

        def on_tab_change(val):
            set_tab(val)

        def on_tab_remove(*args):
            color_store.pop(len(color_store) - 1)
            # burst_settings.remove_color(settings_name)

        def on_tab_add(*args):
            color_store.append(BurstColor())
            # burst_settings.add_color(settings_name)

        try:
            n_tabs = len(color_store)
        # This can happen if the settings are reset while a new set was added and is selected
        # update: not sure how we can get here atm
        except KeyError:
            return

        with solara.ColumnsResponsive([1, 10, 1]):
            solara.IconButton(icon_name="mdi-minus", color="primary", on_click=on_tab_remove)
            with solara.Row():
                with rv.Tabs(
                    centered=True, grow=True, v_model=tab, on_v_model=on_tab_change, height=30
                ):
                    for i in range(n_tabs):  # Ref / fields ?
                        rv.Tab(children=[f"Color {humanize.apnumber(i + 1)}"])
            solara.IconButton(icon_name="mdi-plus", color="primary", on_click=on_tab_add)

        with rv.TabsItems(v_model=tab):
            for i in range(n_tabs):
                with rv.TabItem():
                    BurstColorSettingForm(color_store, i)

        def save_close():
            on_value(color_store.items)
            on_close(False)

        with solara.CardActions():
            # todo align center
            with rv.Layout(row=True):
                solara.Button(
                    label="Save & close",
                    on_click=save_close,
                    text=True,
                    classes=["centered"],
                )
