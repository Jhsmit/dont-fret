from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional, Type

import solara
import solara.lab
from solara.alias import rv

import dont_fret.web.state as state


@solara.component
def FRETStyle():
    solara.Style(Path(__file__).parent / "style.css")


@solara.component
def Snackbar():
    bar = state.snackbar.value

    close_btn = solara.Button(
        label="close",
        color=bar.btn_color,
        text=True,
        on_click=lambda: state.snackbar.update(show=False),
    )
    children = [bar.message, close_btn]
    rv.Snackbar(
        v_model=bar.show,
        color=bar.color,
        timeout=bar.timeout,
        on_v_model=lambda x: state.snackbar.update(show=x),
        children=children,
    )


@solara.component
def RangeInputField(
    label: str,
    value: float | int | None,
    on_value: Callable[[float | int | None], None],
    vtype: Type[float] | Type[int],
    vmin: Optional[float | int] = None,
    vmax: Optional[float | int] = None,
    disabled: bool = False,
    allow_none: bool = False,
):
    """Input field for a float value with a range slider."""

    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError("vmin must be smaller than vmax")

    error = solara.use_reactive(False)
    message = solara.use_reactive("")

    def inputtext_cb(new_value: str) -> None:
        if not new_value:
            if allow_none:
                on_value(None)
                error.set(False)
                message.set("")
                return
            else:
                message.set("Input required")
                error.set(True)
        try:
            value = vtype(new_value)

        except ValueError:
            if vtype == int:
                message.set("Input must be an integer")
            else:
                message.set("Input must be a number")
            error.set(True)
            return
        if vmin is not None and value < vmin:
            message.set(f"Input must be >= {vmin}")
            error.set(True)
            return
        if vmax is not None and value > vmax:
            message.set(f"Input must be <= {vmax}")
            error.set(True)
            return
        error.set(False)
        message.set("")
        on_value(value)

    solara.InputText(
        label=label,
        value=str(value),
        error=error.value,
        message=message.value,
        disabled=disabled,
        on_value=inputtext_cb,
    )


@solara.component
def EditableTitle(initial: str | solara.Reactive[str], edit: bool | solara.Reactive[bool] = False):
    title = solara.use_reactive(initial)
    edit_mode = solara.use_reactive(edit)

    def on_edit(value: str):
        edit_mode.set(False)
        title.set(value)

    def handle(*args):
        if edit_mode.value:
            return
        edit_mode.set(True)

    if edit_mode.value:
        children = [solara.InputText(label="", value=title.value, on_value=on_edit)]

    else:
        children = [solara.Text(title.value)]

    div = solara.Div(children=children)
    solara.v.use_event(div, "dblclick", handle)  # type: ignore


@solara.component
def FigureFromTask(task: solara.lab.Task):  # type: ignore
    solara.ProgressLinear(task.pending)
    if task.latest is None:
        solara.Text("loading...")
    else:
        figure = task.value if task.finished else task.latest
        with solara.Div(style="opacity: 0.3" if task.pending else None):
            solara.FigurePlotly(figure)


@solara.component
def RegexSelectDialog(
    title: str,
    value: list[str],
    on_value: Callable[[list[str]], None],
    values: list[str],
    on_close: Callable[[], None],
    sort: bool = True,
):
    """
    select string by checkboxes or regex
    """
    local_selection = solara.use_reactive(value)
    error = solara.use_reactive("")
    regex = solara.use_reactive("")

    def on_input(value: str):
        try:
            pattern = re.compile(value)
            regex.set(value)
            error.set("")
        except Exception:
            error.set("Invalid regex")
            return
        new_selected = [f for f in values if pattern.search(f)]
        local_selection.set(new_selected)

    def on_save():
        if not local_selection.value:
            return
        if sort:
            on_value(sorted(local_selection.value))
        else:
            on_value(local_selection.value)
        on_close()

    with solara.Card(title):
        with solara.Row(style="align-items: center;"):
            solara.InputText(
                label="regex",
                value=regex.value,
                on_value=on_input,
                continuous_update=True,
                error=error.value,
            )
            solara.Button(label="Select All", on_click=lambda: local_selection.set(values))
            solara.Button(label="Select None", on_click=lambda: local_selection.set([]))
        with solara.v.List(nav=True):
            with solara.v.ListItemGroup(
                v_model=local_selection.value,
                on_v_model=local_selection.set,
                multiple=True,
            ):
                for v in values:
                    with solara.v.ListItem(value=v):
                        with solara.v.ListItemAction():
                            solara.Checkbox(value=v in local_selection.value)
                        solara.v.ListItemTitle(children=[v])

        with solara.CardActions():
            solara.v.Spacer()
            solara.Button(
                "Save",
                icon_name="mdi-content-save",
                on_click=on_save,
                disabled=not local_selection.value,
            )
            solara.Button("Close", icon_name="mdi-window-close", on_click=on_close)
