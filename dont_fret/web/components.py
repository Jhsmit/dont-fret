from __future__ import annotations

from typing import Callable, Optional, Type

import solara
import solara.lab
from solara.alias import rv

import dont_fret.web.state as state

# from dont_fret.web.methods import burst_search, combine_bursts, get_dataframe


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
    solara.v.use_event(div, "dblclick", handle)
