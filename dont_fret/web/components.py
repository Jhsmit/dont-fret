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
    value: float | int,
    on_value: Callable[[float | int | None], None],
    vtype: Type[float] | Type[int],
    vmin: Optional[float | int] = None,
    vmax: Optional[float | int] = None,
    disabled: bool = False,
    enable_restore: bool = True,
):
    """Input field for a float value with a range slider."""

    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError("vmin must be smaller than vmax")

    error, set_error = solara.use_state(False)
    message, set_message = solara.use_state("")

    def inputtext_cb(new_value: str):
        try:
            value = vtype(new_value)

        except ValueError:
            if vtype == int:
                set_message("Input must be an integer")
            else:
                set_message("Input must be a number")
            set_error(True)
            return
        if vmin is not None and value < vmin:
            set_message(f"Input must be >= {vmin}")
            set_error(True)
            return
        if vmax is not None and value > vmax:
            set_message(f"Input must be <= {vmax}")
            set_error(True)
            return
        set_error(False)
        set_message("")
        on_value(value)

    text_field = solara.InputText(
        label=label,
        value=str(value),
        error=error,
        message=message,
        disabled=disabled,
        on_value=inputtext_cb,
    )

    # TODO restore to external component
    if enable_restore:
        with solara.Row():
            (text_field,)  # type: ignore
            solara.IconButton(
                icon_name="mdi-restore", on_click=lambda *args: on_value(vmin or vmax)
            )
    else:
        text_field  # type: ignore
