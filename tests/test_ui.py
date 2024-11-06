"""
playwright ui tests


playwright python code generator: https://playwright.dev/python/docs/codegen

"""

from pathlib import Path

import playwright.sync_api
import polars as pl
import pytest
from IPython.display import display

from dont_fret import cfg
from dont_fret.web.main import Page as MainPage

cwd = Path(__file__).parent

input_dir = cwd / "test_data" / "input"
data_dir = cwd / "test_data" / "output" / "web"

CB_LOCATOR = "div:nth-child({}) > .v-list-item__action > .v-input > .v-input__control > .v-input__slot > .v-input--selection-controls__input > .v-input--selection-controls__ripple"


@pytest.mark.skip("fails on due to missing data files")
def test_mainpage(solara_test, page_session: playwright.sync_api.Page, tmp_path: Path):
    # The test code runs in the same process as solara-server (which runs in a separate thread)
    cfg.web.default_dir = input_dir / "ds2"
    page = MainPage()

    display(page)

    # add photons and do burst search
    page_session.get_by_role("button", name="").click()
    page_session.get_by_text("Photons").click()
    page_session.get_by_role("button", name="Add all files").click()
    page_session.get_by_role("button", name="Search!").click()
    page_session.get_by_text("Burst search completed, found").wait_for()
    page_session.get_by_role("button", name="close").click()
    page_session.get_by_role("button", name="Burst search settings DCBS").click()
    page_session.get_by_text("APBS").click()
    page_session.get_by_role("button", name="Search!").click()

    # download results and compare
    page_session.get_by_role("button", name="").nth(2).click()
    page_session.get_by_text("DCBS").first.click()
    with page_session.expect_download() as download_info:
        page_session.get_by_role("button", name="Download burst data.csv").click()

    download_info.value.save_as(tmp_path / "bursts.csv")
    compare_df = pl.read_csv(tmp_path / "bursts.csv")
    reference_df = pl.read_csv(data_dir / "DCBS_default_bursts.csv")
    assert compare_df.equals(reference_df)
