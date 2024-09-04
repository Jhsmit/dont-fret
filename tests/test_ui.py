import pandas as pd
import playwright.sync_api
import yaml
from pathlib import Path
import pytest

from dont_fret import cfg

cwd = Path(__file__).parent
input_dir = cwd / "test_data" / "input"
data_dir = cwd / "test_data" / "output" / "web"

CB_LOCATOR = "div:nth-child({}) > .v-list-item__action > .v-input > .v-input__control > .v-input__slot > .v-input--selection-controls__input > .v-input--selection-controls__ripple"


@pytest.mark.skip(reason="Takes too long")
def test_main_app(
    page_session: playwright.sync_api.Page, solara_server, solara_app, tmp_path: Path
):
    cfg.web.default_dir = input_dir
    with solara_app("dont_fret.web.app"):
        page_session.goto(solara_server.base_url)

        # switch between APBS / DCBS, fails on solara 1.12
        # Currently fails on timeout, but solara already errored
        page_session.get_by_role("button", name="Burst search settings DCBS").click()
        page_session.get_by_text("APBS").click()
        page_session.get_by_role("button", name="Burst search settings APBS").click()
        page_session.get_by_text("DCBS").click()

        page_session.get_by_text("ds2").click()
        page_session.get_by_role("button", name="Add files").click()
        page_session.get_by_role("button", name="Select None").click()

        for i in range(1, 6):
            cb = page_session.locator(CB_LOCATOR.format(i))
            # expect(cb).not_to_be_checked() # is not a checkbox

        # Enable CBs 4 and 5
        for i in [4, 5]:
            cb = page_session.locator(CB_LOCATOR.format(i))
            cb.click()
            # expect(cb).to_be_checked() # is not a checkbox

        page_session.get_by_role("button", name="Search!").click()
        page_session.get_by_text("Finished burst search.").wait_for()

        page_session.get_by_role("button", name="Burst set").click()
        page_session.get_by_role("option", name="DCBS_default").get_by_text("DCBS_default").click()
        with page_session.expect_download() as download_info:
            page_session.get_by_role("button", name="Download csv").click()
        download_info.value.save_as(tmp_path / "bursts.csv")

        compare_df = pd.read_csv(tmp_path / "bursts.csv")
        reference_df = pd.read_csv(data_dir / "DCBS_default_bursts.csv")

        pd.testing.assert_frame_equal(reference_df, compare_df)

        with page_session.expect_download() as download_info:
            page_session.get_by_role("button", name="Download settings").click()

        download_info.value.save_as(tmp_path / "settings.yaml")
        burst_colors_dict = yaml.safe_load((tmp_path / "settings.yaml").read_text())
        assert burst_colors_dict == cfg.burst_search["DCBS"]

        # Set APBS settings really high, assert we dont find any bursts
        page_session.get_by_role("button", name="Burst search settings DCBS").click()
        page_session.get_by_text("APBS").click()

        page_session.locator(".v-sheet > .v-sheet > button").first.click()
        page_session.get_by_label("L").click()
        page_session.get_by_label("L").fill("5000")
        page_session.get_by_label("L").press("Enter")
        page_session.get_by_role("button", name="Save & close").click()

        page_session.get_by_role("button", name="Search!").click()
        page_session.get_by_text("No bursts found.").wait_for()

        # open the burst view tab
        page_session.get_by_role("tab", name="burst_view").click()
        page_session.get_by_text("Number of bursts: 229").wait_for()

        # open the trace view tab
        page_session.get_by_role("tab", name="trace_view").click()
        page_session.get_by_role("button", name="Photon file").click()
        page_session.get_by_text("f5.ptu").click()
        page_session.get_by_text("Loaded 944419 photons.").click()
