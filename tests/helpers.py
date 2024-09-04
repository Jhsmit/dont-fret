from typing import Type

import numpy as np

from dont_fret.models import Bursts, PhotonData


def assert_burstsets_equal(bs1: Bursts, bs2: Bursts) -> None:
    # This does (probably) not check every datafield
    # TODO add __eq__ on photon objects?
    assert len(bs1) == len(bs2)
    # todo this needs to be listed centrally for future GUI
    for attr in ["n_photons", "E_app", "S_app", "time_length", "time_mean", "time_max"]:
        assert np.allclose(
            getattr(bs1, attr), getattr(bs2, attr), equal_nan=True
        ), f"{attr!r} is not equal"


def assert_photons_equal(ph1: Type[PhotonData], ph2: Type[PhotonData]) -> None:
    assert len(ph1) == len(ph2)

    for attr in ["timestamps", "nanotimes", "photon_times"]:
        a1 = getattr(ph1, attr)
        a2 = getattr(ph2, attr)
        assert np.all(a1 == a2)
        assert a1.dtype == a2.dtype

    for attr in ["monotonic", "timestamps_unit", "nanotimes_unit", "description"]:
        assert getattr(ph1, attr) == getattr(ph2, attr)
