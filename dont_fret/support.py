"""data handling supporting functions"""

import numpy as np


def get_binned(
    times: np.ndarray, binning_time: float, bounds: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns asynchronous data represented as events at discrete `times` into a
    time-binned format.

    Args:
        times: Input event times.
        binning_time: Width of the requested bins.
        bounds: Requested time interval of binned data. Format is (`lower_bound`, `upper_bound`).
            Value of `lower_bound` must be smaller than or equal to the min value of `times`,
            mutatis mutandis `upper_bound`.


    Returns:
          A tuple (`time`, `binned`), where `time` is the bin centers and `binned` the number
          of events in this bin.
    """
    time = np.arange(bounds[0], bounds[-1], binning_time) + binning_time / 2
    t_min = int(bounds[0] / binning_time)

    t = (times / binning_time).astype(int)  # binned times as integer mutiples of binning_time
    binned = np.bincount(t - t_min, minlength=time.size)

    return time, binned
