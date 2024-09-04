from typing import List, Tuple, Union

import numpy as np
from numba import int32, jit, types


@jit(
    (types.Array(types.int64, 1, "C", readonly=True), int32, int32, int32),
    nopython=True,
    cache=True,
    nogil=True,
)
def bs_eggeling(times: np.ndarray, L: int, M: int, tr: int) -> List[Tuple[int, int]]:
    """
    Burst search algorithm from Eggeling et al. 2001.

    "Data registration and selective single-molecule analysis using multi-parameter fluorescence detection"
    DOI: 10.1016/S0168-1656(00)00412-0

    Args:
        times: Array of photon arrival times (uint64)
        L: Number of succesive 'burst' photons to be considered a burst
        M: With at least this many neighbours
        tr: Within this time period from the burst (total time window is 2*tr)

    Returns:
        List of tuples of start and end indices of bursts.

    """

    in_burst = False
    bursts = []
    i_start = 0
    i_lower = 0  # index of lower limit neighbouring photon within time tr
    i_upper = 0  # index of upper limit neighbouring photon within time tr

    for i, time in enumerate(times):
        # Adjust upper limit
        while times[i_upper] < time + tr:
            i_upper += 1
            if i_upper == len(times):
                i_upper -= 1
                break

        # Adjust lower limit
        while times[i_lower] < time - tr:
            i_lower += 1

        n_neighbours = i_upper - i_lower

        if n_neighbours > M:
            if not in_burst:
                in_burst = True
                i_start = i
        elif in_burst:  # Burst has ended
            in_burst = False
            if (i - 1 - i_start) > L:  # Check enough photons in the burst
                bursts.append((i_start, i - 1))

    return bursts


def return_intersections(
    intervals1: List[Tuple[int, int]], intervals2: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Return the intersections of two lists of intervals.

    The intervals need to be ordered.

    From:
    https://codereview.stackexchange.com/questions/178427/given-2-disjoint-sets-of-intervals-find-the-intersections


    Args:
        intervals1: First set of intervals as list of tuples.
        intervals2: First set of intervals as list of tuples.

    Returns:
        List of overlaps of intervals1 and intervals2 as list of tuples.
    """

    i1 = 0
    i2 = 0

    output = []

    interval1 = intervals1[i1]
    interval2 = intervals2[i2]

    while True:
        start = max(interval1[0], interval2[0])
        end = min(interval1[1], interval2[1])

        if start < end:
            output.append((start, end))
            if end == interval1[1]:
                i1 += 1
                if i1 >= len(intervals1):
                    break
                interval1 = intervals1[i1]
            else:
                i2 += 1
                if i2 >= len(intervals2):
                    break
                interval2 = intervals2[i2]

        while interval1[0] >= interval2[1]:
            i2 += 1
            if i2 >= len(intervals2):
                break
            interval2 = intervals2[i2]
        while interval2[0] >= interval1[1]:
            i1 += 1
            if i1 >= len(intervals1):
                break
            interval1 = intervals1[i1]

        if i1 >= len(intervals1):
            break
        if i2 >= len(intervals2):
            break

    return output


def get_intersection(
    interval_1: Tuple[int, int], interval_2: Tuple[int, int]
) -> Union[Tuple[int, int], None]:
    """Returns the intersection (overlap) of two intervals.

    Args:
        interval_1: First interval as tuple.
        interval_2: Second interval as tuple.

    Returns:
        Intersection of the two intervals as tuple or `None` if no intersection.

    """
    start = max(interval_1[0], interval_2[0])
    end = min(interval_1[1], interval_2[1])
    if start < end:
        return start, end
    return None
