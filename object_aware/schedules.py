"""Iteration schedules used by object-aware optimization."""

from typing import List


def object_mark_active_at_iteration(
    enabled: bool,
    start_iteration: int,
    end_iteration: int,
    iteration: int,
) -> bool:
    """Return whether ObjectMark is active in the half-open interval.

    The method uses ``start_iteration <= iteration < end_iteration``.
    """
    return enabled and iteration >= start_iteration and iteration < end_iteration


def default_object_mark_pruning_iterations(
    object_mark_end_iteration: int,
    total_iterations: int,
    start_iteration: int = 2_500,
    interval: int = 500,
) -> List[int]:
    """Return the paper schedule: every 500 steps from 2,500 to the active end.

    The end of the ObjectMark window is exclusive, so the default 30,000-step
    run prunes at 2,500, 3,000, ..., 29,500.
    """
    upper_exclusive = min(total_iterations + 1, object_mark_end_iteration)
    if upper_exclusive <= start_iteration:
        return []
    return list(range(start_iteration, upper_exclusive, interval))
