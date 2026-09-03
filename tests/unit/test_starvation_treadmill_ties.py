"""Treadmill starvation: victims are chosen by age, ties broken at random (not by array position)."""
import numpy as np
from aegis_sim import variables
from aegis_sim.submodels.resources.starvation import Starvation


def _starv(mode):
    s = Starvation()
    s.init(STARVATION_MORTALITY_MAXIMUM=1.0, STARVATION_MORTALITY_FACTOR=None, STARVATION_RESPONSE=mode)
    return s


def test_zoomer_spares_all_older_before_any_younger():
    variables.rng = np.random.default_rng(1)
    ages = np.array([0] * 100 + [25] * 20)
    mask = _starv("treadmill_zoomer").get_mask_kill(ages, resources_scavenged=50)
    assert mask.sum() == 70
    assert not mask[100:].any(), "all 25-year-olds must survive"
    assert mask[:100].sum() == 70


def test_boomer_kills_all_older_before_any_younger():
    variables.rng = np.random.default_rng(1)
    ages = np.array([0] * 100 + [25] * 20)
    mask = _starv("treadmill_boomer").get_mask_kill(ages, resources_scavenged=50)
    assert mask[100:].all(), "all 25-year-olds must die"
    assert mask[:100].sum() == 50


def test_ties_are_broken_at_random_not_by_position():
    """Among equal-aged individuals, the survivors must not be the ones at the end (or start) of the array."""
    variables.rng = np.random.default_rng(7)
    ages = np.zeros(1000, dtype=int)
    reps = 200
    first_half_survivors = 0
    for _ in range(reps):
        mask = _starv("treadmill_zoomer").get_mask_kill(ages, resources_scavenged=500)
        first_half_survivors += (~mask[:500]).sum()
    frac = first_half_survivors / (reps * 500)
    assert 0.45 < frac < 0.55, f"positional bias: {frac:.3f} of the first half survive (expected ~0.5)"
    variables.rng = np.random.default_rng(7)
    mask = _starv("treadmill_boomer").get_mask_kill(ages, resources_scavenged=500)
    assert 200 < (~mask[:500]).sum() < 300


def test_treadmill_brings_population_to_capacity_exactly():
    variables.rng = np.random.default_rng(3)
    ages = variables.rng.integers(0, 50, size=1234)
    for mode in ("treadmill_zoomer", "treadmill_boomer"):
        mask = _starv(mode).get_mask_kill(ages, resources_scavenged=1000)
        assert (~mask).sum() == 1000
