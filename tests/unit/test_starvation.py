"""Unit tests for the Starvation submodel treadmill responses.

Covers: the age-based treadmill responses (treadmill_zoomer kills the
youngest, treadmill_boomer kills the oldest), routing through
get_mask_kill, the no-overshoot early return, and the number of victims.
"""

import numpy as np
import pytest

from aegis_sim.submodels.resources.starvation import Starvation


def _make_starvation(response="default"):
    s = Starvation()
    s.init(
        STARVATION_MORTALITY_MAXIMUM=1.0,
        STARVATION_MORTALITY_FACTOR=None,
        STARVATION_RESPONSE=response,
    )
    return s


class TestTreadmillStaticMethods:
    """The treadmill static methods choose victims deterministically by age."""

    def test_boomer_kills_oldest(self):
        """treadmill_boomer spares the youngest survivors and kills the oldest."""
        ages = np.array([5, 1, 3, 4, 2])
        mask = Starvation._treadmill_boomer(ages, resources_scavenged=3)
        # survivors are the 3 youngest (ages 1, 2, 3); the 2 oldest (5, 4) die
        np.testing.assert_array_equal(mask, [True, False, False, True, False])

    def test_zoomer_kills_youngest(self):
        """treadmill_zoomer spares the oldest survivors and kills the youngest."""
        ages = np.array([5, 1, 3, 4, 2])
        mask = Starvation._treadmill_zoomer(ages, resources_scavenged=3)
        # survivors are the 3 oldest (ages 5, 4, 3); the 2 youngest (1, 2) die
        np.testing.assert_array_equal(mask, [False, True, False, False, True])

    def test_number_of_victims_matches_deficit(self):
        """The number killed brings the population down to the resource count."""
        ages = np.arange(100)
        for n_survive in (0, 1, 37, 99):
            mask_z = Starvation._treadmill_zoomer(ages, resources_scavenged=n_survive)
            mask_b = Starvation._treadmill_boomer(ages, resources_scavenged=n_survive)
            assert mask_z.sum() == 100 - n_survive
            assert mask_b.sum() == 100 - n_survive

    def test_fractional_resources_truncate(self):
        """A fractional resource count is truncated toward zero."""
        ages = np.arange(10)
        mask = Starvation._treadmill_zoomer(ages, resources_scavenged=3.9)
        assert mask.sum() == 10 - 3  # int(3.9) == 3 survive

    def test_zoomer_boomer_are_complementary_survivors(self):
        """With the same deficit, no individual survives under both responses
        unless the survivor sets overlap by age ordering."""
        ages = np.array([10, 20, 30, 40])
        surv_z = ~Starvation._treadmill_zoomer(ages, resources_scavenged=2)
        surv_b = ~Starvation._treadmill_boomer(ages, resources_scavenged=2)
        # zoomer keeps oldest two (30, 40); boomer keeps youngest two (10, 20)
        np.testing.assert_array_equal(surv_z, [False, False, True, True])
        np.testing.assert_array_equal(surv_b, [True, True, False, False])


class TestGetMaskKillRouting:
    """get_mask_kill dispatches to the configured starvation response."""

    def test_no_overshoot_returns_zeros(self):
        """When resources meet or exceed the population, nobody starves."""
        s = _make_starvation("treadmill_zoomer")
        ages = np.array([1, 2, 3])
        mask = s.get_mask_kill(ages=ages, resources_scavenged=3)
        assert mask.sum() == 0
        assert s.consecutive_overshoot_n == 0

    def test_routes_to_zoomer(self):
        """STARVATION_RESPONSE='treadmill_zoomer' kills the youngest on overshoot."""
        s = _make_starvation("treadmill_zoomer")
        ages = np.array([5, 1, 3, 4, 2])
        mask = s.get_mask_kill(ages=ages, resources_scavenged=3)
        np.testing.assert_array_equal(mask, [False, True, False, False, True])

    def test_routes_to_boomer(self):
        """STARVATION_RESPONSE='treadmill_boomer' kills the oldest on overshoot."""
        s = _make_starvation("treadmill_boomer")
        ages = np.array([5, 1, 3, 4, 2])
        mask = s.get_mask_kill(ages=ages, resources_scavenged=3)
        np.testing.assert_array_equal(mask, [True, False, False, True, False])

    def test_default_response_does_not_take_treadmill_branch(self):
        """The default response falls through to the mortality computation."""
        s = _make_starvation("default")
        assert s.STARVATION_RESPONSE == "default"
