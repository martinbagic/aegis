"""Unit tests for the EGG_REPLACEMENT_MODE parameter definition."""

from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS


class TestEggReplacementModeParameter:
    """The EGG_REPLACEMENT_MODE parameter validates its allowed values."""

    def test_default_is_fifo(self):
        """The default preserves the historical first-in-first-out behaviour."""
        assert DEFAULT_PARAMETERS["EGG_REPLACEMENT_MODE"].default == "fifo"

    def test_accepts_fifo_and_random(self):
        """Both supported modes pass validation."""
        param = DEFAULT_PARAMETERS["EGG_REPLACEMENT_MODE"]
        assert param.valid("fifo")
        assert param.valid("random")

    def test_rejects_unknown_mode(self):
        """An unsupported mode fails validation."""
        assert not DEFAULT_PARAMETERS["EGG_REPLACEMENT_MODE"].valid("lifo")
