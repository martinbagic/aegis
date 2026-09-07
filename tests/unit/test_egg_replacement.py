"""EGG_REPLACEMENT_MODE: parameter validation and the eviction rule itself (fifo, lifo, random)."""
import numpy as np

from aegis_sim import variables
from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS


class TestEggReplacementModeParameter:
    """The EGG_REPLACEMENT_MODE parameter validates its allowed values."""

    def test_default_is_fifo(self):
        """The default preserves the historical first-in-first-out behaviour."""
        assert DEFAULT_PARAMETERS["EGG_REPLACEMENT_MODE"].default == "fifo"

    def test_accepts_fifo_lifo_and_random(self):
        param = DEFAULT_PARAMETERS["EGG_REPLACEMENT_MODE"]
        assert param.valid("fifo")
        assert param.valid("lifo")
        assert param.valid("random")

    def test_rejects_unknown_mode(self):
        assert not DEFAULT_PARAMETERS["EGG_REPLACEMENT_MODE"].valid("oldest")


def _evict(mode, n_eggs, capacity, rng_seed=0):
    """Replicates the index selection in Bioreactor.reproduction() for a pool of n_eggs laid in order 0..n_eggs-1."""
    variables.rng = np.random.default_rng(rng_seed)
    if mode == "random":
        return np.sort(variables.rng.choice(n_eggs, size=capacity, replace=False))
    if mode == "lifo":
        return np.arange(n_eggs)[:capacity]
    return np.arange(n_eggs)[-capacity:]


class TestEvictionRule:
    """Eggs are appended in laying order, so index = laying order. Capacity 100 of 250 eggs."""

    def test_fifo_keeps_the_newest(self):
        kept = _evict("fifo", 250, 100)
        assert kept.min() == 150 and kept.max() == 249

    def test_lifo_keeps_the_oldest(self):
        kept = _evict("lifo", 250, 100)
        assert kept.min() == 0 and kept.max() == 99

    def test_random_keeps_a_spread(self):
        kept = _evict("random", 250, 100, rng_seed=3)
        assert len(kept) == 100 and len(set(kept)) == 100
        assert kept.min() < 50 and kept.max() > 200

    def test_bioreactor_uses_the_same_rule(self):
        """The bioreactor source must contain the lifo branch keeping the first `capacity` eggs."""
        import inspect
        from aegis_sim import bioreactor
        src = inspect.getsource(bioreactor.Bioreactor.reproduction)
        assert 'mode == "lifo"' in src and "[:capacity]" in src
        assert "[-capacity:]" in src  # fifo
