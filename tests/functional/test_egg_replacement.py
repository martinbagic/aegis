"""Functional test: oviparous sim under both egg-replacement modes.

With a small egg carrying capacity and a positive incubation period, eggs
accumulate beyond the cap and the eviction branch fires. Both 'fifo' and
'random' modes must drive a full simulation to completion.
"""

import pathlib
import yaml
import pytest

from aegis_sim import run
from tests.functional.conftest import test_experiment_path


@pytest.mark.parametrize("mode", ["fifo", "random"])
def test_run_with_egg_replacement_mode(mode):
    config = {
        "STEPS_PER_SIMULATION": 200,
        "LOGGING_RATE": 200,
        "INITIAL_POPULATION_SIZE": 200,
        "INCUBATION_PERIOD": 20,
        "CARRYING_CAPACITY_EGGS": 10,
        "EGG_REPLACEMENT_MODE": mode,
    }
    path = test_experiment_path / f"test_egg_replacement_{mode}.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f)

    try:
        run(custom_config_path=path, pickle_path=None, overwrite=True, custom_input_params={})
    except Exception as e:
        pytest.fail(f"aegis_sim.run raised an exception for mode={mode}: {e}")
