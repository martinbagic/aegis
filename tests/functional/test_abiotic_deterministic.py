"""Functional test: sim driven by the instant_deterministic abiotic cull.

Exercises the mortality_abiotic branch that calls Abiotic.get_mask_kill,
culling a fixed fraction of the living population at each period boundary.
"""

import pathlib
import yaml
import pytest

from aegis_sim import run
from tests.functional.conftest import test_experiment_path


def test_run_with_instant_deterministic_abiotic():
    config = {
        "STEPS_PER_SIMULATION": 120,
        "LOGGING_RATE": 120,
        "INITIAL_POPULATION_SIZE": 300,
        "ABIOTIC_HAZARD_SHAPE": "instant_deterministic",
        "ABIOTIC_HAZARD_AMPLITUDE": 0.5,
        "ABIOTIC_HAZARD_PERIOD": 30,
    }
    path = test_experiment_path / "test_abiotic_deterministic.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f)

    try:
        run(custom_config_path=path, pickle_path=None, overwrite=True, custom_input_params={})
    except Exception as e:
        pytest.fail(f"aegis_sim.run raised an exception: {e}")


def test_run_with_spare_oldest_cull():
    config = {
        "STEPS_PER_SIMULATION": 120,
        "LOGGING_RATE": 120,
        "INITIAL_POPULATION_SIZE": 300,
        "ABIOTIC_HAZARD_SHAPE": "instant_deterministic",
        "ABIOTIC_HAZARD_AMPLITUDE": 0.5,
        "ABIOTIC_HAZARD_PERIOD": 30,
        "ABIOTIC_CULL_SPARE_OLDEST": True,
    }
    path = test_experiment_path / "test_abiotic_spare_oldest.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f)

    try:
        run(custom_config_path=path, pickle_path=None, overwrite=True, custom_input_params={})
    except Exception as e:
        pytest.fail(f"aegis_sim.run raised an exception with spare_oldest: {e}")
