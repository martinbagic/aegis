"""Functional tests for the HEADSUP initialization guarantee (composite architecture).

HEADSUP = h forces the first MATURATION_AGE + h survival and reproduction loci
to all-ones at initialization; later loci keep their initgeno draw.
"""

import numpy as np
import pytest
import yaml

from aegis_sim import run, submodels, parameterization
from aegis_sim.parameterization import parametermanager
from tests.functional.conftest import test_experiment_path


def _run(config, name):
    path = test_experiment_path / f"{name}.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f)
    run(custom_config_path=path, pickle_path=None, overwrite=True, custom_input_params={})


def test_headsup_forces_leading_loci_to_ones_and_leaves_the_rest_random():
    _run({"STEPS_PER_SIMULATION": 1, "LOGGING_RATE": 1, "INITIAL_POPULATION_SIZE": 200,
          "GENARCH_TYPE": "composite", "BITS_PER_LOCUS": 8, "MATURATION_AGE": 10,
          "HEADSUP": 5, "G_surv_initgeno": 0.0, "G_repr_initgeno": 0.0, "G_repr_evolvable": True},
         "test_headsup_5")
    arch = submodels.architect.architecture
    genomes = arch.init_genome_array(200)
    for name in ("surv", "repr"):
        t = parameterization.traits[name]
        head = genomes[:, :, t.start : t.start + 15]
        tail = genomes[:, :, t.start + 15 : t.end]
        assert head.all(), f"{name}: first MATURATION_AGE + HEADSUP loci must be all ones"
        assert not tail.any(), f"{name}: loci beyond the guarantee must keep initgeno = 0"


def test_headsup_minus_one_is_a_no_op():
    _run({"STEPS_PER_SIMULATION": 1, "LOGGING_RATE": 1, "INITIAL_POPULATION_SIZE": 200,
          "GENARCH_TYPE": "composite", "BITS_PER_LOCUS": 8, "HEADSUP": -1, "G_surv_initgeno": 0.0},
         "test_headsup_off")
    arch = submodels.architect.architecture
    genomes = arch.init_genome_array(200)
    t = parameterization.traits["surv"]
    assert not genomes[:, :, t.slice].any()


def test_headsup_zero_guarantees_exactly_maturation_age():
    _run({"STEPS_PER_SIMULATION": 1, "LOGGING_RATE": 1, "INITIAL_POPULATION_SIZE": 100,
          "GENARCH_TYPE": "composite", "BITS_PER_LOCUS": 4, "MATURATION_AGE": 7,
          "HEADSUP": 0, "G_surv_initgeno": 0.0},
         "test_headsup_0")
    arch = submodels.architect.architecture
    genomes = arch.init_genome_array(100)
    t = parameterization.traits["surv"]
    assert genomes[:, :, t.start : t.start + 7].all()
    assert not genomes[:, :, t.start + 7 : t.end].any()


def test_headsup_population_survives_from_zero_initgeno():
    """With initgeno 0 and no guarantee everyone dies at age 0; with HEADSUP the population persists."""
    _run({"STEPS_PER_SIMULATION": 60, "LOGGING_RATE": 60, "INITIAL_POPULATION_SIZE": 300,
          "GENARCH_TYPE": "composite", "BITS_PER_LOCUS": 8, "MATURATION_AGE": 10,
          "HEADSUP": 5, "G_surv_initgeno": 0.0},
         "test_headsup_survives")
