import pandas as pd
import numpy as np
import json
import time
import copy
import pickle

from aegis.panconfiguration import pan
from aegis.modules.popgenstats import PopgenStats


class Recorder:
    """Data recorder

    Records data generated by the simulation.

    When thinking about recording additional data, consider that there are three recording methods:
        I. Snapshots (record data from the population at a specific stage)
        II. Flushes (collect data over time then flush)
        III. One-time records
    """

    # TODO add headers to csv's

    def __init__(self, ecosystem_id, MAX_LIFESPAN, gstruc_shape):
        # Define output paths and make necessary directories
        opath = pan.output_path / str(ecosystem_id)
        self.paths = {
            "BASE_DIR": opath,
            "snapshots_genotypes": opath / "snapshots" / "genotypes",
            "snapshots_phenotypes": opath / "snapshots" / "phenotypes",
            "snapshots_demography": opath / "snapshots" / "demography",
            "visor": opath / "visor",
            "visor_spectra": opath / "visor" / "spectra",
            "input_summary": opath,
            "output_summary": opath,
            "pickles": opath / "pickles",
            "popgen": opath / "popgen",
            "phenomap": opath,
        }
        for path in self.paths.values():
            path.mkdir(exist_ok=True, parents=True)

        self.MAX_LIFESPAN = MAX_LIFESPAN

        # Initialize collection
        self._collection = {
            # collected in ecosystem.reproduction
            "age_at_birth": [0] * MAX_LIFESPAN,
            # collected in ecosystem._kill
            "age_at_overshoot": [0] * MAX_LIFESPAN,
            "age_at_genetic": [0] * MAX_LIFESPAN,
            "age_at_season_shift": [0] * MAX_LIFESPAN,
            "age_at_end_of_sim": [0] * MAX_LIFESPAN,
            "age_at_environment": [0] * MAX_LIFESPAN,
            "age_at_disease": [0] * MAX_LIFESPAN,
            # collected in ecosystem.run_stage
            "cumulative_ages": [0] * MAX_LIFESPAN,
        }
        self.collection = copy.deepcopy(self._collection)

        # Needed for output summary
        self.extinct = False

        # PopgenStats
        self.popgenstats = PopgenStats()

        # Add headers
        for key in self._collection.keys():
            with open(self.paths["visor_spectra"] / f"{key}.csv", "ab") as f:
                array = np.arange(MAX_LIFESPAN)
                np.savetxt(f, [array], delimiter=",", fmt="%i")

        with open(self.paths["visor"] / "genotypes.csv", "ab") as f:
            array = np.arange(
                gstruc_shape[0] * gstruc_shape[1] * gstruc_shape[2]
            )  # (ploidy, length, bits_per_locus)
            np.savetxt(f, [array], delimiter=",", fmt="%i")

        with open(self.paths["visor"] / "phenotypes.csv", "ab") as f:
            array = np.arange(gstruc_shape[1])  # number of phenotypic values
            np.savetxt(f, [array], delimiter=",", fmt="%i")

    # ===============================
    # RECORDING METHOD I. (snapshots)
    # ===============================

    def record_visor(self, population):
        """Record data that is needed by visor."""
        if pan.skip(pan.VISOR_RATE_) or len(population) == 0:
            return

        # genotypes.csv | Record allele frequency
        with open(self.paths["visor"] / "genotypes.csv", "ab") as f:
            array = population.genomes.reshape(len(population), -1).mean(0)
            np.savetxt(f, [array], delimiter=",", fmt="%1.3e")

        # phenotypes.csv | Record median phenotype
        with open(self.paths["visor"] / "phenotypes.csv", "ab") as f:
            array = np.median(population.phenotypes, 0)
            np.savetxt(f, [array], delimiter=",", fmt="%1.3e")

        self.flush()

    def record_snapshots(self, population):
        """Record demographic, genetic and phenotypic data from the current population."""
        if pan.skip(pan.SNAPSHOT_RATE_) or len(population) == 0:
            return

        # genotypes
        df_gen = pd.DataFrame(np.array(population.genomes.reshape(len(population), -1)))
        df_gen.reset_index(drop=True, inplace=True)
        df_gen.columns = [str(c) for c in df_gen.columns]
        df_gen.to_feather(self.paths["snapshots_genotypes"] / f"{pan.stage}.feather")

        # phenotypes
        df_phe = pd.DataFrame(np.array(population.phenotypes))
        df_phe.reset_index(drop=True, inplace=True)
        df_phe.columns = [str(c) for c in df_phe.columns]
        df_phe.to_feather(self.paths["snapshots_phenotypes"] / f"{pan.stage}.feather")

        # demography
        dem_attrs = ["ages", "births", "birthdays"]
        demo = {attr: getattr(population, attr) for attr in dem_attrs}
        df_dem = pd.DataFrame(demo, columns=dem_attrs)
        df_dem.reset_index(drop=True, inplace=True)
        df_dem.to_feather(self.paths["snapshots_demography"] / f"{pan.stage}.feather")

    def record_popgenstats(self, genomes, mutation_rate_func):
        """Record population size in popgenstats, and record popgen statistics."""
        self.popgenstats.record_pop_size_history(genomes)

        if pan.skip(pan.POPGENSTATS_RATE_) or len(genomes) == 0:
            return

        mutation_rates = mutation_rate_func("muta")
        self.popgenstats.calc(genomes, mutation_rates)

        # Record simple statistics
        array = list(self.popgenstats.emit_simple().values())
        if None in array:
            return

        with open(self.paths["popgen"] / "simple.csv", "ab") as f:
            np.savetxt(f, [array], delimiter=",", fmt="%1.3e")

        # Record complex statistics
        complex_statistics = self.popgenstats.emit_complex()
        for key, array in complex_statistics.items():
            with open(self.paths["popgen"] / f"{key}.csv", "ab") as f:
                np.savetxt(f, [array], delimiter=",", fmt="%1.3e")

    def record_pickle(self, population):
        if (
            pan.skip(pan.PICKLE_RATE_) and not pan.stage == 1
        ):  # Also records the pickle before the first stage
            return

        with open(self.paths["pickles"] / str(pan.stage), "wb") as f:
            pickle.dump(population, f)

    # ==============================
    # RECORDING METHOD II. (flushes)
    # ==============================

    def collect(self, key, ages):
        """Add data into memory which will be recorded later."""
        self.collection[key] += np.bincount(ages, minlength=self.MAX_LIFESPAN)

    def flush(self):
        """Record data that has been collected over time."""
        # spectra/*.csv | Age distribution of various subpopulations (e.g. population that died of genetic causes)
        for key, val in self.collection.items():
            with open(self.paths["visor_spectra"] / f"{key}.csv", "ab") as f:
                array = np.array(val)
                np.savetxt(f, [array], delimiter=",", fmt="%i")

        # Reinitialize the collection
        self.collection = copy.deepcopy(self._collection)

    # =================================
    # RECORDING METHOD III. (record once)
    # =================================

    def record_phenomap(self, map_):
        with open(self.paths["phenomap"] / "phenomap.csv", "w") as f:
            np.savetxt(f, map_, delimiter=",", fmt="%f")

    def record_output_summary(self):
        summary = {
            "extinct": self.extinct,
            "random_seed": pan.random_seed,
            "time_start": pan.time_start,
            "time_end": time.time(),
            "jupyter_path": str(pan.output_path.absolute()),
        }
        with open(self.paths["output_summary"] / "output_summary.json", "w") as f:
            json.dump(summary, f, indent=4)

    def record_input_summary(self):
        summary = {
            # "extinct": self.extinct,
            "random_seed": pan.random_seed,
            "time_start": pan.time_start,
            # "time_end": time.time(),
            "jupyter_path": str(pan.output_path.absolute()),
        }
        with open(self.paths["input_summary"] / "input_summary.json", "w") as f:
            json.dump(summary, f, indent=4)

    # def record_jupyter_path(self):
    #     with open(pan.here / "help/paths.txt", "a") as f:
    #         f.write(str(pan.output_path.absolute()) + "\n")
