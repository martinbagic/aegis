import logging


class Param:
    def __init__(
        self, name, domain, default, info, dtype, drange, inrange=lambda x: True
    ):

        self.name = name
        self.domain = domain
        self.default = default
        self.info = info
        self.dtype = dtype
        self.drange = drange
        self.inrange = inrange

    def convert(self, value):
        if value is None or value == "":
            return self.default
        elif self.dtype == bool:
            return (value == "True" or value == "true")
        else:
            return self.dtype(value)

    def valid(self, value):
        # Not valid if wrong data type
        if not isinstance(value, self.dtype):
            logging.error(
                f"Value {value} is not of valid type {self.dtype} but of type {type(value)}"
            )
            return False

        # Not valid if not in range
        if not self.inrange(value):
            return False

        # Valid
        return True


params = {
    "RANDOM_SEED_": Param(
        name="RANDOM_SEED_",
        domain="recording",
        default=None,
        info="If nothing is given, a random integer will be used as the seed; otherwise the given integer will be used as the seed",
        dtype=int,
        drange="{None, (-inf, inf)}",
        inrange=lambda x: True,
    ),
    "STAGES_PER_SIMULATION_": Param(
        name="STAGES_PER_SIMULATION_",
        domain="recording",
        default=100000,
        info="How many stages does the simulation run for?",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
    ),
    "LOGGING_RATE_": Param(
        name="LOGGING_RATE_",
        domain="recording",
        default=1000,
        info="Log every ?-th stage; 0 for no logging",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "PICKLE_RATE_": Param(
        name="PICKLE_RATE_",
        domain="recording",
        default=100000,
        info="Pickle population every ? stages; 0 for no pickles",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "SNAPSHOT_RATE_": Param(
        name="SNAPSHOT_RATE_",
        domain="recording",
        default=10000,
        info="Take a snapshot every ? stages; 0 for no snapshots",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "VISOR_RATE_": Param(
        name="VISOR_RATE_",
        domain="recording",
        default=1000,
        info="Take a visor snapshot every ? stages; 0 for no visor records",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "POPGENSTATS_RATE_": Param(
        name="POPGENSTATS_RATE_",
        domain="recording",
        default=1000,
        info="Record population genetic stats about the population every ? stages; 0 for no popgen stats",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "POPGENSTATS_SAMPLE_SIZE_": Param(
        name="POPGENSTATS_SAMPLE_SIZE_",
        domain="recording",
        default=100,
        info="Number of individuals to use when calculating population genetic metrics",
        dtype=int,
        drange="{0, [3, inf)}",
        inrange=lambda x: x == 0 or x >= 3,
    ),
    "ECOSYSTEM_NUMBER_": Param(
        name="ECOSYSTEM_NUMBER_",
        domain="ecology",
        default=1,
        info="Number of subpopulations",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
    ),
    "MAX_POPULATION_SIZE": Param(
        name="MAX_POPULATION_SIZE",
        domain="ecology",
        default=1000,
        info="Number of individuals in the population",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
    ),
    "OVERSHOOT_EVENT": Param(
        name="OVERSHOOT_EVENT",
        domain="ecology",
        default="starvation",
        info="Who dies when everyone is starving?",
        dtype=str,
        drange="{starvation, cliff, treadmill_random, treadmill_zoomer, treadmill_boomer}",
        inrange=lambda x: x
        in (
            "starvation",
            "cliff",
            "treadmill_random",
            "treadmill_zoomer",
            "treadmill_boomer",
        ),
    ),
    "CLIFF_SURVIVORSHIP": Param(
        name="CLIFF_SURVIVORSHIP",
        domain="ecology",
        default=None,
        info="What fraction of population survives after a cliff?; null if not applicable",
        dtype=float,
        drange="{None, (0,1)}",
        inrange=lambda x: x is None or (0 < x < 1),
    ),
    "STAGES_PER_SEASON": Param(
        name="STAGES_PER_SEASON",
        domain="ecology",
        default=0,
        info="How many stages does one season last; 0 for no seasons",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "MAX_LIFESPAN": Param(
        name="MAX_LIFESPAN",
        domain="genetics",
        default=50,
        info="Maximum lifespan",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
    ),
    "MATURATION_AGE": Param(
        name="MATURATION_AGE",
        domain="genetics",
        default=10,
        info="Age at which reproduction is possible",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
    ),
    "BITS_PER_LOCUS": Param(
        name="BITS_PER_LOCUS",
        domain="genetics",
        default=8,
        info="Number of bits that each locus has",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
    ),
    "HEADSUP": Param(
        name="HEADSUP",
        domain="initialization",
        default=-1,
        info="-1 if no preevolution, 0 for maturity guarantee, +x for headsup",
        dtype=int,
        drange="{-1, 0, [1, inf)}",
        inrange=lambda x: x in (-1, 0) or x >= 1,
    ),
    "REPRODUCTION_MODE": Param(
        name="REPRODUCTION_MODE",
        domain="genetics",
        default="asexual",
        info="Mode of reproduction",
        dtype=str,
        drange="{sexual, asexual, asexual_diploid}",
        inrange=lambda x: x in ("sexual", "asexual", "asexual_diploid"),
    ),
    "RECOMBINATION_RATE": Param(
        name="RECOMBINATION_RATE",
        domain="genetics",
        default=0,
        info="Rate of recombination; 0 if no recombination",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "MUTATION_RATIO": Param(
        name="MUTATION_RATIO",
        domain="genetics",
        default=0.1,
        info="Ratio of 0->1 mutations to 1->0 mutations",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "MUTATION_METHOD": Param(
        name="MUTATION_METHOD",
        domain="computation",
        default="by_bit",
        info="Mutate by XOR with a randomized bit matrix or generate random indices to mutate",
        dtype=str,
        drange="{by_bit, by_index}",
        inrange=lambda x: x in ("by_bit", "by_index"),
    ),
    "PHENOMAP_METHOD": Param(
        name="PHENOMAP_METHOD",
        domain="computation",
        default="by_loop",
        info="Non-vectorized, vectorized and null method of calculating phenotypic values",
        dtype=str,
        drange="{by_loop, by_dot, by_dummy}",
        inrange=lambda x: x in ("by_loop", "by_dot", "by_dummy"),
    ),
    "ENVIRONMENT_CHANGE_RATE": Param(
        name="ENVIRONMENT_CHANGE_RATE",
        domain="ecology",
        default=0,
        info="Environmental map changes every ? stages; if no environmental change",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "G_surv_evolvable": Param(
        name="G_surv_evolvable",
        domain="genetics",
        default=True,
        info="Is survival an evolvable trait?",
        dtype=bool,
        drange="",
        inrange=lambda x: True,
    ),
    "G_surv_agespecific": Param(
        name="G_surv_agespecific",
        domain="genetics",
        default=True,
        info="Is survival age-specific?",
        dtype=bool,
        drange="",
        inrange=lambda x: True,
    ),
    "G_surv_interpreter": Param(
        name="G_surv_interpreter",
        domain="genetics",
        default="binary",
        info="",
        dtype=str,
        drange="",
    ),
    "G_surv_lo": Param(
        name="G_surv_lo",
        domain="genetics",
        default=0,
        info="Minimum survival rate",
        dtype=float,
        drange="",
    ),
    "G_surv_hi": Param(
        name="G_surv_hi",
        domain="genetics",
        default=1,
        info="Maximum survival rate",
        dtype=float,
        drange="",
    ),
    "G_surv_initial": Param(
        name="G_surv_initial",
        domain="initialization",
        default=1,
        info="Initial survival rate",
        dtype=float,
        drange="",
    ),
    "G_repr_evolvable": Param(
        name="G_repr_evolvable",
        domain="genetics",
        default=True,
        info="Is fertility an evolvable trait?",
        dtype=bool,
        drange="",
    ),
    "G_repr_agespecific": Param(
        name="G_repr_agespecific",
        domain="genetics",
        default=True,
        info="Is fertility age-specific?",
        dtype=bool,
        drange="",
    ),
    "G_repr_interpreter": Param(
        name="G_repr_interpreter",
        domain="genetics",
        default="binary",
        info="",
        dtype=str,
        drange="",
    ),
    "G_repr_lo": Param(
        name="G_repr_lo",
        domain="genetics",
        default=0,
        info="Minimum fertility",
        dtype=float,
        drange="",
    ),
    "G_repr_hi": Param(
        name="G_repr_hi",
        domain="genetics",
        default=0.5,
        info="Maximum fertility",
        dtype=float,
        drange="",
    ),
    "G_repr_initial": Param(
        name="G_repr_initial",
        domain="initialization",
        default=1,
        info="Initial fertility rate",
        dtype=float,
        drange="",
    ),
    "G_neut_evolvable": Param(
        name="G_neut_evolvable",
        domain="genetics",
        default=False,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_neut_agespecific": Param(
        name="G_neut_agespecific",
        domain="genetics",
        default=None,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_neut_interpreter": Param(
        name="G_neut_interpreter",
        domain="genetics",
        default=None,
        info="",
        dtype=str,
        drange="",
    ),
    "G_neut_lo": Param(
        name="G_neut_lo",
        domain="genetics",
        default=None,
        info="",
        dtype=float,
        drange="",
    ),
    "G_neut_hi": Param(
        name="G_neut_hi",
        domain="genetics",
        default=None,
        info="",
        dtype=float,
        drange="",
    ),
    "G_neut_initial": Param(
        name="G_neut_initial",
        domain="initialization",
        default=1,
        info="",
        dtype=float,
        drange="",
    ),
    "G_muta_evolvable": Param(
        name="G_muta_evolvable",
        domain="genetics",
        default=False,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_muta_agespecific": Param(
        name="G_muta_agespecific",
        domain="genetics",
        default=None,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_muta_interpreter": Param(
        name="G_muta_interpreter",
        domain="genetics",
        default=None,
        info="",
        dtype=str,
        drange="",
    ),
    "G_muta_lo": Param(
        name="G_muta_lo",
        domain="genetics",
        default=None,
        info="",
        dtype=float,
        drange="",
    ),
    "G_muta_hi": Param(
        name="G_muta_hi",
        domain="genetics",
        default=None,
        info="",
        dtype=float,
        drange="",
    ),
    "G_muta_initial": Param(
        name="G_muta_initial",
        domain="initialization",
        default=0.001,
        info="Initial mutation rate",
        dtype=float,
        drange="",
    ),
    "PHENOMAP_SPECS": Param(
        name="PHENOMAP_SPECS",
        domain="genetics",
        default=[],
        info="",
        dtype=list,
        drange="",
    ),
    "NOTES": Param(
        name="NOTES",
        domain="recording",
        default=[],
        info="",
        dtype=list,
        drange="",
    ),
}
