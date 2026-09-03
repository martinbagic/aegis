import numpy as np
from aegis_sim import constants
from aegis_sim import variables

from aegis_sim.submodels.genetics.composite.interpreter import Interpreter
from aegis_sim import parameterization
from aegis_sim.submodels.genetics import ploider


class CompositeArchitecture:
    """

    GUI
    - when pleiotropy is not needed;
    - it is quick, easy to analyze, delivers a diversity of phenotypes
    - every trait (surv repr muta neut) can be evolvable or not
    - if not evolvable, the value is set by !!!
    - if evolvable, it can be agespecific or age-independent
    - probability of a trait at each age is determined by a BITS_PER_LOCUS adjacent bits forming a "locus" / gene
    - the method by which these loci are converted into a phenotypic value is the Interpreter type

    """

    def __init__(self, BITS_PER_LOCUS, AGE_LIMIT, THRESHOLD, HEADSUP=-1, MATURATION_AGE=0):
        self.BITS_PER_LOCUS = BITS_PER_LOCUS
        # number of leading survival/reproduction loci forced to all-ones at init (None = no guarantee)
        self.headsup = (MATURATION_AGE + HEADSUP) if HEADSUP > -1 else None
        self.n_loci = sum(trait.length for trait in parameterization.traits.values())
        self.length = self.n_loci * BITS_PER_LOCUS
        self.AGE_LIMIT = AGE_LIMIT

        self.evolvable = [trait for trait in parameterization.traits.values() if trait.evolvable]

        self.interpreter = Interpreter(
            self.BITS_PER_LOCUS,
            THRESHOLD,
        )

    def get_number_of_bits(self):
        return ploider.ploider.y * self.n_loci * self.BITS_PER_LOCUS

    def get_shape(self):
        return (ploider.ploider.y, self.n_loci, self.BITS_PER_LOCUS)

    def init_genome_array(self, popsize):
        # TODO enable agespecific False
        array = variables.rng.random(size=(popsize, *self.get_shape()))

        for trait in parameterization.traits.values():
            array[:, :, trait.slice] = array[:, :, trait.slice] < trait.initgeno

        # HEADSUP guarantee: founders survive and reproduce through the first
        # MATURATION_AGE + HEADSUP ages regardless of initgeno; later ages stay random.
        if self.headsup is not None:
            for name in ("surv", "repr"):
                trait = parameterization.traits[name]
                if trait.evolvable and trait.length > 0:
                    n = min(self.headsup, trait.length)
                    array[:, :, trait.start : trait.start + n] = 1

        return array

    def compute(self, genomes):

        if genomes.shape[1] == 1:  # Do not calculate mean if genomes are haploid
            genomes = genomes[:, 0]
        else:
            genomes = ploider.ploider.diploid_to_haploid(genomes)

        interpretome = np.zeros(shape=(genomes.shape[0], genomes.shape[1]), dtype=np.float32)
        for trait in parameterization.traits.values():
            loci = genomes[:, trait.slice]  # fetch
            probs = self.interpreter.call(loci, trait.interpreter)  # interpret
            # self.diffuse(probs)
            interpretome[:, trait.slice] += probs  # add back

        return interpretome

    # def diffuse(self, probs):
    #     window_size = parametermanager.parameters.DIFFUSION_FACTOR * 2 + 1
    #     p = np.empty(shape=(probs.shape[0], probs.shape[1] + window_size - 1))
    #     p[:, :window_size] = np.repeat(probs[:, 0], window_size).reshape(-1, window_size)
    #     p[:, window_size - 1 :] = probs[:]
    #     diffusome = np.convolve(p[0], np.ones(window_size) / window_size, mode="valid")

    def get_map(self):
        pass
