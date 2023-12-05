import math
from aegis.help import other


class Disease:
    """Disease manager

    Disease status:
        0 .. susceptible
        1 .. infected
        -1 .. dead
    """

    def __init__(self, BACKGROUND_INFECTIVITY, TRANSMISSIBILITY, RECOVERY_RATE, FATALITY_RATE):
        self.BACKGROUND_INFECTIVITY = BACKGROUND_INFECTIVITY
        self.TRANSMISSIBILITY = TRANSMISSIBILITY
        self.RECOVERY_RATE = RECOVERY_RATE
        self.FATALITY_RATE = FATALITY_RATE

    def infection_probability(self, infection_density):
        return self.BACKGROUND_INFECTIVITY - 0.5 + 1 / (1 + math.exp(-self.TRANSMISSIBILITY * infection_density))

    def __call__(self, population):
        """
        First try infecting susceptible.
        """
        probs = other.rng.random(len(population), dtype=float)

        # current status
        infected = population.disease == 1
        susceptible = population.disease == 0

        # compute infection probability
        infection_density = infected.sum() / len(population)
        infection_probability = self.infection_probability(infection_density=infection_density)

        # recoveries from old infections
        population.disease[infected & (probs < self.RECOVERY_RATE)] = 0

        # fatalities
        # overrides recoveries
        population.disease[infected & (probs < self.FATALITY_RATE)] = -1

        # new infections
        population.disease[susceptible & (probs < infection_probability)] = 1
