import numpy as np

class MatingManager:
    def __init__(self):
        pass

    def pair_up_polygamously(self, sexes, ages=None, preference="none"):
        """
        Return indices of reproducing, sex-sorted individuals.
        Make sure no same-sex fertilization is happening.

        When `preference` is 'oldest' or 'youngest' (and `ages` is given), maters are
        paired age-assortatively: both sexes are sorted by age and paired in order, so the
        oldest (or youngest) male is coupled with the oldest (or youngest) female. Otherwise
        pairing is random.
        """

        indices_male = (sexes == 0).nonzero()[0]
        indices_female = (sexes == 1).nonzero()[0]

        # Compute number of pairs
        n_males = len(indices_male)
        n_females = len(indices_female)
        n_pairs = min(n_males, n_females)

        if preference in ("oldest", "youngest") and ages is not None:
            male_order = np.argsort(ages[indices_male], kind="stable")
            female_order = np.argsort(ages[indices_female], kind="stable")
            if preference == "oldest":
                male_order = male_order[::-1]
                female_order = female_order[::-1]
            indices_male = indices_male[male_order]
            indices_female = indices_female[female_order]
        else:
            # Shuffle
            np.random.shuffle(indices_male)
            np.random.shuffle(indices_female)

        # Pair up
        males = indices_male[:n_pairs]
        females = indices_female[:n_pairs]

        return males, females

    def pair_up_monogamously(self, sexes):
        return
