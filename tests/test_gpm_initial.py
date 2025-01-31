# import pytest

import numpy as np

from aegis import pan

G_surv_initpheno = 0.1013
G_repr_initpheno = 0.2124
G_muta_initpheno = 0.377
AGE_LIMIT = 44

pan.init_minimal(
    custom_input_params={
        "PHENOMAP": {
            "MA, 10": [["surv", "agespec", [0.1]], ["repr", "agespec", [0.1]]],
        },
        "G_surv_initpheno": G_surv_initpheno,
        "G_repr_initpheno": G_repr_initpheno,
        "G_muta_initpheno": G_muta_initpheno,
        "AGE_LIMIT": AGE_LIMIT,
        "BITS_PER_LOCUS": 1,
    },
)


def test_call():
    from aegis.modules.setup.init import architecture
    from aegis.modules.genetics.architecture.gpm import GPM

    # from aegis.init import phenomap

    phenolist = [
        (0, "surv", 8, 0.07932046620388114),
        (1, "surv", 1, 0.07372414038566093),
        (2, "surv", 11, 0.12298855000652478),
        (3, "surv", 14, 0.04388531825897038),
        (4, "surv", 0, 0.05819249677812861),
        (5, "surv", 10, 0.11493884493689391),
        (6, "surv", 4, 0.038825459925726544),
        (7, "surv", 4, 0.023220460349031927),
        (8, "surv", 16, 0.06514584827631599),
        (9, "surv", 11, 0.016041944783693276),
        (0, "repr", 6, 0.1432999963019765),
        (1, "repr", 18, 0.11322365367123742),
        (2, "repr", 5, 0.14502911690362205),
        (3, "repr", 3, 0.08244829825124439),
        (4, "repr", 7, 0.02431899078904654),
        (5, "repr", 15, 0.020515279563935767),
        (6, "repr", 5, 0.0062288269810071035),
        (7, "repr", 19, 0.08792598603184054),
        (8, "repr", 6, 0.21877552130505284),
        (9, "repr", 19, 0.1482034870062023),
    ]

    gpm = GPM(AGE_LIMIT=AGE_LIMIT, phenolist=phenolist)
    interpretome = np.random.random(size=(1009, 111))
    phenodiff = gpm.phenodiff(vectors=interpretome, zeropheno=architecture.get_number_of_phenotypic_values())
    phenowhole = gpm.add_initial_values(phenodiff)

    for before, after in zip(phenodiff, phenowhole):
        assert before[0] + G_surv_initpheno == after[0]
        assert before[AGE_LIMIT] + G_repr_initpheno == after[AGE_LIMIT]  # no phenomap at this bit index
        assert before[AGE_LIMIT * 2] + G_muta_initpheno == after[AGE_LIMIT * 2]  # no phenomap at this bit index

    # for interpretomei, phenodiffi in zip(interpretome, phenodiff):

    #     # test 1
    #     assert phenodiffi[8] == interpretomei[0] * phenolist[0][3]

    #     # test 2
    #     assert phenodiffi[11] == interpretomei[2] * phenolist[2][3] + interpretomei[9] * phenolist[9][3]

    #     # test 3
    #     assert phenodiffi[AGE_LIMIT + 18] == interpretomei[1] * phenolist[10 + 1][3]

    #     # test 4
    #     assert (
    #         phenodiffi[AGE_LIMIT + 6]
    #         == interpretomei[0] * phenolist[10 + 0][3] + interpretomei[8] * phenolist[10 + 8][3]
    #     )
