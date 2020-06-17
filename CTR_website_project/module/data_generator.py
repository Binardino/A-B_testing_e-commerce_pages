import scipy.stats as scs
import pandas as pd
import numpy as np

#function generating random data
def generate_data(n_A, n_B, p_A, p_B, days=None, control_label='A', test_label='B'):
    """Returns pandas df with random CTR data
    n_A (int) : sample size for control group
    n_B (int) : sample size for test group

    p_A (float) : CR for group A
    p_B (float) : CR for group B

    days (int) : duration of the test

    return df
    """

    data = []

    N = n_A + n_B

    #distribute events based on proportion of group size
    group_bern = scs.bernoulli(n_A / (n_A + n_B))

    #initiate bernoulli distributions from which to randomly sample
    A_bern = scs.bernoulli(p_A)
    B_bern = scs.bernoulli(p_B)

    for i in range(N):
        row = {}

        if days is not None:
            if type(days) == int:
                row['ts'] = i // (N//days)
            else:
                raise ValueError('Provide an int for days param')
        row['group'] = group_bern.rvs()

        if row['group'] == 0:
            row['converted'] = A_bern.rvs()
        else:
            row['converted'] = B_bern.rvs()

        data.append(row)
    
    df = pd.DataFrame(data)

    df['group'] = df['group'].apply(lambda x: control_label if x == 0 else test_label)

    return df