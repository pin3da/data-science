#! /usr/bin/python3

import numpy as np
import pandas as pd

data = pd.read_csv('../data2/train.csv')
test = pd.read_csv('../data2/test.csv')
obs = (data.opened == 1).astype('int64').as_matrix()
p = (np.sum(obs) / obs.shape[0])

ans = (np.random.rand(test.shape[0]) > p).astype('int64')
for i in ans:
    print (i)
