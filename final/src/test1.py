#! /usr/bin/python3

import numpy as np
import pandas as pd

data = pd.read_csv('../data2/test.csv')
part = data['contest_participation_count_30_days'].as_matrix()
ans = (np.random.rand(data.shape[0]) < 0.8).astype('int64')
ids = part > 0

for i in range(data.shape[0]):
    if (ids[i]):
        ans[i] = np.random.rand() < 0.9

for i in ans:
   print (i)
