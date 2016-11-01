#! /usr/bin/python3

import numpy as np
import pandas as pd

data = pd.read_csv('../data2/test.csv')
ans = (np.random.rand(data.shape[0]) < 0.7).astype('int64')
for i in ans:
    print (i)
