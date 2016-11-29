#! /usr/bin/python3

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

data = pd.read_csv('../train.csv')
Y = data['opened']
X = data.columns.tolist()
X.remove('opened')
X.remove('open_time')
X = data[X]

Y = Y.as_matrix()
X = X.as_matrix()

np.savetxt('../data.txt', X)
np.savetxt('../label.txt', Y)
