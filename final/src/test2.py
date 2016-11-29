#! /usr/bin/python3

import numpy as np

from time import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import v_measure_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score)

X = np.loadtxt('../data.txt')
Y = np.loadtxt('../label.txt')

X = np.nan_to_num(X)

kmeans = MiniBatchKMeans(n_clusters=2, batch_size=2000,
                         random_state=0)

for i in range(20):
    start_time = time()
    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    kmeans.fit(X_train, y_train)
    y_pred = kmeans.predict(X_test)

    print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
    print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
    print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))
    print("\ttime: %1.3f\n" % (time() - start_time))
