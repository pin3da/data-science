#! /usr/bin/python3

import numpy as np

from time import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

X = np.loadtxt('../data.txt')
Y = np.loadtxt('../label.txt')

X = np.nan_to_num(X)

kmeans = MiniBatchKMeans(n_clusters=2, batch_size=2000,
                         random_state=0)

print ("MiniBatchKMeans...")
start_time = time()
y_kmeans = kmeans.fit_predict(X)
print ("Done in {:.2f}s. V-measure: {:.4f}".format(
    time() - start_time,
    v_measure_score(Y, y_kmeans)))

print ("F1 score {:.4f}".format(f1_score(Y, y_kmeans, average='weighted')))


skf = StratifiedKFold(n_splits=10, shuffle=True)
scores = cross_val_score(kmeans, X, Y, cv=skf, scoring='f1_weighted')
print (scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
