#! /usr/bin/python3

import numpy as np
import sys

if (len(sys.argv) < 3):
    print ('Usage:\n%s correct.txt test.txt' % sys.argv[0])
    exit(0)

correct = np.loadtxt(sys.argv[1], dtype='int64')
test = np.loadtxt(sys.argv[2], dtype='int64')

ok = np.sum(correct == test)
per = (ok / (1.0 + len(correct)))
print ('%d differences: %.3f %% per ok' % (len(correct) - ok, per))

TP = np.sum((test[correct == test]) == 1)
TN = np.sum((test[correct == test]) == 0)
FP = np.sum((test[correct != test]) == 1)
FN = np.sum((test[correct != test]) == 0)

assert((TP + TN + FP + FN) == len(test))

P = TP / (TP + FP)
R = TP / (TP + FN)
S = (2 * P * R) / (P + R)
print ('F1 score %.3lf' % S)
