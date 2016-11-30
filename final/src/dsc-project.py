
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from time import time

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

import warnings
warnings.filterwarnings('ignore')


# In[2]:

from sklearn.utils import resample
from sklearn import preprocessing

X = np.loadtxt('../data.txt')
Y = np.loadtxt('../label.txt').astype(int)

X = np.nan_to_num(X)

Xp, Yp = resample(X, Y, n_samples = 5000)


Xp_scaled = preprocessing.scale(Xp)


# In[3]:

def trial(X, Y, method, name, scoring='f1'):
    print (name)
    start_time = time()
    method.fit(X, Y)
    y_pred = method.predict(X)
    print(classification_report(Y, y_pred))
    print ("\tDone in %.2f s" % (time() - start_time))
    print ("Cross validation...")
    start_time = time()
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    scores = cross_val_score(method, X, Y, cv=skf, scoring=scoring, n_jobs=-1)
    accu = cross_val_score(method, X, Y, cv=skf, n_jobs=-1)
    print ("\tAccuracy: %0.2f (+/- %0.2f)" % (accu.mean(), accu.std() * 2))
    print ("\tF1 score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print ("\tDone in %.2f s" % (time() - start_time))


# ## Visualization
#
# ### PCA and Kernel PCA

# In[4]:

from sklearn.decomposition import PCA, KernelPCA

kpca = KernelPCA(n_components = 2, kernel="rbf")
X_kpca = kpca.fit_transform(Xp)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(Xp)
reds = Yp == 0
blues = Yp == 1

plt.plot(X_kpca[reds, 0], X_kpca[reds, 1], "rx")
plt.plot(X_kpca[blues, 0], X_kpca[blues, 1], "bo")
plt.title('Kernel PCA')
plt.show()
plt.plot(X_pca[reds, 0], X_pca[reds, 1], "rx")
plt.plot(X_pca[blues, 0], X_pca[blues, 1], "bo")
plt.title('PCA')
plt.show()


# ## random guess

# In[5]:

y_pred = np.random.randint(0, 2, Y.shape[0])
scores = np.array([f1_score(Y, np.random.randint(0, 2, Y.shape[0])) for i in range(10)])
accu = np.array([accuracy_score(Y, np.random.randint(0, 2, Y.shape[0])) for i in range(10)])
print ("Accuracy: %0.2f (+/- %0.2f)" % (accu.mean(), accu.std() * 2))
print ("F1 score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# ## Linear methods

# In[6]:

from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression


# In[7]:

trial(X, Y, Perceptron(), 'Perceptron')


# In[8]:

trial(X, Y, LogisticRegression(), 'Log Reg')


# In[9]:

trial(X, Y, SGDClassifier(), 'SGD Clasifier')


# In[10]:

trial(X, Y, SGDClassifier(loss="log", penalty="l2"), 'SGD with log')


# In[11]:

pac = PassiveAggressiveClassifier(random_state=9,
                                  class_weight='balanced',
                                  n_jobs=-1,
                                  n_iter=9)
trial(X, Y, pac, 'Passive Aggresive Clasifier')


# ## Non - Linear models

# In[12]:

from sklearn import svm
import warnings
warnings.filterwarnings('ignore')


# In[13]:

# Subsample
trial(Xp, Yp, svm.SVC(), 'SVC')


# In[14]:

# Subsample
trial(Xp, Yp, svm.NuSVC(gamma=1e9), 'Nu SVC')


# ## Non linear Transformations

# In[15]:

from sklearn.ensemble import RandomTreesEmbedding, ExtraTreesClassifier

# use RandomTreesEmbedding to transform data
hasher = RandomTreesEmbedding(n_jobs=-1)
X_randomTrees = hasher.fit_transform(X)


# In[16]:

trial(X_randomTrees, Y, LogisticRegression(), 'Random Trees Embedding')


# In[17]:

trees = ExtraTreesClassifier()
trial(Xp, Yp, trees, 'trees')


# In[18]:

from sklearn.naive_bayes import BernoulliNB
nb = BernoulliNB()
trial(X_randomTrees, Y, nb, 'Naive bayes')
trial(X, Y, nb, 'Naive bayes')


# In[19]:

from sklearn.kernel_approximation import RBFSampler


# In[20]:

rbf_feature = RBFSampler(gamma=1, random_state=1)
X_rbf = rbf_feature.fit_transform(Xp)
trial(X_rbf, Yp, trees, 'RBF transformation')


# ## Manifold learning

# In[21]:

from sklearn import neighbors
nnc = neighbors.KNeighborsClassifier(n_neighbors = 1,
                                     weights='uniform',
                                     algorithm='kd_tree')
trial(Xp, Yp, nnc, 'K Neighbors')


# In[22]:

nnc = neighbors.KNeighborsClassifier(n_neighbors = 4,
                                     weights='distance',
                                     algorithm='auto')
trial(Xp, Yp, nnc, 'Nearest neigbors')


# In[23]:

from sklearn.neighbors import RadiusNeighborsClassifier

rnc = RadiusNeighborsClassifier(radius=100)
trial(Xp_scaled, Yp, rnc, 'Radius neighbors classiflier')


# ## Multi layer peceptron

# In[24]:

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='relu',
                    hidden_layer_sizes=(100,33), random_state=10, tol=1e-9,
                    max_iter=400)

trial(Xp, Yp, mlp, 'Multi layer perceptron')


# In[25]:

parameters = {'solver': ['lbfgs'],
              'alpha': 10.0 ** -np.arange(-1, 7),
              'hidden_layer_sizes': [(100,33)],
              'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'max_iter' : [400],
              'tol': [1e-5, 1e-9],
              'learning_rate': ['constant', 'invscaling', 'adaptive']
             }
mlp = MLPClassifier()
# cv = StratifiedShuffleSplit(n_splits=10)
skf = StratifiedKFold(n_splits=10, shuffle=True)
clf = GridSearchCV(mlp, parameters, cv=skf, n_jobs=-1, scoring='f1')
start_time = time()
# clf.fit(Xp, Yp)
#
# The best parameters are {'tol': 1e-05, 'activation': 'identity', 'alpha': 1.0, 'learning_rate': 'adaptive', 'hidden_layer_sizes': (100, 33), 'solver': 'lbfgs', 'max_iter': 400} with a score of 0.50
# Done in 11507.70 s

print("The best parameters are %s with a score of %0.2f"
      % (clf.best_params_, clf.best_score_))
print ("\tDone in %.2f s" % (time() - start_time))


# In[26]:

mlp = MLPClassifier(solver='lbfgs', alpha=1e-6,
                    hidden_layer_sizes=(100,33),
                    learning_rate='invscaling',
                    tol = 1e-5)

trial(Xp, Yp, mlp, 'Multi layer perceptron')


# In[27]:

parameters = {'n_neighbors': [1, 2, 4, 10, 20],
              'weights': ['uniform', 'distance']}

ncc = neighbors.KNeighborsClassifier()
# cv = StratifiedShuffleSplit(n_splits=10)
skf = StratifiedKFold(n_splits=10, shuffle=True)
clf = GridSearchCV(ncc, parameters, cv=skf, n_jobs=-1, scoring='f1')
start_time = time()
print ("start")
clf.fit(Xp, Yp)
print("The best parameters are %s with a score of %0.2f"
      % (clf.best_params_, clf.best_score_))
print ("\tDone in %.2f s" % (time() - start_time))


# In[41]:

parameters = {'n_estimators': [1024],
              'min_samples_split': [256],
              'class_weight': ['balanced', None],
              'max_depth': [None],
              'max_features': [5, None, 'sqrt']
             }

trees = ExtraTreesClassifier()
skf = StratifiedKFold(n_splits=10, shuffle=True)
clf = GridSearchCV(trees, parameters, cv=skf, n_jobs=-1, scoring='f1')
start_time = time()
'''start
The best parameters are {'class_weight': 'balanced_subsample', 'max_depth': None, 'n_estimators': 1024, 'min_samples_split': 256} with a score of 0.51
Done in 1698.70 s
'''

print ("start")
clf.fit(Xp, Yp)
print("The best parameters are %s with a score of %0.2f"
      % (clf.best_params_, clf.best_score_))
print ("\tDone in %.2f s" % (time() - start_time))


# In[36]:

trees = ExtraTreesClassifier(class_weight='balanced',
                             n_estimators=1024,
                             min_samples_split=256)

trial(Xp, Yp, trees, 'Extra tree classifier optimized')

