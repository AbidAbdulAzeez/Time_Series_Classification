
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

import numpy as np
import pandas as pd

def train_DT(X,y):
    clf = DecisionTreeClassifier()
    clf.fit(X,y)

def train_RF100(X,y):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X,y)

def train_SVM(X,y):
    clf = SVC(random_state=8, max_iter=1000)
    clf.fit(X,y)

def train_KNN(X,y):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X,y)

def train_SGD(X,y):
    clf = SGDClassifier(loss='squared_hinge', max_iter=2000)
    clf.fit(X,y)

def train_MLP(X,y):
    clf = MLPClassifier(hidden_layer_sizes=(64, 16, 4), random_state=8, max_iter = 300)
    clf.fit(X,y)

def train_LGBM(X,y):
    clf = LGBMClassifier()
    clf.fit(X,y)