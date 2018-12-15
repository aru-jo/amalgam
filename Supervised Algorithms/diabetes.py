#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:08:12 2017

@author: aravind

"""
#comparing algos using sci-kit learn


import pandas

import matplotlib.pyplot as plot 

from sklearn import model_selection

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

#loading sample data set

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"

#feature set

names = ['preg','plas','pres','skin','test','mass','pedi','age','class']

# creating a frame for csv values

data_frame = pandas.read_csv(url, names= names)

# reading values

val_array = data_frame.values

X = val_array[:,0:8]

y = val_array[:,8]

# cross validation 

seed = 7

# prepare models

models = [] 

models.append(('LR',LogisticRegression()))
models.append(('NB',GaussianNB()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('SVM',SVC()))

#EVALUATE MODEL

results = [] 

names = [] 

scoring = 'accuracy' 

for name, model in models:
    
    k_fold = model_selection.KFold(n_splits = 10, random_state = seed)

    cv_res = model_selection.cross_val_score(model,X,y,cv=k_fold,scoring=scoring)
    
    results.append(cv_res)
    
    names.append(name)
    
    msg = "%s: %f (%f)" % (name,cv_res.mean(),cv_res.std())
    
    print(msg)

#box plot algorithm comparison

fig = plot.figure()

fig.suptitle('Comparing Algorithms - PIMA Indian')

ax = fig.add_subplot(111)

plot.boxplot(results)

ax.set_xticklabels(names)

plot.show()



    







