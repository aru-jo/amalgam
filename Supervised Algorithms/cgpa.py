# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:50:52 2017

@author: Aravind
"""
import pandas

import matplotlib.pyplot as plot

from sklearn import model_selection

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.tree import DecisionTreeClassifier
 


names= ['cgpa','cint','class']

frame  = pandas.read_csv('ip.csv',names = names)

arr = frame.values

X = arr[:,0:2]

y = arr[:,2]

models = [] 

models.append(('LR',LogisticRegression()))

models.append(('NB',GaussianNB()))

models.append(('LDA',LinearDiscriminantAnalysis()))

models.append(('CART',DecisionTreeClassifier()))

models.append(('KNN',KNeighborsClassifier()))

models.append(('SVM',SVC()))

seed = 7

scoring ='accuracy' 

results = [] 

names = [] 

for na,mod in models:
    
    k_fold = model_selection.KFold(n_splits = 10, random_state = seed)
    
    cv_res = model_selection.cross_val_score(mod,X,y,cv=k_fold,scoring=scoring)
    
    results.append(cv_res)
    
    names.append(na)
    
    msg = " %s : %f - %f " %(na,cv_res.mean(),cv_res.std())
    
    print(msg)
    
figr = plot.figure()

figr.suptitle("Comparing Algorithms - IP Mark")

ax = figr.add_subplot(111)

plot.boxplot(results)

ax.set_xticklabels(names)

plot.show()









