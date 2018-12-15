''' 
Author : Aravind 
'''
# ML Comparison on data set - supervised class. 
# Predefined data set
# Iris data set

import pandas

import matplotlib.pyplot as plot

from sklearn import model_selection

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data"

names = ['sepl','sepw','petl','petw','class']

data_frame = pandas.read_csv(url,names= names )

array = data_frame.values

X = array[:,0:4]

y = array[:,4]

seed = 4

models = []

models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('LR',LogisticRegression()))
models.append(('KNN',KNeighborsClassifier()))



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

fig.suptitle('Comparing Algorithms - IRIS')

ax = fig.add_subplot(111)

plot.boxplot(results)

ax.set_xticklabels(names)

plot.show()


