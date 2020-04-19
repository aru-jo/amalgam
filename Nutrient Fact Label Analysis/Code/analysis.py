# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:50:52 2017

@author: Aravind

"""
import pandas
import matplotlib.pyplot as plot
import numpy as np
from sklearn.manifold import TSNE
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
 

names= ['Cals', 'Protein' ,'Carbohydrate ','Total Sugar' ,'Total Fat','Saturated Fat','Monounsaturated Fat','Polyunsaturated Fat'	,'DHA','EPA','Total  Dietary Fibre','Cholestrol','Sodium','	Iron','Potassium','Magnesium','Phosphorous','Folate','Lycopene','Thiamin','Riboflavin','Niacin','Vitamin A','Vitamin B12','Vitamin C','Vitamin D','Vitamin E','Alcohol','Class']
frame  = pandas.read_excel('union_final_1.xlsx',4)
arr = frame.values
X = arr[:,0:28] 
E = arr[:,0]
F = arr[:,4]
y = arr[:,28]
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
    k_fold = model_selection.KFold(n_splits = 100, random_state = seed)
    cv_res = model_selection.cross_val_score(mod,X,y,cv=k_fold,scoring=scoring)
    results.append(cv_res)
    names.append(na)
    msg = " %s : %f - %f " %(na,cv_res.mean(),cv_res.std())
    print(msg)
    
def showBoxPlot():
	figr = plot.figure()
	figr.suptitle("Comparing Algorithms - Final Data Set\n Cross Validation")
	ax = figr.add_subplot(111)	
	plot.boxplot(results)
	ax.set_xticklabels(names)
	plot.show()

def showHist():
	plot.title('Class Distribution')
	plot.xlabel('Class Label')
	plot.ylabel('Frequency')
	plot.hist([y],stacked="false",color=['b'])
	plot.show()

def scatterPlot():
	plot.title('Scatter Plot')
	plot.xlabel('Energy(kcal)/Total Fat(g)')
	plot.ylabel('Class Label')
	plot.scatter(E,y,label='Energy',color='r', marker= '*',s=1)
	plot.scatter(F,y,label='Fat',color='k', marker='x', s=1 )
	plot.legend()
	plot.show()
'''
def Tsne():
	tsne =  TSNE(n_components=2,random_state=0)
	twoD = tsne.fit_transform(X)
	#markers = ('s','d','o','^','v')
	#color_map={0:'red',1:'blue',2:'lightgreen',3:'purple',4:'cyan'}
	plot.figure()
	for idx,cl in enumerate(np.unique(twoD)):
		plot.scatter(x=twoD[cl,0],y=twoD[cl,1],label=cl)
	plot.show()
		#plot.scatter(x=twoD[cl,0],y=twoD[cl,1],c=color_map[idx],marker=markers[idx],label=cl)


Tsne()
'''

showBoxPlot()
showHist()
scatterPlot()












