import pandas as pa 

import matplotlib.pyplot as plt 
 
from sklearn import tree

frame  = pa.read_excel('union_final_2.xlsx',5)

frame2 = pa.read_excel('union_final_2.xlsx',6)

arr = frame.values

t_arr = frame2.values

X = arr[:,0:28] 

y = arr[:,28]

t_X = t_arr[:,0:28]

t_y = t_arr[:,28]

Classifier = tree.DecisionTreeClassifier()

Classifier = Classifier.fit(X,y)

Classifier = Classifier.predict(t_X)

print(Classifier)

print(t_y)









