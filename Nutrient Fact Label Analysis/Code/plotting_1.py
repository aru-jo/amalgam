


import itertools
import numpy as np
import pandas as pa 
import matplotlib.pyplot as plt 
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


frame  = pa.read_excel('union_final_1.xlsx',5)
frame2 = pa.read_excel('union_final_1.xlsx',6)
arr = frame.values
t_arr = frame2.values
X = arr[:,0:28] 
y = arr[:,28]
t_X = t_arr[:,0:28]
t_y = t_arr[:,28]
class_names =[0,1,2,3,4]
Classifier = tree.DecisionTreeClassifier()
Classifier = Classifier.fit(X,y)
y_pred = Classifier.predict(t_X)

print(accuracy_score(t_y,y_pred))

for i in t_y:
  for j in y_pred:
    print(i,j)



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(t_y, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()




