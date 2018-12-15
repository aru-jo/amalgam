
# coding: utf-8

# In[28]:

#preprocessing

#imports

import cv2
import os
from random import shuffle
from tqdm import tqdm
import numpy as np

#directories containing train,test images

train_dir = "/Users/aravind/Desktop/dogsvscats/train"
test_dir =  "/Users/aravind/Desktop/dogsvscats/test"

#size of the image - resized

img_size = 50

#learning rate

lr = 1e-3

#0.001

model_name ='dvc-{}-{}'.format(lr,'2conv-basic')

# model

# conv basic 2 layered conv neural net


# In[29]:

def ret_label(img):
    lab = img.split('.')[-3]
    if lab == 'dog':
        return [0,1]
    elif lab == 'cat':
        return [1,0]    


# In[30]:

def create_train_data():
    td = []
    for img in tqdm(os.listdir(train_dir)):
        label = ret_label(img)
        path = os.path.join(train_dir,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(img_size,img_size))
        td.append([np.array(img),np.array(label)])
    shuffle(td)
    np.save('train_data.npy',td)
    return td


# In[31]:

def test_data():
    ted = []
    for img in tdqm(os.listdir(test_dir)):
        img_no = img.split('.')[0]
        path = os.path.join(test_dir,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(img_size,img_size))
        ted.append([np.array(img),img_no])
    np.save('test_data.npy',ted)
    return ted

        


# In[32]:

train_data = create_train_data()


# In[ ]:



