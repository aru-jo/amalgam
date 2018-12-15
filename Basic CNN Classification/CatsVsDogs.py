import cv2
import os
from random import shuffle
from tqdm import tqdm
import numpy as np
train_dir = '/Users/aravind/Desktop/Projects/dogsvscats/train'
test_dir =  '/Users/aravind/Desktop/Projects/dogsvscats/test'
img_size = 50

lr = 1e-3
model_name ='dogsvscats-{}-{}'.format(lr,'6conv-basic-video')

def ret_label(img):
    lab = img.split('.')[-3]
    if lab == 'dog':
        return [0,1]
    elif lab == 'cat':
        return [1,0]    

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

def test_data():
    ted = []
    for img in tqdm(os.listdir(test_dir)):
        img_no = img.split('.')[0]
        path = os.path.join(test_dir,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(img_size,img_size))
        ted.append([np.array(img),img_no])
    np.save('test_data.npy',ted)
    return ted

train_data = create_train_data()

import tflearn

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(model_name)):
    model.load(model_name)
    print('model-loaded')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,img_size,img_size,1)
Y = [i[1] for i in train]

test_X = np.array([i[0] for i in test]).reshape(-1,img_size,img_size,1)
test_Y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_X}, {'targets': test_Y}), snapshot_step=500, show_metric=True, run_id=model_name)

model.save(model_name)

import matplotlib.pyplot as plt

#test_data = test_data()
test_data = np.load('test_data.npy')

fig=plt.figure()

for num,data in enumerate(test_data[:18]):
    # cat: [1,0]
    # dog: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,6,num+1)
    orig = img_data
    data = img_data.reshape(img_size,img_size,1)
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label='Dog'
    else: str_label='Cat'
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
