# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:33:47 2017

@author: pedro_barros
"""

from os import listdir
from os.path import isfile, join
from random import shuffle

import sys

sys.path.append('/usr/local/lib/python2.7/dist-packages')


import cv2

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

from matplotlib  import pyplot as plt

size_img = 40

#   input_img = Input(shape=(1, 225, 225))  # adapt this if using `channels_first` image data format

input_img = Input(shape=(size_img*size_img,))
encoded = Dense(128*2, activation='relu')(input_img)
encoded = Dense(64*2, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
encoded = Dense(64*2, activation='relu')(encoded)
encoded = Dense(128*2, activation='relu')(encoded)
decoded = Dense(size_img*size_img, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')



#from keras.datasets import mnist
import numpy as np

#(x_train, _), (x_test, _) = mnist.load_data()



###################################################################################

mypath = '../db/'



input_shape = (1, size_img, size_img)

size_train = 468

x_train = []

y_train = []

x_test = []

y_test = []


onlyfiles = [f for f in listdir(mypath+'aedes/') if isfile(join(mypath+'aedes/', f))]

X = []
Y = []

aux = [f for f in listdir(mypath+'culex/') if isfile(join(mypath+'culex/', f))]

onlyfiles = onlyfiles + aux

shuffle(onlyfiles)


for f in onlyfiles:
    print f
    if 'culex' in f:
        img = cv2.imread(mypath+'culex/{}'.format(f))
        Y.append(0)
    if 'aedes' in f:
        img = cv2.imread(mypath+'aedes/{}'.format(f))
        Y.append(1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (threshName, threshMethod) = ("THRESH_BINARY", cv2.THRESH_BINARY)
    (T, thresh) = cv2.threshold(gray, 150, 255, threshMethod)
    X.append(cv2.resize(thresh, (size_img, size_img)))


X = np.array(X)
Y = np.array(Y)

X = X.reshape(X.shape[0], size_img * size_img).astype('float32')

x_train = X[:size_train]
x_train = x_train/255.

y_train = Y[:size_train]

x_test = X[size_train:]
x_test=x_test/255.


###################################################################################



from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=2000,
                batch_size=32,
                shuffle=True,
                validation_data=(x_test, x_test),
                verbose = 2,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(size_img, size_img))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(size_img, size_img))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

