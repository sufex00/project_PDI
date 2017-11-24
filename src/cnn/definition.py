os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import keras
#import tensorflow as tf
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l2, l1
from keras import backend as K
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import fbeta_score
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
import time
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import cv2
import sys
import tqdm
from tqdm import *
K.set_image_dim_ordering('th')
from os import listdir
from os.path import isfile, join
from random import shuffle

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

execfile('zika_model.py')
execfile('preparation.py')
execfile('train.py')
execfile('predict.py')


