import os
import numpy
import sys

sys.path.append('/usr/loca/bin/python2.7/dist-packages')
#theano.config.openmp=True

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data

mypath = '../db/'

nfolds=3

size_img = 300

input_shape= (3, size_img, size_img)

num_classes = 10 

sum_score = 0

size_train = 440

execfile('definition.py')

#(x_train, y_train), (x_test, y_test) = mnist.load_data()



def main():
    #if(sys.argv[1] == 'train'):
	h = Train(input_shape)
    #if(sys.argv[1] == 'predict'):
	r = Predict(x_test)
if __name__ == '__main__':
    main()



# file output

