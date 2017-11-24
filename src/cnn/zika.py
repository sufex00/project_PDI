import os
import numpy
import sys

sys.path.append('/usr/local/lib/python2.7/dist-packages')
#theano.config.openmp=True

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data

mypath = '../db/'

nfolds=6

size_img = 225 

input_shape= (1, size_img, size_img)

num_classes = 1 

sum_score = 0

size_train = 468

input_shape_autoencoder = 64

execfile('definition.py')

#(x_train, y_train), (x_test, y_test) = mnist.load_data()



def main():

    if(sys.argv[1] == 'train'):
	h = Train(input_shape)
    if(sys.argv[1] == 'predict'):
	p, matrix = Predict(x_test)
	a = 0
	for e in range(y_test.shape[0]):
		if y_test[e] == p[e]:
			a +=1
	print y_test
	print a
	print a/y_test.shape[0]
	print matrix
    if(sys.argv[1] == 'autoencoder'):
	autoencoder(input_shape_autoencoder)

if __name__ == '__main__':
    main()



# file output

