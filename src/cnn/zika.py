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

nfolds=10

size_img = 225 

input_shape= (3, size_img, size_img)

num_classes = 1 

sum_score = 0

size_train = 468

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
	print a
	print matrix

if __name__ == '__main__':
    main()



# file output

