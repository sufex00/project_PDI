import os
import numpy

#theano.config.openmp=True

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data

mypath = '../../db/'

nfolds=3

size_img = 100

input_shape= (3, size_img, size_img)

num_classes = 10 

sum_score = 0

execfile('definition.py')

#(x_train, y_train), (x_test, y_test) = mnist.load_data()

size_train = 600


def main():
    #if(sys.argv[1] == 'train'):
    	Train(input_shape)
    #if(sys.argv[1] == 'predict'):
    #r = Predict(x_test)
if __name__ == '__main__':
    main()



# file output

