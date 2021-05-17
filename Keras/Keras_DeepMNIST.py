import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K 


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

