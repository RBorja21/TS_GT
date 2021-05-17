import tensorflow as tf 
import numpy as np
import math 

from keras.models import Sequential
from keras.layers.core import Dense, Activation

# generation some house sizes between 1000 and 3500
num_hose = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=5300, size=num_hose)

# generate house prices from house size with a random noise added
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_hose)



# normalized values to prevent under/overflows.
def normalize(array):
    return (array - array.mean()) / array.std()

# define number of training samples, 0.7=70%
num_train_samples = math.floor(num_hose * 0.7)

# define training data 
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asarray(house_price[:num_train_samples])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# define test data 
test_house_size = np.asarray(house_size[num_train_samples:])
test_house_price = np.asarray(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price) 

# define the NN for doing Linear Regression
model = Sequential()
model.add(Dense(1, input_shape=(1,), init='uniform', activation='linear'))
model.compile(loss='mean_squared_error', optimizer='sgd') # Loss and optimiezer; sgd ~ descenso de gradiente stocástico

# fit/train the model 
model.fit(train_house_size_norm, train_price_norm, nb_epoch=300)

# note: fit cost values will be different because we did not use NN in original.
score = model.evaluate(test_house_price_norm, test_house_price_norm)
print("\nloss on test : {0}".format(score))