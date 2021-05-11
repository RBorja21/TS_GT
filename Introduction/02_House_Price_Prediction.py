#  Simple prediction of house prices based on house size 

import tensorflow as tf  
import numpy as np 
import math 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation  

# generation some house sizes between 1000 and 3500
num_hose = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=5300, size=num_hose)

# generate house prices from house size with a random noise added
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_hose)

# plot generated house and size 
plt.plot(house_size, house_price, "bx") # bx = blue x
plt.ylabel("Price")
plt.xlabel("Size")
plt.show() 


 


