import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data 

# create input object which reads data from MNIST datasets. Perform one-hot encoding to define the digit
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 

# using Interactie session makes it the default sessions so we do not need to pass sess
sess = tf.InteractiveSession()

# define placeholders for MNIST input data 
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# change the MNIST input data from a list of values to a 28 X 28 pixel X 1 grayscale value cube 
#   which the Convolution NN can use. 
x_image = tf.reshape(x, [-1, 28, 28,1], name="x_image")

# define helper functions to created weights and baises variables, and convolution, and pooling layers
# we are using RELU as our activiation fucntion. These must be initialized to a small positive number
# and with some noise so you don´t end up goint to zero when comparing diffs.. 
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name) 

# Convolution and Pooling - we do Convolution, and the pooling to control overfitting 
def conv2d(x, W, name=None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name=name)

# define layers in the NN

# 1st Convolution layers 
with tf.name_scope('Conv1'): # 32 features for each 5X5 patch the image 
    W_conv1 = weight_variable([5, 5, 1, 32], name="weight") #tamaño del filtro=5x5, 1=# channels; 32=cuántas features determinará cada filtro
    b_conv1 = bias_variable([32], name="bias")
    # do convolution on images, add bias and push through RELU activation
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, name="conv2d") + b_conv1, name="relu")
    # take results and run through max_pool
    h_pool1 = max_pool_2x2(h_conv1, name="pool")


# 2nd convolution layer 
with tf.name_scope('Conv2'):
    # process the 32 features from Convolution layer 1, in 5 X 5 patch. Return 64 features weights and biases 
    W_conv2 = weight_variable([5, 5, 32, 64], name='weight')
    b_conv2 = bias_variable([64], name='bias')
    # do convolution of the output of the 1st convolution layer. Pool results 
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, name='conv2') + b_conv2, name='relu')
    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('FC'):
    # fully connected layer 
    W_fc1 = weight_variable([7 * 7 * 64, 1024], name="weight")
    b_fc1 = bias_variable([1024], name="bias")

    # connect output of pooling layer 2 as input to full connected layer 
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="relu")


# dropout some neurons to reduce overfitting
keep_prob = tf.placeholder(tf.float32, name="keep_prob") # get dropout probability as a training input 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob )

with tf.name_scope("Readout"):
    # readout layer 
    W_fc2 = weight_variable([1024, 10], name="weight")
    b_fc2 = bias_variable([10], name="bias")

# define model 
#y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# loss measurement 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# loss optimization
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# what is correct 
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
# how accurate is it?
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize all of the variables 
sess.run(tf.global_variables_initializer())

# Train model 
import time 

# define number of steps and how often we display progress
num_steps = 3000
display_every = 100

# start timer 
start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_:batch[1], keep_prob: 0.5})

    # periodic status display
    if i%display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_:batch[1], keep_prob: 1.0
        })
        end_time = time.time()
        print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time-start_time, train_accuracy*100))

# display summary
# time to train 
end_time = time.time()
print("Total training time for {0} batches: {1:.2f} seconds".format(i+1, end_time-start_time))

# accuracy on test data 
print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})*100.0))

sess.close()


