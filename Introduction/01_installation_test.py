# import TensorFlow 
import tensorflow as tf 

sess = tf.Session()

hello= tf.constant("Hello :, RBB ")
print(sess.run(hello))


a = tf.constant(20)
b = tf.constant(21)

print(' a + b = {0}'.format(sess.run(a + b)))

sess.close()