import numpy as np
import tensorflow as tf


sess = tf.Session()

a=tf.Variable(np.array([[1,2,3],[5,6,7]]))
sess.run(tf.global_variables_initializer())
# print(sess.run(a))
print(sess.run(tf.reduce_sum(a)))