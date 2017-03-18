from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


num=5000
x_train=mnist.train.images[:num,:]
y_train=mnist.train.labels[:num,:]


x = tf.placeholder(tf.float32, shape=[None, 784],name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10],name='y_')

w = tf.Variable(tf.zeros([784,10]),name='w')
b = tf.Variable(tf.zeros([10]),name='b')

y = tf.nn.softmax(tf.matmul(x,w) + b)

loss = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

for i in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys}, sess)

_w=w.eval(sess)
_b=b.eval(sess)