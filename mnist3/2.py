from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


num=5000
x_train=mnist.train.images[:num,:]
y_train=mnist.train.labels[:num,:]


x = tf.placeholder(tf.float32, shape=[None, 784],name='input_m')
y_ = tf.placeholder(tf.float32, shape=[None, 10],name='y_')

w = tf.Variable(tf.zeros([784,10]),name='w')
b = tf.Variable(tf.zeros([10]),name='b')

y = tf.nn.softmax(tf.matmul(x,w) + b,name='output_m')

loss = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys}, sess)


_w=w.eval(sess)
_b=b.eval(sess)

sess.close()

# # Create new graph for exporting
g_2 = tf.Graph()
with g_2.as_default():
    x_2 = tf.placeholder("float", [None, 784], name="input")
    y_2 = tf.nn.softmax(tf.matmul(x_2, _w) + _b, name="output")

    sess_2 = tf.Session()
    sess_2.run(tf.global_variables_initializer())

    tf.train.write_graph(g_2.as_graph_def(), './model','mn.pb', as_text=False)


# x_test = mnist.test.images[1:2,:]
#
# answer = sess.run(y, feed_dict={x: x_test})
# print(answer.argmax())