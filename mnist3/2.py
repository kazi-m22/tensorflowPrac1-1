from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


num=5000
train_steps=2500

x_train=mnist.train.images[:num,:]
y_train=mnist.train.labels[:num,:]


x = tf.placeholder(tf.float32, shape=[None, 784],name='input_m')
y_un = tf.placeholder(tf.float32, shape=[None, 10],name='y_un')

w = tf.Variable(tf.zeros([784,10]),name='w')
b = tf.Variable(tf.zeros([10]),name='b')

y = tf.nn.softmax(tf.matmul(x,w) + b,name='output_known')

loss = -tf.reduce_sum(y_un * tf.log(y))

trainer= tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    trainer.run({x: batch_xs, y_un: batch_ys}, sess)



w_value=w.eval(sess)
b_value=b.eval(sess)

sess.close()

# # Create new graph for exporting
graph_model = tf.Graph()
with graph_model.as_default():
    x_model = tf.placeholder("float", [None, 784], name="input")
    y_model = tf.nn.softmax(tf.matmul(x_model, w_value) + b_value, name="output")

    sess_model = tf.Session()
    sess_model.run(tf.global_variables_initializer())

    tf.train.write_graph(graph_model.as_graph_def(), './model','mn.pb', as_text=False)


# x_test = mnist.test.images[1:2,:]
#
# answer = sess.run(y, feed_dict={x: x_test})
# print(answer.argmax())