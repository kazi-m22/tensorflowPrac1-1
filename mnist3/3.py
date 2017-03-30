from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#
# x_test = mnist.test.images[1:2,:]
# img=x_test[0]

# sess= tf.Session()
# print("load graph")
# with gfile.FastGFile("./model/mn.pb",'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     sess.graph.as_default()
#     tf.import_graph_def(graph_def, name='')
# print("map variables")

x_test = mnist.test.images[:, :]
im = x_test

# x=tf.get_default_graph().get_tensor_by_name('input:0')
# y_test=tf.get_default_graph().get_tensor_by_name('output:0')
# answer = sess.run(y_test, feed_dict={x: im})
# print(answer.argmax())


# print(im.shape)
# plt.imshow(im, cmap=plt.get_cmap('gray_r'))
# plt.show()

print(x_test.shape)