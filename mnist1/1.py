from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import random as ran
import tensorflow as tf
import os
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.util import compat
np.set_printoptions(threshold=np.nan)


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def TRAIN_SIZE(num):
    print ('Total Training Images in Dataset = ' + str(mnist.train.images.shape))
    print ('--------------------------------------------------')
    x_train = mnist.train.images[:num,:]
    print ('x_train Examples Loaded = ' + str(x_train.shape))
    y_train = mnist.train.labels[:num,:]
    print ('y_train Examples Loaded = ' + str(y_train.shape))
    print('')
    return x_train, y_train

def TEST_SIZE(num):
    print ('Total Test Examples in Dataset = ' + str(mnist.test.images.shape))
    print ('--------------------------------------------------')
    x_test = mnist.test.images[:num,:]
    print ('x_test Examples Loaded = ' + str(x_test.shape))
    y_test = mnist.test.labels[:num,:]
    print ('y_test Examples Loaded = ' + str(y_test.shape))
    return x_test, y_test

def display_digit(num):
    print(y_train[num])
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28,28])
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

def display_mult_flat(start, stop):
    images = x_train[start].reshape([1,784])
    for i in range(start+1,stop):
        images = np.concatenate((images, x_train[i].reshape([1,784])))
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()

def display_compare(num):
    # THIS WILL LOAD ONE TRAINING EXAMPLE
    x_train = mnist.train.images[num,:].reshape(1,784)
    y_train = mnist.train.labels[num,:]
    # THIS GETS OUR LABEL AS A INTEGER
    label = y_train.argmax()
    # THIS GETS OUR PREDICTION AS A INTEGER
    prediction = sess.run(y, feed_dict={x: x_train}).argmax()
    plt.title('Prediction: %d Label: %d' % (prediction, label))
    plt.imshow(x_train.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
    plt.show()

x_train, y_train = TRAIN_SIZE(55000)
#
# display_digit(0)

# display_mult_flat(0,4)



x = tf.placeholder(tf.float32, shape=[None, 784],name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10],name='y_')

W = tf.Variable(tf.zeros([784,10]),name='w')
b = tf.Variable(tf.zeros([10]),name='b')

y = tf.nn.softmax(tf.matmul(x,W) + b)


# ta=np.array(sess.run(y, feed_dict={x: x_train}))
# sess.run(y, feed_dict={x: x_train})
# print(ta.shape)
# print(sess.run(tf.log(y)))

# nums=np.array(sess.run(y, feed_dict={x: x_train}))
#
# print(nums)
# print(sess.run(tf.zeros([4])))
# print(sess.run(tf.nn.softmax(tf.constant([0.1, 0.005, 2]))))
#
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),1))
# init = tf.global_variables_initializer()
#
#
#
# sess = tf.Session()
# sess.run(init)
# tf.train.write_graph(sess.graph_def, './sum', 'graph.pbtxt')
display_digit(100)

# x_train, y_train = TRAIN_SIZE(5500)
# x_test, y_test = TEST_SIZE(10000)
# LEARNING_RATE = 0.1
# TRAIN_STEPS = 2500
#
#
# training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# for i in range(TRAIN_STEPS+1):
#     sess.run(training, feed_dict={x: x_train, y_: y_train})
#     if i%100 == 0:
#         print('Training Step:' + str(i) + '  Accuracy =  ' + str(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})) + '  Loss = ' + str(sess.run(cross_entropy, {x: x_train, y_: y_train})))
#

# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     weight = sess.run(W)[:,i]
#     plt.title(i)
#     plt.imshow(weight.reshape([28,28]), cmap=plt.get_cmap('seismic'))
#     frame1 = plt.gca()
#     frame1.axes.get_xaxis().set_visible(False)
#     frame1.axes.get_yaxis().set_visible(False)
#
# plt.show()

# x_train, y_train = TRAIN_SIZE(1)
# display_digit(0)
#
# answer = sess.run(y, feed_dict={x: x_train})
# print(answer.argmax())



# # display_compare(ran.randint(0, 55000))
# tf.app.flags.DEFINE_integer('training_iteration', TRAIN_STEPS,'number of training iterations.')
# tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
# tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
# FLAGS = tf.app.flags.FLAGS
#
# export_path = os.path.join(compat.as_bytes('mo'),compat.as_bytes(str(FLAGS.model_version)))
# print ('Exporting trained model to', export_path)
#
# builder = saved_model_builder.SavedModelBuilder(export_path)
# builder.add_meta_graph_and_variables(
#       sess, [tag_constants.SERVING],
#       signature_def_map={
#            'predict_images':
#                prediction_signature,
#            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
#                classification_signature,
#       },
#       legacy_init_op=legacy_init_op)
# builder.save()