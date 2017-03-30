import os.path as path
import glob2
from PIL import Image
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

image_list = []
imgExt = ("png","jpg","jpeg")

def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])


for filename in glob2.glob('data/*/**.*'):
    if((path.splitext(filename)[1][1:]) in imgExt):

        im = Image.open(filename)
        im = im.resize((28, 28), Image.ANTIALIAS)

        na = PIL2array(im)
        # na = na[:, :]

        img = (na > na.mean()) * 255
        np.place(na, na < 255, 1)
        np.place(na, na == 255, 0)
        # fa=na.flatten()
        fa = na.reshape([1, 784])
        image_list.append(fa)


# print(image_list[500])

label = []

for l1 in range(0,10):
    for l2 in range(0,120):
        label.append(l1)
# print(label[500])


num=5000
train_steps=2500

x_train=image_list[:1200,:]
y_train=label[:1200,:]

x = tf.placeholder(tf.float32, shape=[None, 784],name='input_m')
y_un = tf.placeholder(tf.float32, shape=[None, 10],name='y_un')

w = tf.Variable(tf.zeros([784,10]),name='w')
b = tf.Variable(tf.zeros([10]),name='b')

y = tf.nn.softmax(tf.matmul(x,w) + b,name='output_known')

loss = -tf.reduce_sum(y_un * tf.log(y))
trainer= tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(train_steps):
    trainer.run({x: x_train, y_un: y_train}, sess)

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

