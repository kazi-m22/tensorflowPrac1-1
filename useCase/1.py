from PIL import Image
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.platform import gfile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sess= tf.Session()
print("load graph")
with gfile.FastGFile("./model/b2.pb",'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
print("map variables")


def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0],3)

im = Image.open("ek.jpg")
threshold = 200
im=im.resize((28,28), Image.ANTIALIAS)
im = im.point(lambda p: p > threshold and 255)

na=PIL2array(im)
na=na[:,:,0]

img=(na > na.mean())*255
np.place(na,na<255,1)
np.place(na,na==255,0)
# fa=na.flatten()
fa=na.reshape([1,784])
# print(fa)

x=tf.get_default_graph().get_tensor_by_name('input:0')
y_test=tf.get_default_graph().get_tensor_by_name('output:0')
answer = sess.run(y_test, feed_dict={x: fa})
print(np.argmax(answer))
# print(answer)
