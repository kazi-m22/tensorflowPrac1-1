from PIL import Image
import numpy as np
from numpy import  array
import tensorflow as tf
import os
from tensorflow.python.platform import gfile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# sess= tf.Session()
# print("load graph")
# with gfile.FastGFile("./model/mn.pb",'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     sess.graph.as_default()
#     tf.import_graph_def(graph_def, name='')
# print("map variables")

def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)
im = Image.open("test1.jpg")
f = []
 #location of images( i haad 0-5
path = "/Users/saumyashovanroy/PycharmProjects/tensorflowPrac1/ownTrain/data"
files = os.listdir()

size = len(files)
i = 0
j = 0
f = []
final =[]
label = []
label1 = []
while i< 10:

    path2 = path +'/'+ str(i)

    os.chdir(path2)
    print(path2)

    files = os.listdir()

    while j<120:

        im = Image.open(files[j])
        threshold = 200
        im=im.resize((28,28), Image.ANTIALIAS)
        im = im.point(lambda p: p > threshold and 255)

        na = np.array(im)
    # na=na[:,:,0]

        img=(na > na.mean())*255
        np.place(na,na<255,1)
        np.place(na,na==255,0)
    # fa=na.flatten()
        fa=na.reshape([1,784])
        fa = list(fa)
        fa2 = fa                      #keep i = 0 if this for only one number

        # final = final+fa2
        final = final + fa2
        j+=1

    f = final + f

    i+=1

n1=0
# print(f[901])
for l1 in range(0,10):
    for l2 in range(0,120):
        label.append(l1)
print(label[500])

# x=tf.get_default_graph().get_tensor_by_name('input:0')
# y_test=tf.get_default_graph().get_tensor_by_name('output:0')
# answer = sess.run(y_test, feed_dict={x: fa})
# print(np.argmax(answer))
