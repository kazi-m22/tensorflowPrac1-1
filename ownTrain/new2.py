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
        final.append(im)
        j+=1

    f = final + f

    i+=1

n1=0


# mm=Image.open(f[500])
im[1199].show()