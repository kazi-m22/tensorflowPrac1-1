import os.path as path
import glob2
from PIL import Image
import numpy as np




def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])

im = Image.open('./data/0/1.jpg').convert("L")
# im=im.resize((512, 512), Image.ANTIALIAS)

threshold = 200
im=im.resize((28,28), Image.ANTIALIAS)
im = im.point(lambda p: p > threshold and 255)
# im.show()
#
na=PIL2array(im)
na=na[:,:]
#
img=(na > na.mean())*255
np.place(na,na<255,1)
np.place(na,na==255,0)
# # fa=na.flatten()
fa=na.reshape([1,784])
print(fa)