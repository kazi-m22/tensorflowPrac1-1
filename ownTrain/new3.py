import os.path as path
import glob2
from PIL import Image
import numpy as np
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


print(image_list[500])