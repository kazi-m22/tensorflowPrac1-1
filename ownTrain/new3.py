import os.path as path
import glob2
from PIL import Image
import numpy as np
image_list = []
imgExt = ("png","jpg","jpeg")

def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)


for filename in glob2.glob('data/*/**.*'):
    if((path.splitext(filename)[1][1:]) in imgExt):

        img=Image.open(filename)
        image_list.append(img)


image_list[500].show()