import os
from PIL import Image
import numpy as np

os.chdir("./data2")
files = os.listdir()
final = []
for i in range (10):

    im = Image.open(files[i])
    threshold = 252
    im=im.resize((28,28), Image.ANTIALIAS)
    im = im.point(lambda p: p > threshold and 255)
    na = np.array(im)
    img=(na > na.mean())*255
    np.place(na,na<255,1)
    np.place(na,na==255,0)
    fa=na.reshape([1,784])
    fa = list(fa)
    final.append(fa)


# print(final[3])
im.show(final[6][0].reshape(28,28))

