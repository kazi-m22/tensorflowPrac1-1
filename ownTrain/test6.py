import numpy as np

im = np.zeros((784))
a=[]
b=[]
for i in range(0,1200):
    b.append(i)
    a.append(im)


c= np.column_stack((a,b))
print(c.shape)