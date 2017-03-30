import numpy as np


# im = np.zeros((784))
# a=[]
# b=[]
# for i in range(0,1200):
#     b.append(i)
#     a.append(im)
#
# # print(a[0].shape)
# # c= np.column_stack((a,b))
# # print(c.shape)
#
# aa = np.array(a)
# bb = np.array(b)
#
#
#
#
#
# print(aa.shape)
# print(bb.shape)

def conver(nu):
    con_a=np.zeros(10)
    con_a[nu]=1
    return con_a

conver(1)