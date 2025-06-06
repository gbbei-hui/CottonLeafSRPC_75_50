import numpy as np
import torch

# 1.加载txt  -->  numpy  -->  tensor

def  loadtxt_my( txtfile):
    data = np.loadtxt(txtfile)
    #data = torch.tensor(data).float()  # torch.tensor(data)  # torch.float64,即double类型。

    # print(type(data))
    # print(data.shape)



    return data

if __name__ == '__main__':
    txtfile = './10000.txt'
    dsaafd = loadtxt_my(txtfile)

    print(type(dsaafd))
    print(dsaafd.shape)



