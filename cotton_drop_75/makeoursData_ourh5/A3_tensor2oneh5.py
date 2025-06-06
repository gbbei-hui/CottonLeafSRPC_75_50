import torch
import h5py

def  tensor_h5save(h5savefile,tensordata):
    # 二维度是要升维
    if  len(tensordata) == 2:  # 3维的话，写进txt中
        tensordata = torch.unsqueeze(tensordata, dim=0)  # 指定位置增加维度

    # 写入h5
    file = h5py.File(h5savefile, 'w')  # 最原始文件夹
    file.create_dataset('data', data=tensordata)

if __name__ == '__main__':
    pass





