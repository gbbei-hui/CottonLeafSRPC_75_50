import torch.utils.data as data
import os
import os.path
import torch
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# dataset_path=os.path.abspath(os.path.join(BASE_DIR, '../data_FP_Folding_4096/DrawPoint/'))
#


###############  main   调用的
class PartDataset(data.Dataset):
    def __init__(self, dir_point, npoints=2500, classification=False, class_choice=None, split='train', normalize=True):
        self.npoints = npoints
        self.dir_point = dir_point
        self.classification = classification
        self.normalize = normalize

        if not class_choice is None:
            print('class_choice   必须是None')

        # dir_point = '../data_our_all/TrainPoint/02691156/points'
        fns = sorted(os.listdir(dir_point))

        self.datapath = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            filepath = os.path.join(dir_point, token + '.txt')   # .pts   .txt
            self.datapath.append(('Airplane',token, filepath))


    def __getitem__(self, index):
        #one fn  =  ('Airplane',token, filepath)
        fn = self.datapath[index]  #  所有文件，随机取一个文件 array
        name = fn[1]
        point_set = np.loadtxt(fn[2]).astype(np.float32)
        if self.normalize == True:
            point_set = self.pc_normalize(point_set)
        sdsafdsfds = point_set.shape[0]
        choice = np.random.choice(sdsafdsfds, self.npoints, replace=True)  # 2048是点数， 这里  xiu gai
        # resample
        point_set = point_set[choice, :]

        
        # To Pytorch
        point_set = torch.from_numpy(point_set)

        return name,point_set



    def __len__(self):
        return len(self.datapath)
       
    def pc_normalize(self, pc):
        """ pc: NxC, return NxC """
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)       # | 0/64 [00:00<?, ?it/s]451 Loss  drop512,whole: 0.17114077205769718 0.14060971373692155
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc


if __name__ == '__main__':
    dset = PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice=None, npoints=4096, split='train')
#    d = PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=False, class_choice=None, npoints=4096, split='test')
    print(len(dset))
    ps, cls = dset[10000]
    print(cls)
#    print(ps.size(), ps.type(), cls.size(), cls.type())
#    print(ps)
#    ps = ps.numpy()
#    np.savetxt('ps'+'.txt', ps, fmt = "%f %f %f")
