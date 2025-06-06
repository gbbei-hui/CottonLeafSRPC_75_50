import os

import random
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import h5py
import A1_ReadDir
import A2_txt2tensor
import A3_tensor2oneh5
import data_utils as d_utils
import shapenet_part_loader
import numpy as np
import arguments

from shape_utils import random_occlude_pointcloud as crop_shape
from A3_FPS_Npoint2Npoint import sample_and_group as FPS





makeType = 'train'  #   'train'  or  'test'

wholeP= 2048        # 一个完整的点云
saveP = 2048-1536
dropP =wholeP - saveP




###### read data path
readdata = '../data_origin/' + makeType +'/'

##### save name  json
jsonnamefile =  './savename_' + makeType +'.txt'
f_write = open(jsonnamefile, "w")
namelist = []




opt = arguments.parse_args()
if opt.gpu_ids[0] >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_environ
    strTemp = 'cuda:' + str(opt.gpu_ids[0])
    strTemp = 'cuda:0'
    device = torch.device(strTemp) if torch.cuda.is_available() else torch.device('cpu')
else:
     device = torch.device('cpu')

device = torch.device('cpu')


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)


transforms = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
    ]
)



########################################
######   DataLoader  加载数据  ##############
###############################################

dset = shapenet_part_loader.PartDataset( dir_point= readdata,classification=True, class_choice=None, npoints=opt.pnum, split='train')
assert dset
dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize, shuffle=None,num_workers = int(opt.workers))

def num2str(count):
    if count <= 9:
        afterx = '000' + str(count)
    elif count >= 10 and count <= 99:
        afterx = '00' + str(count)
    elif count >= 100 and count <= 999:
        afterx = '0' + str(count)
    else:
        afterx = str(count)

    return afterx



def  saveBatchPoint(type,savepath,_namelist, data):

    if(os.path.exists(savepath) == False):
        os.makedirs(savepath)

    i = 0
    for onedata in data:
        Newdata = onedata.cpu().detach().numpy()
        str_index=_namelist[i]


        if type=='txt':
            np.savetxt(savepath +   str_index + '.txt', Newdata, fmt="%0.4f")
        elif type=='h5':
            # 写入h5
            file = h5py.File(savepath +   str_index + '.h5', 'w')  # 最原始文件夹
            file.create_dataset('data', data=Newdata[:,:3])
        else:
            print('类型出错')

        i = i +1




def  saveFPpoint(epoch,name, data):
    savepath = './output_train/'

    if len(data.shape) == 4:
        data = data.squeeze(1)

    Newdata = data[0].cpu().detach().numpy()
    np.savetxt(savepath + str(epoch) + name + '.txt', Newdata, fmt="%0.3f")




# Function to find nearby points to a reference point
def find_nearby_points(points, reference_point, num_neighbors=5):
    distances = torch.norm(points - reference_point, dim=1)
    nearest_indices = torch.topk(distances, num_neighbors + 1, largest=False).indices
    return points[nearest_indices[1:]]  # Exclude the reference point itself
def add_new_points(reference_point, nearby_points, num_new_points=3):
    mean_position = torch.mean(nearby_points, dim=0)
    direction_vector = mean_position - reference_point
    new_points = [reference_point + i * direction_vector for i in range(1, num_new_points + 1)]
    return torch.stack(new_points)

def add_bilinearP(points, num_neighbors=2, num_new_points=1):
    all_new_points = []
    for reference_point in points:
        nearby_points = find_nearby_points(points, reference_point, num_neighbors)
        new_points = add_new_points(reference_point, nearby_points, num_new_points)
        all_new_points.append(new_points)
    return torch.cat(all_new_points)


def add_noiseP(point_cloud, noise_scale=0.0031):
    """
    给点云添加噪声
    :param point_cloud: 输入的点云，形状为 (N, 3)，其中 N 是点的数量
    :param noise_scale: 噪声的比例，默认值为 0.01
    :return: 添加噪声后的点云
    """
    noise = torch.randn_like(point_cloud) * noise_scale
    noisy_point_cloud = point_cloud + noise
    return noisy_point_cloud

def addTorchNoise(Batchpoint_cloud):  # （B，512，3）  ---》 （B,1536,3）

    arraytorch = []
    for i in range(Batchpoint_cloud.shape[0]):

        noisy_p1 = add_bilinearP(Batchpoint_cloud[i],num_neighbors=2, num_new_points=1)
        noisy_p2 = add_bilinearP(Batchpoint_cloud[i], num_neighbors=3, num_new_points=1)
        noisy_p3 = add_bilinearP(Batchpoint_cloud[i], num_neighbors=4, num_new_points=1)
        noisy_p4 = add_bilinearP(Batchpoint_cloud[i], num_neighbors=5, num_new_points=1)  # 参考邻居点4个点 ，生成2个， 不会跳出整体框架，且可在周边生成（邻3生2， 会跳出）
        noisy_p5 = add_bilinearP(Batchpoint_cloud[i], num_neighbors=4, num_new_points=2)  # 参考邻居点4个点 ，生成2个， 不会跳出整体框架，且可在周边生成（邻3生2， 会跳出）

        # noisy_p6 = add_noiseP(Batchpoint_cloud[i], noise_scale=0.0015)   #   增加 少量边界点
        # noisy_p6 = torch.unsqueeze(noisy_p6, 0)
        # noisy_p6 = FPS(  noisy_p6 ,  int((noisy_p6.shape[1])/8)  )
        # noisy_p6 = torch.squeeze(noisy_p6, 0)


        onenoise=torch.cat([noisy_p1,noisy_p2,noisy_p3,noisy_p4,noisy_p5],0)
        onenoise=onenoise.unsqueeze(0)
        arraytorch.append(onenoise)

    all_noise=torch.cat(arraytorch, 0)
    end =torch.cat([all_noise,Batchpoint_cloud] ,1)
    end = FPS(end, saveP )


    return end



def splitpoint(gt):
    ##########################################################
    ####################################################
    # start_time = time.time()

    num_holes = 1
    crop_point_num = dropP
    context_point_num = dropP
    N = wholeP
    points = torch.squeeze(gt, 1)
    points = points.cpu()  # tensor  CPU 形式
    partials = []
    fine_gts, interm_gts = [], []
    N_partial_points = N - (crop_point_num * num_holes)
    centroids = np.asarray(
        [[1, 0, 0], [0, 0, 1], [1, 0, 1], [-1, 0, 0], [-1, 1, 0]])
    batch_size = gt.size()[0]
    for m in range(batch_size):
        # partial, fine_gt, interm_gt = crop_shape(
        partial, fine_gt = crop_shape(
            points[m],
            centroids=centroids,
            scales=[crop_point_num, (crop_point_num + context_point_num)],
            n_c=num_holes
        )

        if partial.shape[0] > N_partial_points:
            assert num_holes > 1
            # sampling without replacement
            choice = torch.randperm(partial.size(0))[:N_partial_points]
            partial = partial[choice]

        partials.append(partial)
        fine_gts.append(fine_gt)
        # interm_gts.append(interm_gt)

    gt_crop_dense = partials = torch.stack(partials).to(device)  # [B,  N-drop，3,]
    gt_drop = fine_gts = torch.stack(fine_gts).to(device)  # [B, 512, 3]
    # interm_gts = torch.stack(interm_gts).to(device)  # [B, 1024, 3]  # 暂时 不用
    gt = gt.to(device)
    return gt,gt_crop_dense,gt_drop



def savealldata(index,namelist,gt_crop_dense_512,gt_input,upinput,gt_drop_1536):


    batch_size = gt_input.size()[0]

    # 计算真实 的一般大小， 并保存
    gt = torch.cat([gt_crop_dense_512 ,gt_drop_1536],1)
    gt_half =FPS(gt, int((gt.shape[1])/2) )

    # txt
    partial_dense    = './saveData_txt/'+ makeType +'/partial_dense/02691156/'
    partial_input    = './saveData_txt/'+ makeType +'/partial_input/02691156/'
    partial_bilinear = './saveData_txt/'+ makeType +'/partial_bilinear/02691156/'
    drop_dense       = './saveData_txt/'+ makeType +'/drop_dense/02691156/'
    path_gt_half     = './saveData_txt/'+ makeType +'/gt_half/02691156/'
    path_gt          = './saveData_txt/'+ makeType +'/gt/02691156/'

    saveBatchPoint('txt',partial_dense, namelist, gt_crop_dense_512)
    saveBatchPoint('txt',partial_input, namelist, gt_input)
    saveBatchPoint('txt',partial_bilinear, namelist, upinput)
    saveBatchPoint('txt',drop_dense, namelist, gt_drop_1536)
    saveBatchPoint('txt',path_gt_half, namelist, gt_half)
    saveBatchPoint('txt',path_gt, namelist, gt)


    # h5 -- ours
    partial_dense    = './saveData_h5_our/'+ makeType +'/partial_dense/02691156/'
    partial_input    = './saveData_h5_our/'+ makeType +'/partial_input/02691156/'
    partial_bilinear = './saveData_h5_our/'+ makeType +'/partial_bilinear/02691156/'
    drop_dense       = './saveData_h5_our/'+ makeType +'/drop_dense/02691156/'
    path_gt_half     = './saveData_h5_our/'+ makeType +'/gt_half/02691156/'

    saveBatchPoint('h5',partial_dense, namelist, gt_crop_dense_512)
    saveBatchPoint('h5',partial_input, namelist, gt_input)
    saveBatchPoint('h5',partial_bilinear, namelist, upinput)
    saveBatchPoint('h5',drop_dense, namelist, gt_drop_1536)
    saveBatchPoint('h5',path_gt_half, namelist, gt_half)

    # h5 -- convention
    partial_input    = './saveData_h5_convention/'+ makeType +'/partial/02691156/'
    path_gt          = './saveData_h5_convention/'+ makeType +'/gt/02691156/'
    saveBatchPoint('h5',partial_input, namelist, gt_input)
    saveBatchPoint('h5',path_gt,namelist, gt)


    # saveBatchPoint('h5',path_gt, index* batch_size, gt)


if opt.D_choose == 1:

        for index, data in enumerate(dataloader, 0): #   取数据，直到取完

            namelist,gt = data     # 取数据
            gt = gt[:, :, :3]
            #######
            for onename in namelist:
                xxxxxx = "\"" + onename + "\","  # 去掉换行符 +
                f_write.write(xxxxxx + '\n')  # 只是加了引号， 写入txt中
                print(xxxxxx)


            # (1) data prepare
            ###########################
            gt,gt_crop_dense_1536,gt_drop_512 = splitpoint(gt)  #  # gt = gt_crop_dense + gt_drop
            gt_input = FPS(gt_crop_dense_1536,int(saveP/4)) # RandomSample(gt_crop_dense,512)
            upinput = addTorchNoise(gt_input)

            savealldata(index,namelist,gt_crop_dense_1536,gt_input,upinput,gt_drop_512)


            print(index)


f_write.close()

