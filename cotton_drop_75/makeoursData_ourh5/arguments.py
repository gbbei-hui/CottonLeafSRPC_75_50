
import argparse

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--dataroot', default='../data_our_all/TrainPoint', help='path to dataset')
    parser.add_argument('--gpu_environ', default='0', type=str, metavar='DEVICE', help='GPU to use, ')
    parser.add_argument('--gpu_ids', default=[0], type=list, metavar='DEVICE', help='GPU to use, ')
    parser.add_argument('--niter', type=int, default=302, help='number of epochs to train for')  # default=201
    parser.add_argument('--batchSize', type=int, default=4, help='size')  # 设置 即num_epochs/batch_size的值是一个正整数。

    parser.add_argument('--pnum', type=int, default=2048, help=' 送到网络中数量/原始文件中点数  2048 4096')
    parser.add_argument('--point_scales_list', type=list, default=[2048, 1024, 512],
                        help='number of points in each scales')



    parser.add_argument('--crop_point_num', type=int, default=512, help='0 means do not use else use with this weight')
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--TxtLineA', default='./output_testpoint/WriteTxtALoss.txt', type=str, help='损失 画曲线的地方')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')

    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--cuda', type=bool, default=False, help='enables cuda')
    parser.add_argument('--D_choose', type=int, default=1, help='0 not use D-net,1 use D-net')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--num_scales', type=int, default=3, help='number of scales')

    parser.add_argument('--each_scales_size', type=int, default=1, help='each scales size')
    parser.add_argument('--wtl2', type=float, default=0.95, help='0 means do not use else use with this weight')
    parser.add_argument('--cropmethod', default='random_center', help='random|center|random_center')



    return parser.parse_args()