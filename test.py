import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str, default='brats2019') # la, Pancreas, brats2019
parser.add_argument('--exp', type=str, default='BraTS2019_25/train') # LA_8; Pancreas_12; BraTS2019_25
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--speed', type=int, default=2)
parser.add_argument('-g', '--gpu', type=str,  default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
#from DiffVNet.cnn_vnet import VNet
from DiffVNet.cnn_vnet_proto import VNet
from utils import read_list, maybe_mkdir, test_all_case
from utils import config

config = config.Config(args.task)

if __name__ == '__main__':
    for epoch_num in range(70,71):
        snapshot_path = f'./logs1/{args.exp}/'
        ckpt_path = os.path.join(snapshot_path, f'ckpts/model_{epoch_num}.pth')
        if not os.path.exists(ckpt_path):
            continue

        stride_dict = {
            1: (18, 4),  # la
            2: (64, 64)  # brats2019
        }
        stride = stride_dict[args.speed]
        test_save_path = f'./logs1/{args.exp}/predictions/epoch={epoch_num}/'
        maybe_mkdir(test_save_path)
        print(snapshot_path)

        model = VNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=True
        ).cuda()

        with torch.no_grad():
            model.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)
            model.eval()
            print(f'load checkpoint from {ckpt_path}')
            test_all_case(
                args.task,
                model,
                read_list(args.split, task=args.task),
                num_classes=config.num_cls,
                patch_size=config.patch_size,
                stride_xy=stride[0],
                stride_z=stride[1],
                test_save_path=test_save_path
            )