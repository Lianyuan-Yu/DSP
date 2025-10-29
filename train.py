import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys, logging
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str, default='brats2019') # la, brats2019
parser.add_argument('--exp', type=str, default='BraTS2019_25/train') # LA_8; BraTS2019_25
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_0.1')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_0.1')
parser.add_argument('-se', '--split_eval', type=str, default='test')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True)
parser.add_argument('-ep', '--max_epoch', type=int, default=350)
parser.add_argument('--tl', type=int, default=70)
parser.add_argument('--gamma1', type=int, default=0.01)
parser.add_argument('--gamma2', type=int, default=0.7)
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--unsup_loss', type=str, default='w_ce+dice')
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--mu', type=float, default=2.0) #2.0
parser.add_argument('-s', '--ema_w', type=float, default=0.99)
parser.add_argument('-r', '--mu_rampup', action='store_true', default=True)
parser.add_argument('-cr', '--rampup_epoch', type=float, default=None)
parser.add_argument('--speed', type=int, default=0)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from DiffVNet.diff_vnet import DiffVNet1
from DiffVNet.cnn_vnet_proto import VNet
from utils import maybe_mkdir, get_lr, fetch_data, GaussianSmoothing, seed_worker, poly_lr, sigmoid_rampup
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss
from data.data_loaders import DatasetAllTasks
from utils.config import Config
from data.StrongAug import get_StrongAug, ToTensor, CenterCrop
from data.brats2019 import BraTS2019, RandomCrop, RandomRotFlip, ToTensor

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config = Config(args.task)

def get_current_mu(epoch):
    if args.mu_rampup:

        if args.rampup_epoch is None:
            args.rampup_epoch = args.max_epoch
        return args.mu * sigmoid_rampup(epoch, args.rampup_epoch)
    else:
        return args.mu


def make_loss_function(name, weight=None):
    if name == 'ce':
        return RobustCrossEntropyLoss()
    elif name == 'wce':
        return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice':
        return DC_and_CE_loss()
    elif name == 'wce+dice':
        return DC_and_CE_loss(w_ce=weight)
    elif name == 'w_ce+dice':
        return DC_and_CE_loss(w_dc=weight, w_ce=weight)
    else:
        raise ValueError(name)

def rec_loss(x1, x2, epsilon=1e-8):
    Df = 1 - (x1 - x2)

    Df = torch.clamp(Df, min=epsilon)

    loss = -Df * torch.log(Df)

    return torch.mean(torch.abs(loss))

def make_loader(split, dst_cls=DatasetAllTasks, repeat=None, is_training=True, unlabeled=False, task="", transforms_tr=None):
    if is_training:
        dst = dst_cls(
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            transform=transforms_tr,
            task=task,
            num_cls=config.num_cls
        )
        return DataLoader(
            dst,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            drop_last=True
        )
    else:
        dst = dst_cls(
            split=split,
            is_val=True,
            task=task,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                CenterCrop(config.patch_size),
                ToTensor()
            ])
        )
        return DataLoader(dst, pin_memory=True)

def make_model_all():
    model_1 = VNet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=True
    ).cuda()

    model_2 = DiffVNet1(
        n_channels=config.num_channels,
        n_classes=config.num_cls+config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=True
    ).cuda()

    optimizer = optim.SGD([{'params': model_1.parameters()},
                           {'params': model_2.parameters()}],
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )
    return model_1, model_2, optimizer

def compute_proto_prob(proto_ck, feature, temp=1):
    proto_seg = torch.einsum('b c d h w, k c d h w -> b k d h w', feature, proto_ck) / temp
    return proto_seg

if __name__ == '__main__':
    import random
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # make logger file
    snapshot_path = f'./logs1/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))

    # make logger
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S', force=True
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    # make data loader
    if args.task == 'brats2019':
        transforms_train_labeled = transforms.Compose([RandomRotFlip(),RandomCrop(config.patch_size),ToTensor()])
        transforms_train_unlabeled = transforms.Compose([RandomRotFlip(),RandomCrop(config.patch_size),ToTensor()])
    else:
        transforms_train_labeled = get_StrongAug(config.patch_size, 3, 0.7)
        transforms_train_unlabeled = get_StrongAug(config.patch_size, 3, 0.7)

    unlabeled_loader = make_loader(args.split_unlabeled, transforms_tr=transforms_train_unlabeled, task=args.task, unlabeled=True)
    labeled_loader = make_loader(args.split_labeled, transforms_tr=transforms_train_labeled, task=args.task, repeat=len(unlabeled_loader.dataset))

    eval_loader = make_loader(args.split_eval, task=args.task, is_training=False)

    logging.info(f'{len(labeled_loader)} itertations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} itertations per epoch (unlabeled)')

    # make model, optimizer, and lr scheduler
    model_1, model_2, optimizer = make_model_all()

    # make loss function
    deno_loss = make_loss_function(args.sup_loss)

    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    mu = get_current_mu(0)

    best_eval1 = 0.0
    best_eval2 = 0.0
    best_epoch1 = 0
    best_epoch2 = 0
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []

        model_1.train()
        model_2.train()
        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader)):

            optimizer.zero_grad()

            image_l, label_l = fetch_data(batch_l)
            label_l = label_l.long()
            image_u = fetch_data(batch_u, labeled=False)

            shp = (config.batch_size, config.num_cls) + config.patch_size
            label_l_onehot = torch.zeros(shp).cuda()
            label_l_onehot.scatter_(1, label_l, 1)
            xl_start = label_l_onehot * 2 - 1

            if args.mixed_precision:
                with (autocast()):
                    if epoch_num < args.tl:
                        # model_1
                        f_l, o_l, proto_l = model_1(image_l, label=label_l)
                        proto_l_seg = compute_proto_prob(proto_l, f_l)

                        loss_1 = deno_loss(o_l, label_l)

                        y_pk_l = torch.cat([xl_start, proto_l_seg], dim=1)

                        # model_2
                        xl_t, t_l, noise_l = model_2(x=y_pk_l, pred_type="q_sample")
                        proto_l_xi = model_2(x=xl_t, step=t_l, image=image_l, pred_type="D_xi_l")
                        loss_2 = deno_loss(proto_l_xi[:, :config.num_cls, :, :, :], label_l)

                        loss = loss_1 + loss_2

                    else:
                        # model_1
                        f_l, o_l, proto_l = model_1(image_l, label=label_l)
                        f_u, o_u, proto_u = model_1(image_u)

                        proto_l_seg = compute_proto_prob(proto_l, f_l)
                        proto_u_seg = compute_proto_prob(proto_u, f_u)

                        y_pk_l = torch.cat([xl_start, proto_l_seg], dim=1)
                        y_pk_u = torch.cat([o_u, proto_u_seg], dim=1)

                        # model_2
                        xl_t, t_l, noise_l = model_2(x=y_pk_l, pred_type="q_sample")
                        proto_l_xi = model_2(x=xl_t, step=t_l, image=image_l, pred_type="D_xi_l")

                        xu_t, t_u, noise_u = model_2(x=y_pk_u, pred_type="q_sample")
                        proto_u_xi = model_2(x=xu_t, step=t_u, image=image_u, pred_type="D_xi_l")

                        with torch.no_grad():
                            smoothing = GaussianSmoothing(config.num_cls, 3, 1)
                            proto_u_xi_soft_y = smoothing(F.gumbel_softmax(proto_u_xi[:, :config.num_cls, :, :, :], dim=1))
                            proto_u_xi_soft_pk = smoothing(F.gumbel_softmax(proto_u_xi[:, config.num_cls:, :, :, :], dim=1))
                            pseudo_label_y = torch.argmax(proto_u_xi_soft_y, dim=1, keepdim=True)
                            pseudo_label_pk = torch.argmax(proto_u_xi_soft_pk, dim=1, keepdim=True)

                        # loss_1
                        L_sup_1 = deno_loss(o_l, label_l)
                        L_unsup_1_y = deno_loss(o_u, pseudo_label_y.detach())
                        L_unsup_1_pk = deno_loss(o_u, pseudo_label_pk.detach())
                        L_unsup_1 = L_unsup_1_y + L_unsup_1_pk
                        loss_1 = L_sup_1 + args.gamma1 * mu * L_unsup_1 #gama1 0.01

                        # loss_2
                        loss_2_y = deno_loss(proto_l_xi[:, :config.num_cls, :, :, :], label_l)
                        loss_2_pk = deno_loss(proto_l_xi[:, config.num_cls:, :, :, :], label_l)
                        loss_2 = loss_2_y + loss_2_pk

                        loss = loss_1 + args.gamma2 * loss_2 #gama2 0.7

                # backward passes should not be under autocast.
                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer)
                amp_grad_scaler.update()

            else:
                raise NotImplementedError

            loss_list.append(loss.item())

        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)} | lr : {get_lr(optimizer)} | mu : {mu}')
        optimizer.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)

        mu = get_current_mu(epoch_num)

        # =======================================================================================
        # Validation
        # =======================================================================================
        if epoch_num % 1 == 0:
            if epoch_num >= 25:
                save_path0 = os.path.join(snapshot_path, f'ckpts/model_{epoch_num}.pth')
                torch.save({'state_dict': model_1.state_dict(), }, save_path0)

            dice_list = [[] for _ in range(config.num_cls-1)]
            model_1.eval()
            model_2.eval()

            dice_func = SoftDiceLoss(smooth=1e-8, do_bg=False)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)
                    p_u_theta = model_1(image, use_prototype=False)[1]
                    del image

                    shp = (p_u_theta.shape[0], config.num_cls) + p_u_theta.shape[2:]
                    gt = gt.long()

                    y_onehot = torch.zeros(shp).cuda()
                    y_onehot.scatter_(1, gt, 1)

                    x_onehot = torch.zeros(shp).cuda()
                    p_u_theta = torch.argmax(p_u_theta, dim=1, keepdim=True).long()
                    x_onehot.scatter_(1, p_u_theta, 1)

                    dice = dice_func(x_onehot, y_onehot, is_training=False)
                    dice = dice.data.cpu().numpy()
                    for i, d in enumerate(dice):
                        dice_list[i].append(d)

            dice_mean = []
            for dice in dice_list:
                dice_mean.append(np.mean(dice))
            logging.info(f'evaluation epoch {epoch_num}, dice: {np.mean(dice_mean)}, {dice_mean}')
            if np.mean(dice_mean) > best_eval1:
                best_eval1 = np.mean(dice_mean)
                best_epoch1 = epoch_num
                save_path = os.path.join(snapshot_path, f'ckpts/best_model1.pth')
                torch.save({'state_dict': model_1.state_dict(),}, save_path)
                logging.info(f'saving best model to {save_path}')
                save_path1 = os.path.join(snapshot_path, f'ckpts/best_model_{epoch_num}.pth')
                torch.save({'state_dict': model_1.state_dict(), }, save_path1)
            logging.info(f'\t best eval dice is {best_eval1} in epoch {best_epoch1}')
            if epoch_num - best_epoch1 == config.early_stop_patience:
                logging.info(f'Early stop.')
                break