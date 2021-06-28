# how to use
# [Top] python train_pixelsnail.py --hierarchy top
# [Bottom] python train_pixelsnail.py --hierarchy bottom

import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.dataset import LMDBDataset
from models.pixelsnail import PixelSNAIL


def train(args, epoch, loader, model, optimizer):
    # define loss function
    loader = tqdm(loader)
    criterion = nn.CrossEntropyLoss()

    for i, (top, bottom, label) in enumerate(loader):
        model.zero_grad()
        top = top.cuda()
        if args.hierarchy == 'top':
            target = top
            out, _ = model(top)
        else:
            bottom = bottom.cuda()
            target = bottom
            out, _ = model(bottom, condition=top)

        loss = criterion(out, target)
        loss.backward()

        optimizer.step()

        # record
        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'[Epoch]: {epoch}; [Loss]: {loss.item():.5f}; '
                f'[Acc]: {accuracy:.5f}; [Lr]: {lr:.5f}'
            )
        )

def main():

    # input parameters
    parser = argparse.ArgumentParser(description='PyTorch VQ-VAE-2')
    parser.add_argument('--batch', type=int, default=48, help='train batch size')
    parser.add_argument('--epoch', type=int, default=420, help='total train epochs')
    parser.add_argument('--hierarchy', type=str, default='top', help='top or bottom')
    parser.add_argument('--lr', type=float, default=3e-4, help='train learning rate')
    parser.add_argument('--lmdb_path', type=str, default='/data3/liumingzhou/mylmdb', help='path to extracted codes')
    args = parser.parse_args()

    # set the GPU number we want use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    # create dataloader
    dataset = LMDBDataset(args.lmdb_path)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=16, drop_last=True)

    # build model
    # [Note] for Top and Bottom, the model setting parameters are slightly different
    if args.hierarchy == 'top':
        model = PixelSNAIL(
            shape =[32, 32], n_class=512, channel=256, kernel_size=5,
            n_block=4, n_res_block=4, res_channel=256, attention=True, dropout=0.1, n_out_res_block=0)
    elif args.hierarchy == 'bottom':
        model = PixelSNAIL(
            shape =[64, 64], n_class=512, channel=256, kernel_size=5,
            n_block =4, n_res_block=4, res_channel=256, attention=False, dropout=0.1, n_cond_res_block=3,
            cond_res_channel=256)
    else:
        raise ValueError('hierarchy must be either top or bottom')

    # define optimizer and start training
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model = nn.DataParallel(model)

    for i in range(args.epoch):
        train(args, i, loader, model, optimizer)
        if i % 25 == 0:
            torch.save(
                {'model': model.module.state_dict(), 'args': args},
                f'/data3/liumingzhou/checkpoint/pixelsnail_{args.hierarchy}_{str(i).zfill(3)}.ckpt')

if __name__ == '__main__':
    main()
