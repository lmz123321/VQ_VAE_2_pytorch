# this script define the process of training the first stage of VA-VAE-2
# usage: python train_vqvae.py --dataset [Your Dataset Root Path]

import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from models.vq_vae_2 import VQVAE

def train(epoch,model,train_loader,optimizer,args):
    criterion = nn.MSELoss()
    lenHist = list()
    lossHist = list()
    reconstcHist = list()
    vqHist = list()
    for ind, (image,label) in enumerate(train_loader):
        image = image.cuda()
        model.zero_grad()
        optimizer.zero_grad()

        out, vq_loss = model(image)
        vq_loss = vq_loss.mean()
        reconstru_loss = criterion(out, image)

        loss = reconstru_loss + vq_loss
        loss.backward()

        optimizer.step()
        #record
        lenHist.append(image.shape[0])
        lossHist.append(loss.item())
        reconstcHist.append(reconstru_loss.item())
        vqHist.append(vq_loss.item())

    # logs and sample reconstruction
    loss = np.sum(np.array(lossHist)*np.array(lenHist)/np.sum(np.array(lenHist)))
    mse = np.sum(np.array(reconstcHist)*np.array(lenHist)/np.sum(np.array(lenHist)))
    vq = np.sum(np.array(vqHist)*np.array(lenHist)/np.sum(np.array(lenHist)))

    print('[EPOCH]: {}, [LR]: {}, [LOSS]: {:.4f},[MSE Loss]: {:.4f}, [VQ Loss]: {:.4f}'.format(
        epoch, optimizer.param_groups[0]["lr"], loss, mse, vq,
    ))
    if epoch%25 ==0 or epoch==(args.epochs-1):
        model.eval()
        samples = image[:32]
        with torch.no_grad():
            out, _ = model(samples)
            utils.save_image(torch.cat([samples,out], dim=0),
                              os.path.join(args.save_dir,'reconstru_{}.png'.format(epoch)),
                              nrow = 16,
                              normalize = True,
                              range = (-1,1))
        model.train()


def main():
    # input parameters
    parser = argparse.ArgumentParser(description='PyTorch VQ-VAE-2')
    parser.add_argument("--size", type=int, default=256, help='input image size')
    parser.add_argument("--epochs", type=int, default=100, help='total training epochs')
    parser.add_argument("--lr", type=float, default=3e-4, help='init learning rate')
    parser.add_argument("--batch_size", type=float, default=128, help='batch size')
    parser.add_argument("--workers", type=float, default=16, help='number workers for data loader')
    parser.add_argument("--save_dir", type=str, default='/data3/liumingzhou/checkpoint/', help='where to save checkpoints')
    parser.add_argument("--dataset", type=str, default='/data3/liumingzhou/homework/', help='root of training dataset')
    args = parser.parse_args()

    # set the GPU number we want use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # define training hyper parameters
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    custom_transforms = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            SetRange,
        ]
    )
    dataset = datasets.ImageFolder(args.dataset, transform=custom_transforms)
    train_loader = DataLoader(dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.workers, pin_memory=True)

    # define model
    cudnn.benchmark = True
    model = VQVAE()
    model = DataParallel(model)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # start training
    for epoch in tqdm(range(args.epochs)):

        train(epoch,model,train_loader,optimizer,args)
        # save checkpoint
        if epoch%25 ==0 or epoch==(args.epochs-1):
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_dir,'vq_vae_2_{}.ckpt'.format(epoch)))

if __name__ == '__main__':
    main()