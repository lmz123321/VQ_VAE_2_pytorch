import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import lmdb

import os
from utils.dataset import ImageFileDataset, CodeRow, extract
from models.vq_vae_2 import VQVAE


def main():
    # input parameters
    parser = argparse.ArgumentParser(description='PyTorch VQ-VAE-2')
    parser.add_argument('--size', type=int, default=256,
                        help='input image size')
    parser.add_argument('--ckpt', type=str, default='/data3/liumingzhou/checkpoint/vq_vae_2_99.ckpt',
                        help='VQ VAE training checkpoint')
    parser.add_argument('--name', type=str, default='/data3/liumingzhou/lmdb/',
                        help='output path')
    parser.add_argument('--path', type=str, default='/data3/liumingzhou/homework/',
                        help='face dataset path')
    args = parser.parse_args()

    # set the GPU number we want use
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # define a dataloader for inference
    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # build our custom dataset and dataloader
    dataset = ImageFileDataset(args.path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # prepare models
    model = VQVAE()
    model.load_state_dict(torch.load(args.ckpt))
    model = model.cuda()
    model.eval()

    # open extracted codes
    env = lmdb.open(args.name, map_size=100 * 1024 * 1024 * 1024)
    extract(env, loader, model)

if __name__ == '__main__':
    main()