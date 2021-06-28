# how to use
# [Top] CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4
# python train_pixelsnail.py --hierarchy top
# [Bottom] XXX python train_pixelsnail.py --hierarchy bottom

import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import LMDBDataset
from models.pixelsnail import PixelSNAIL
import logging
#from utils.comm import synchronize, get_rank
#import socket
'''
def _find_free_port():

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port
'''

def train(args, epoch, loader, model, optimizer, device):
    loader.sampler.set_epoch(epoch)
    # define loss function
    criterion = nn.CrossEntropyLoss()

    for i, (top, bottom, label) in enumerate(loader):
        model.zero_grad()
        top = top.to(device)
        if args.hierarchy == 'top':
            target = top
            out, _ = model(top)
        else:
            bottom = bottom.to(device)
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
    logging.info('[Epoch]: {}, [Lr]: {:.4f}, [Loss]: {:.4f}, [Acc]: {:.4f}'
                .format(epoch,lr,loss.item(),accuracy))

def main():

    # input parameters
    parser = argparse.ArgumentParser(description='PyTorch VQ-VAE-2')
    parser.add_argument('--batch', type=int, default=48, help='train batch size')
    parser.add_argument('--epoch', type=int, default=200, help='total train epochs')
    parser.add_argument('--hierarchy', type=str, default='top', help='top or bottom')
    parser.add_argument('--lr', type=float, default=3e-4, help='train learning rate')
    parser.add_argument('--lmdb_path', type=str, default='/data3/liumingzhou/lmdb', help='path to extracted codes')
    parser.add_argument('--local_rank', type=int, default=-1, help='multiple GPUs training paras')
    args = parser.parse_args()

    # set the GPU number we want use
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'

    # set parameters for distributed training (DDP)
    # - only one machine, so set_device(0)
    torch.cuda.set_device(args.local_rank)
    #port = _find_free_port()
    #dist_url = f"tcp://127.0.0.1:{port}"

    torch.distributed.init_process_group(
        backend="nccl")#, init_method=dist_url, rank=args.local_rank)
    #synchronize()

    device = torch.device("cuda", args.local_rank)

    # create dataloader for DDP
    dataset = LMDBDataset(args.lmdb_path)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #num_gpus = 4
    #images_per_batch = dataset.__len__() // args.batch
    #images_per_gpu = images_per_batch // num_gpus
    #batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, images_per_gpu, drop_last=True)

    loader = DataLoader(dataset, sampler=sampler,batch_size=args.batch//4, num_workers=16, drop_last=True)

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
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #model = nn.DataParallel(model)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank)

    # create logs
    rank = torch.distributed.get_rank()
    logging.basicConfig(level=logging.INFO if rank in [-1, 0] else logging.WARN)
    logging.info("Start training")

    for i in range(args.epoch):
        train(args, i, loader, model, optimizer, device)
        if i % 25 == 0 or i == (args.epoch-1):
            if torch.distributed.get_rank() == 0:
                torch.save(
                    {'model': model.module.state_dict(), 'args': args},
                    f'/data3/liumingzhou/checkpoint/pixelsnail_{args.hierarchy}_{str(i).zfill(3)}.ckpt')

if __name__ == '__main__':
    main()
