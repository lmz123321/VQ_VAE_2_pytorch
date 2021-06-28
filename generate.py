# how to use
# # python generate.py --vqvae /data3/liumingzhou/checkpoint/vqvae_551.pt --top /data3/liumingzhou/checkpoint/pixelsnail_top_401.pt
# --bottom /data3/liumingzhou/checkpoint/pixelsnail_bottom_401.pt sample.png

import argparse
import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from models.vq_vae_2 import VQVAE
from models.pixelsnail import PixelSNAIL

def sample_model(model, batch, size, temperature, condition=None):
    r"""
    This function finish per pixels dependency sampling
    :param model:
    :param batch:
    :param size:
    :param temperature:
    :param condition:
    :return:
    """
    with torch.no_grad():
        row = torch.zeros(batch, *size, dtype=torch.int64).cuda()
        cache = {}

        for i in tqdm(range(size[0])):
            for j in range(size[1]):
                out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
                prob = torch.softmax(out[:, :, i, j] / temperature, 1)
                sample = torch.multinomial(prob, 1).squeeze(-1)
                row[:, i, j] = sample
    return row


def load_model(mod, checkpoint):
    r"""
    This fucntion load model weights
    :param model:
    :param checkpoint:
    :return:
    """
    if mod == 'vqvae':
        model = VQVAE()
    elif mod == 'pixelsnailTop':
        model = PixelSNAIL(
            shape=[32, 32], n_class=512, channel=256, kernel_size=5,
            n_block=4, n_res_block=4, res_channel=256, attention=True, dropout=0.1, n_out_res_block=0)

    elif mod == 'pixelsnailBottom':
        model = PixelSNAIL(
            shape=[64, 64], n_class=512, channel=256, kernel_size=5,
            n_block=4, n_res_block=4, res_channel=256, attention=False, dropout=0.1, n_cond_res_block=3,
            cond_res_channel=256)
    else:
        raise ValueError

    ckpt = torch.load(checkpoint)

    if mod == 'vqvae':
        model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt['model'])

    model = model.cuda()

    return model

def main():
    # input parameters
    parser = argparse.ArgumentParser(description='PyTorch VQ-VAE-2')
    parser.add_argument('--batch', type=int, default=8, help='inference batch size')
    parser.add_argument('--vqvae_ckpt', type=str, default='/data3/liumingzhou/checkpoint/vq_vae_2_99.ckpt',
                        help='path to VQ-VAE 2 ckpt')
    parser.add_argument('--pixelsnail_top_ckpt', type=str, default='/data3/liumingzhou/checkpoint/pixelsnail_top_199.ckpt',
                        help='path to pixel snail top ckpt')
    parser.add_argument('--pixelsnail_bottom_ckpt', type=str, default='/data3/liumingzhou/checkpoint/pixelsnail_bottom_199.ckpt',
                        help='path to pixel snail bottom ckpt')
    parser.add_argument('--out', type=str, default='/data3/liumingzhou/checkpoint/sample.png', help='output image filename')
    args = parser.parse_args()

    # set the GPU number we want use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # load model weights
    vqvae = load_model('vqvae', args.vqvae_ckpt)
    pixelsnailTop = load_model('pixelsnailTop', args.pixelsnail_top_ckpt)
    pixelsnailBottom = load_model('pixelsnailBottom', args.pixelsnail_bottom_ckpt)

    # sample from rand noise
    topSample = sample_model(pixelsnailTop, args.batch, [32, 32], 1.0)
    bottomSample = sample_model(pixelsnailBottom, args.batch, [64, 64], 1.0, condition=topSample)

    # reconstruction from noise
    decoded_sample = vqvae.decode_code(topSample, bottomSample)
    decoded_sample = decoded_sample.clamp(-1, 1)

    # save results
    save_image(decoded_sample, args.out, normalize=True, range=(-1, 1))

if __name__ == '__main__':
    main()