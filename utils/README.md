# VQ VAE-2 training and inference in PyTorch

This implements VA VAE-2 on face reconstruction and generation tasks.

## Requirments

- Dependencies: see details in requirements.txt
- For training: you may need to download a rather large face dataset, In my case, 7500 images from http://www.seeprettyface.com/mydataset.html are used 

## Training

To train a model, there are three stages to finish

- Firstly, train a VQ-VAE (cost 1 TITAN-X GPU about one hour)

```
python train_vqvae.py --save_dir [Where You Want To Save Checkpoints] --dataset [Where Is Your Dataset]
```

- Secondly, you need to extract the latent codes of the entire dataset by

```
python extract_codes.py --name [Where You Want To Save Results] --path [Where Is Your Dataset]
```

- Finally, you need to train a hirachical Pixel CNN model called Pixel Snail on two level of latents (cost 5 hours on 4 TITAN-X GPU)

```
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 python train_pixelsnail.py --hierarchy top
```

```
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 python train_pixelsnail.py --hierarchy bottom
```

## Generation

To sample from random distribution and generate a face image we never see in training dataset, use:

```
python generate.py --vqvae_ckpt --pixelsnail_top_ckpt --pixelsnail_bottom_ckpt --out [Where You Want Save Results]
```

## Copyrights

Some codes are referred on https://github.com/deepmind/sonnet (TensorFlow version by Original Authors of VQ VAE-2)

Problems contact: XXX@XXX.com
