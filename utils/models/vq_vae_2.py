import torch
from torch import nn
from torch.nn import functional as F

# Reference https://github.com/deepmind/sonnet (TensorFlow version by Original Paper Authors)

class Quantizer(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super(Quantizer, self).__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.beta = 0.25

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def embed_code(self, embed_inds):

        return F.embedding(embed_inds, self.embed.transpose(0, 1))

    def forward(self, input):
        r"""
        Vector Quantizer in VQ-VAE-2
        :param input: [B,H,W,C]
        """
        flatten = input.reshape(-1, self.dim) # [BHW,C]
        # Compute L2 distance between latents and embedding weights
        dist = (flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True))

        # Get the encoding that has the min distance
        embed_inds = torch.argmin(dist, dim=1) # [BHW,]

        # Convert to one-hot encodings
        embed_one_hot = F.one_hot(embed_inds, self.n_embed).type(flatten.dtype)

        # Quantize the latents
        embed_inds = embed_inds.view(*input.shape[:-1]) # [B,H,W]
        quantizedLatent = self.embed_code(embed_inds) # [B,H,W,D]

        # Do some normalization
        if self.training:
            embed_onehot_sum = embed_one_hot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_one_hot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantizedLatent.detach(), input)
        embedding_loss = F.mse_loss(quantizedLatent, input.detach())
        vqLoss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantizedLatent = input + (quantizedLatent - input).detach()

        return quantizedLatent, vqLoss, embed_inds

class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size = 3, padding = 1, bias = False),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(out_channels, in_channels,
                                                kernel_size = 1, bias = False))
    def forward(self, input):

        return input + self.resblock(input)

class Encoder(nn.Module):
    r"""
    Here, we build the encoder for VQ-VAE-2
    it can be used either for topFeat or bottomFeat
    """
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super(Encoder, self).__init__()
        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel//2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel//2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1)]
        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel//2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel//2, channel, 3, padding=1)]
        else:
            raise ValueError

        for i in range(n_res_block):
            blocks.append(ResidualLayer(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):

        return self.blocks(input)


class Decoder(nn.Module):
    r"""
    Here, we build the decoder for VQ-VAE-2
    it can be used either for topFeat or bottomFeat
    """
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super(Decoder, self).__init__()
        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]
        for i in range(n_res_block):
            blocks.append(ResidualLayer(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))

        # stride is 4 means top feat, while stride is 2 means bottom feat
        # for these 2 case, the final layers of decoder is slightly different
        if stride == 4:
            blocks.extend(
                [nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                 nn.ReLU(inplace=True),
                 nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1)])
        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1))
        else:
            raise ValueError

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):

        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        channels=128,
        n_res_blocks=2,
        n_res_channels=32,
        embed_dim=64,
        n_embed=512
    ):
        super(VQVAE, self).__init__()

        self.encoderBottom = Encoder(in_channels, channels, n_res_blocks, n_res_channels, stride=4)
        self.encoderTop = Encoder(channels, channels, n_res_blocks, n_res_channels, stride=2)

        # decrease channels dimension of top and bottom Feat before Quantizer
        self.pre_quantize_layerTop = nn.Conv2d(channels, embed_dim, 1)
        self.pre_quantize_layerBottom = nn.Conv2d(embed_dim + channels, embed_dim, 1)

        # define Quantizer for top and bottom Feat
        self.quantizerTop = Quantizer(embed_dim, n_embed)
        self.quantizerBottom = Quantizer(embed_dim, n_embed)

        # decode Top latent
        self.decode_top = Decoder(
            embed_dim, embed_dim, channels, n_res_blocks, n_res_channels, stride=2)

        # upsample the top latent 4 times to match the size of bottom latent
        self.unsample_layer = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1)
        # given fused latent, reconstruct observation images
        self.decode_layer = Decoder(
            embed_dim + embed_dim,
            in_channels,
            channels,
            n_res_blocks,
            n_res_channels,
            stride=4,
        )

    def encode(self, input):
        r"""
        input -> feat -> quantized latent
        Note: VQ-VAE-2 has a hierarchical structure, which means: input -> bottomFeat -> topFeat
        :param input: [B,3,H,W]
        """
        bottomFeat = self.encoderBottom(input)
        topFeat = self.encoderTop(bottomFeat) # [B,C,H,W]

        # 1) quantization of Top feature
        topFeat = self.pre_quantize_layerTop(topFeat).permute(0, 2, 3, 1) # [B,H,W,C]
        quantTop, vqLossTop, id_t = self.quantizerTop(topFeat)
        quantTop = quantTop.permute(0, 3, 1, 2) # [B,C,H,W]
        vqLossTop = vqLossTop.unsqueeze(0)

        # 2) fusion of bottomFeat and decoded topFeat
        decodeTop = self.decode_top(quantTop)
        bottomFeat = torch.cat([decodeTop, bottomFeat], 1)

        # 3) quantization of Bottom feature
        quantBottom = self.pre_quantize_layerBottom(bottomFeat).permute(0, 2, 3, 1)
        quantBottom, vqLossBottom, id_b = self.quantizerBottom(quantBottom)
        quantBottom = quantBottom.permute(0, 3, 1, 2) # [B,C,H,W]
        vqLossBottom = vqLossBottom.unsqueeze(0)

        return quantTop, quantBottom, vqLossTop + vqLossBottom, id_t, id_b

    def decode(self, quantTop, quantBottom):
        r"""
        fuse unsampled quantTop and quantBottom, then use a decoder to reconstruct image
        :param quantTop:
        :param quantBottom:
        """
        quantTop = self.unsample_layer(quantTop)
        quant = torch.cat([quantTop, quantBottom], 1)
        out = self.decode_layer(quant)

        return out

    def decode_code(self, codeTop, codeBottom):

        quantTop = self.quantizerTop.embed_code(codeTop).permute(0, 3, 1, 2) # [B,C,H,W]
        quantBottom = self.quantizerBottom.embed_code(codeBottom).permute(0, 3, 1, 2) # [B,C,H,W]

        out = self.decode(quantTop, quantBottom)

        return out

    def forward(self, input):

        quantTop, quantBottom, vqLoss, _, _ = self.encode(input)

        reconstruction = self.decode(quantTop, quantBottom)

        return reconstruction, vqLoss