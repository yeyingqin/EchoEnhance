import copy
import glob
import math
import os
import random
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import Augmentor
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from PIL import Image
import time

from tensorboardX.utils import make_grid
from torch import einsum, nn
from torch.optim import Adam, RAdam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import utils
from tqdm.auto import tqdm
from thop import profile
import copy
import importlib
import torchvision.transforms as transforms
import torchvision.utils as utils
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.fft as fft
import csv
import time


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])
# helpers functions
metric_module = importlib.import_module('metrics')

import torch.fft


class FrequencyAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, low_freq_cutoff=0.1, high_freq_factor=2.0):
        super(FrequencyAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.low_freq_cutoff = low_freq_cutoff
        self.high_freq_factor = high_freq_factor

        # 定义低频注意力机制
        self.low_freq_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )

        # 定义高频注意力机制
        self.high_freq_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获取频率分量
        low_freq, high_freq = self._split_frequency(x)

        # 计算低频和高频的注意力权重
        low_freq_attention = self.low_freq_attention(low_freq)
        high_freq_attention = self.high_freq_attention(high_freq)

        # 强制关注高频
        high_freq_attention *= self.high_freq_factor
        low_freq_attention *= self.low_freq_cutoff

        self.visualize_attention(low_freq_attention, 'Low Frequency Attention')
        # self.visualize_attention(high_freq_attention, 'High Frequency Attention')


        # 应用注意力机制
        attention_weights = self.sigmoid(low_freq_attention) * self.sigmoid(high_freq_attention)
        out = x * attention_weights

        return out

    def visualize_attention(self, attention_matrix, title):
        # 转换为CPU并转换为numpy数组
        attention_matrix_np = attention_matrix.detach().cpu().numpy()

        # 可视化注意力矩阵（这里选择第一个通道作为示例）
        # print("kaishishuchula")
        # plt.figure(figsize=(8, 8))
        # plt.imshow(attention_matrix_np[0, 0], cmap='viridis')
        # plt.colorbar()
        # plt.title(title)
        # plt.xlabel("Key Positions")
        # plt.ylabel("Query Positions")
        # plt.show()
        # plt.savefig('output_image.png')


    def _split_frequency(self, x):
        # 对输入进行傅里叶变换
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))

        # 计算频率分量的幅值
        magnitude = torch.abs(x_fft_shifted)

        # 获取低频和高频分量
        low_freq_mask = magnitude < self.low_freq_cutoff
        high_freq_mask = magnitude >= self.low_freq_cutoff

        low_freq = torch.fft.ifftshift(x_fft_shifted * low_freq_mask, dim=(-2, -1))
        high_freq = torch.fft.ifftshift(x_fft_shifted * high_freq_mask, dim=(-2, -1))

        low_freq = torch.fft.ifft2(low_freq, dim=(-2, -1)).real
        high_freq = torch.fft.ifft2(high_freq, dim=(-2, -1)).real

        return low_freq, high_freq




def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# normalization functions


def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5

# small helper modules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        condition=False,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.depth = len(dim_mults)
        input_channels = channels + channels * (1 if condition else 0)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x, time):
        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = x[..., :H, :W].contiguous()
        return x


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()

    def forward(self, x):
        B, C, H, W = x.size()

        # 行处理
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2

        # 列处理
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        # 计算子带
        x_LL = x1 + x2 + x3 + x4  # 低频
        x_HL = -x1 - x2 + x3 + x4  # 水平高频
        x_LH = -x1 + x2 - x3 + x4  # 垂直高频
        x_HH = x1 - x2 - x3 + x4  # 对角高频

        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()

    def forward(self, x):
        B, C, H, W = x.size()
        C = C // 4

        # 分离子带
        x_LL = x[:, 0:C, :, :]
        x_HL = x[:, C:C * 2, :, :]
        x_LH = x[:, C * 2:C * 3, :, :]
        x_HH = x[:, C * 3:C * 4, :, :]

        # 创建输出
        x = torch.zeros([B, C, H * 2, W * 2], device=x.device)

        # 重构
        x[:, :, 0::2, 0::2] = (x_LL - x_HL - x_LH + x_HH) / 2
        x[:, :, 1::2, 0::2] = (x_LL - x_HL + x_LH - x_HH) / 2
        x[:, :, 0::2, 1::2] = (x_LL + x_HL - x_LH - x_HH) / 2
        x[:, :, 1::2, 1::2] = (x_LL + x_HL + x_LH + x_HH) / 2

        return x


class WaveletFrequencyAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, mode="high_pass"):
        super(WaveletFrequencyAttention, self).__init__()
        self.in_channels = in_channels

        # 小波变换模块
        self.dwt = DWT()
        self.iwt = IWT()

        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

        # 频率模式
        self.mode = mode

    def forward(self, x):
        # 保存原始输入用于残差连接
        residual = x

        # 应用小波变换
        wavelet_coeffs = self.dwt(x)

        # 分离子带
        c = x.size(1)
        ll = wavelet_coeffs[:, 0:c, :, :]
        hl = wavelet_coeffs[:, c:c * 2, :, :]
        lh = wavelet_coeffs[:, c * 2:c * 3, :, :]
        hh = wavelet_coeffs[:, c * 3:c * 4, :, :]

        # 根据模式应用不同的频率增强
        if self.mode == "high_pass":
            # 增强高频信息 (HL, LH, HH)
            b, c, h, w = ll.shape
            y = self.avg_pool(ll).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)

            # 对低频应用较小权重
            ll = ll * y * 0.5

            # 对高频应用较大权重
            hl = hl * 2.0
            lh = lh * 2.0
            hh = hh * 2.0

        elif self.mode == "low_pass":
            # 增强低频信息 (LL)
            b, c, h, w = ll.shape
            y = self.avg_pool(ll).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)

            # 对低频应用较大权重
            ll = ll * y * 2.0

            # 对高频应用较小权重
            hl = hl * 0.5
            lh = lh * 0.5
            hh = hh * 0.5

        # 重新组合小波系数
        enhanced_coeffs = torch.cat([ll, hl, lh, hh], dim=1)

        # 逆小波变换
        output = self.iwt(enhanced_coeffs)

        # 残差连接
        return output + residual


class UNet1(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        condition=False,
        use_wavelet=True,  # 添加使用小波变换的开关
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.depth = len(dim_mults)
        input_channels = channels + channels * (1 if condition else 0)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # 替换原有的频域注意力为小波频域注意力
        if use_wavelet:
            self.freq_attention_b = WaveletFrequencyAttention(
                mid_dim, reduction=16, mode="low_pass")
            self.freq_attention_s1 = WaveletFrequencyAttention(
                256, reduction=16, mode="high_pass")
            self.freq_attention_s2 = WaveletFrequencyAttention(
                128, reduction=16, mode="high_pass")
        # else:
        #     # 保持原有的频域注意力
        #     self.freq_attention_b= FrequencyAttention(
        #         mid_dim, reduction=16, mode="low_pass")
        #     self.freq_attention_s1 = FrequencyAttention(
        #         256, reduction=16, mode="high_pass")
        #     self.freq_attention_s2 = FrequencyAttention(
        #         128, reduction=16, mode="high_pass")
        # Add FrequencyAttention after mid_block1
        # self.freq_attention_b = FrequencyAttention(mid_dim, reduction=16, low_freq_cutoff=2.0, high_freq_factor=0.5)
        # self.freq_attention_s1 = FrequencyAttention(256, reduction=16, low_freq_cutoff=0.5, high_freq_factor=2.0)
        # self.freq_attention_s2 = FrequencyAttention(128, reduction=16, low_freq_cutoff=0.5, high_freq_factor=2.0)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x, time):
        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.freq_attention_b(x)  # 应用 FrequencyAttention
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            skip_connection = h.pop()
            if skip_connection.size(1) == 256:
                skip_connection = self.freq_attention_s1(skip_connection)
            if skip_connection.size(1) == 128:
                skip_connection = self.freq_attention_s2(skip_connection)

            x = torch.cat((x, skip_connection), dim=1)
            x = block1(x, t)

            # Process the second skip connection
            skip_connection = h.pop()
            if skip_connection.size(1) == 256:
                skip_connection = self.freq_attention_s1(skip_connection)
            if skip_connection.size(1) == 128:
                skip_connection = self.freq_attention_s2(skip_connection)

            x = torch.cat((x, skip_connection), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = x[..., :H, :W].contiguous()
        return x







class UnetRes(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        num_unet=1,
        condition=False,
        objective='pred_res_noise',
        test_res_or_noise="res_noise"
    ):
        super().__init__()
        self.condition = condition
        self.channels = channels
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        self.num_unet = num_unet
        self.objective = objective
        self.test_res_or_noise = test_res_or_noise
        # determine dimensions

        self.unet0 = UNet1(dim,
                        init_dim=init_dim,
                        out_dim=out_dim,
                        dim_mults=dim_mults,
                        channels=channels,
                        resnet_block_groups=resnet_block_groups,
                        learned_variance=learned_variance,
                        learned_sinusoidal_cond=learned_sinusoidal_cond,
                        random_fourier_features=random_fourier_features,
                        learned_sinusoidal_dim=learned_sinusoidal_dim,
                        condition=condition)

    def forward(self, x, time):
        if self.objective == "pred_noise":
            time = time[1]
        elif self.objective == "pred_res":
            time = time[0]
        return [self.unet0(x, time)]

# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def gen_coefficients(timesteps, schedule="increased", sum_scale=1, ratio=1):
    if schedule == "increased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        alphas = y/y_sum
    elif schedule == "decreased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        y = torch.flip(y, dims=[0])
        alphas = y/y_sum
    elif schedule == "lamda":
        x = np.linspace(0.0001, 0.02, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        alphas = 1 - y
    elif schedule == "average":
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    elif schedule == "normal":
        sigma = 1.0
        mu = 0.0
        x = np.linspace(-3+mu, 3+mu, timesteps, dtype=np.float32)
        y = np.e**(-((x-mu)**2)/(2*(sigma**2)))/(np.sqrt(2*np.pi)*(sigma**2))
        y = torch.from_numpy(y)
        alphas = y/y.sum()
    else:
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)

    return alphas*sum_scale

# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)



def high_pass_filter(x, threshold):
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = 0
    x_freq = x_freq * mask

    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    return x_filtered


def low_pass_filter(x, threshold):
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.zeros((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = 1
    x_freq = x_freq * mask

    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    return x_filtered

def mixed_frequency_loss(input, target, threshold=1, high_weight=10, low_weight=5):
    input_high = high_pass_filter(input, threshold)
    target_high = high_pass_filter(target, threshold)
    input_low = low_pass_filter(input, threshold)
    target_low = low_pass_filter(target, threshold)

    high_freq_loss = torch.nn.functional.mse_loss(input_high, target_high)
    mse_loss = torch.nn.functional.mse_loss(input, target)
    low_freq_loss = torch.nn.functional.mse_loss(input_low, target_low)


    loss = high_weight * high_freq_loss + 5* mse_loss + low_freq_loss*low_weight
    return loss

class ResidualDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        delta_end = 1.5e-3,
        sampling_timesteps=None,
        loss_type='l1',
        objective='pred_res_noise',
        ddim_sampling_eta=0.,
        condition=False,
        sum_scale=None,
        test_res_or_noise="None",
    ):
        super().__init__()
        assert not (
            type(self) == ResidualDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.image_size = image_size
        self.objective = objective
        self.condition = condition
        self.test_res_or_noise = test_res_or_noise
        self.delta_end = delta_end

        if self.condition:
            self.sum_scale = sum_scale if sum_scale else 0.01
            # ddim_sampling_eta = 0.25
        else:
            self.sum_scale = sum_scale if sum_scale else 1.

        beta_schedule = "linear"
        beta_start = 0.0001
        beta_end = 0.02
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            betas = (torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float32) ** 2)
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            betas = betas_for_alpha_bar(timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        delta_start = 1e-6
        delta = torch.linspace(delta_start, self.delta_end, timesteps, dtype=torch.float32)
        delta_cumsum = delta.cumsum(dim=0).clip(0, 1)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumsum = 1-alphas_cumprod ** 0.5
        betas2_cumsum = 1-alphas_cumprod

        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        alphas = alphas_cumsum-alphas_cumsum_prev
        alphas[0] = 0
        betas2 = betas2_cumsum-betas2_cumsum_prev
        betas2[0] = 0
        # print(alphas_cumsum[-1])
        # print(delta_cumsum[-1])
        # raise
        betas_cumsum = torch.sqrt(betas2_cumsum)

        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters
        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('delta', delta)
        register_buffer('delta_cumsum', delta_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1-alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1',
                        betas2_cumsum_prev/betas2_cumsum)
        register_buffer('posterior_mean_coef2', (betas2 *
                        alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2/betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def init(self):
        timesteps = 1000

        beta_schedule = "linear"
        beta_start = 0.0001
        beta_end = 0.02
        if beta_schedule == "linear":
            betas = torch.linspace(
                beta_start, beta_end, timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            betas = betas_for_alpha_bar(timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        delta_start = 1e-6
        delta = torch.linspace(delta_start, self.delta_end, timesteps, dtype=torch.float32)
        delta_cumsum = delta.cumsum(dim=0).clip(0, 1)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumsum = 1-alphas_cumprod ** 0.5
        betas2_cumsum = 1-alphas_cumprod

        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        alphas = alphas_cumsum-alphas_cumsum_prev
        alphas[0] = alphas[1]
        betas2 = betas2_cumsum-betas2_cumsum_prev
        betas2[0] = betas2[1]

        betas_cumsum = torch.sqrt(betas2_cumsum)

        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)

        self.alphas = alphas
        self.alphas_cumsum = alphas_cumsum
        self.delta = delta
        self.delta_cumsum = delta_cumsum
        self.one_minus_alphas_cumsum = 1-alphas_cumsum
        self.betas2 = betas2
        self.betas = torch.sqrt(betas2)
        self.betas2_cumsum = betas2_cumsum
        self.betas_cumsum = betas_cumsum
        self.posterior_mean_coef1 = betas2_cumsum_prev/betas2_cumsum
        self.posterior_mean_coef2 = (
            betas2 * alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum
        self.posterior_mean_coef3 = betas2/betas2_cumsum
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def predict_noise_from_res(self, x_t, t, x_input, pred_res):
        return (
            (x_t - (1-extract(self.delta_cumsum,t,x_t.shape)) * x_input - (extract(self.alphas_cumsum, t, x_t.shape)-1)
             * pred_res)/extract(self.betas_cumsum, t, x_t.shape)
        )

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise):
        return (
            (x_t-extract(self.alphas_cumsum, t, x_t.shape)*x_input -
             extract(self.betas_cumsum, t, x_t.shape) * noise)/extract(self.one_minus_alphas_cumsum, t, x_t.shape)
        )

    def predict_start_from_res_noise(self, x_t, t, x_res, noise):
        return (
            x_t-extract(self.alphas_cumsum, t, x_t.shape) * x_res -
            extract(self.betas_cumsum, t, x_t.shape) * noise
        )

    def q_posterior_from_res_noise(self, x_res, noise, x_t, t):
        return (x_t-extract(self.alphas, t, x_t.shape) * x_res -
                (extract(self.betas2, t, x_t.shape)/extract(self.betas_cumsum, t, x_t.shape)) * noise)

    def q_posterior(self, pred_res, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_input, x, t, task=None, clip_denoised=True):
        if not self.condition:
            x_in = x
        else:
            x_in = torch.cat((x, x_input), dim=1)
        model_output = self.model(x_in,
                                  [self.alphas_cumsum[t]*self.num_timesteps,
                                      self.betas_cumsum[t]*self.num_timesteps])
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_denoised else identity

        if self.objective == 'pred_res_noise':
            if self.test_res_or_noise == "res_noise":
                pred_res = model_output[0]
                pred_noise = model_output[1]
                pred_res = maybe_clip(pred_res)
                x_start = self.predict_start_from_res_noise(
                    x, t, pred_res, pred_noise)
                x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "res":
                pred_res = model_output[0]
                pred_res = maybe_clip(pred_res)
                pred_noise = self.predict_noise_from_res(
                    x, t, x_input, pred_res)
                x_start = x_input - pred_res
                x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "noise":
                pred_noise = model_output[1]
                x_start = self.predict_start_from_xinput_noise(
                    x, t, x_input, pred_noise)
                x_start = maybe_clip(x_start)
                pred_res = x_input - x_start
                pred_res = maybe_clip(pred_res)
        elif self.objective == 'pred_x0_noise':
            pred_res = x_input-model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == "pred_noise":
            pred_noise = model_output[0]
            x_start = self.predict_start_from_xinput_noise(
                x, t, x_input, pred_noise)
            x_start = maybe_clip(x_start)
            pred_res = x_input - x_start
            pred_res = maybe_clip(pred_res)
        elif self.objective == "pred_res":
            pred_res = model_output[0]
            pred_res = maybe_clip(pred_res)
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res)
            x_start = x_input - pred_res
            x_start = maybe_clip(x_start)

        return ModelResPrediction(pred_res, pred_noise, x_start)

    def p_mean_variance(self, x_input, x, t):
        preds = self.model_predictions(x_input, x, t)
        pred_res = preds.pred_res
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            pred_res=pred_res, x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x_input, x, t: int):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x_input, x=x, t=batched_times)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, x_input, shape, last=True):
        x_input = x_input[0]

        batch, device = shape[0], self.betas.device

        if self.condition:
            img = x_input+math.sqrt(self.sum_scale) * \
                torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None

        if not last:
            img_list = []

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img, x_start = self.p_sample(x_input, img, t)

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def ddim_sample(self, x_input, shape, last=True, task=None):
        x_input = x_input[0]

        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)#[:num]

        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        if self.condition:
            img = (1-self.delta_cumsum[-1]) * x_input + math.sqrt(self.sum_scale) * torch.randn(shape, device=device)
            # img = (1-self.delta_cumsum[-1]) * x_input + self.betas_cumsum[-1] * torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None
        type = "use_pred_noise"
        # type = "use_x_start"
        last=False

        if not last:
            img_list = []

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            preds = self.model_predictions(x_input, img, time_cond, task)

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum-alpha_cumsum_next
            delta_cumsum = self.delta_cumsum[time]
            delta_cumsum_next = self.delta_cumsum[time_next]
            delta = delta_cumsum-delta_cumsum_next
            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum-betas2_cumsum_next
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = eta * (betas2*betas2_cumsum_next/betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                betas2_cumsum_next-sigma2).sqrt()/betas_cumsum
            q = betas2_cumsum_next / betas2_cumsum

            if eta == 0:
                noise = 0
            else:
                noise = torch.randn_like(img)
            if type == "use_pred_noise":
                img = img - alpha*pred_res + delta*x_input + sigma2.sqrt()*noise
            elif type == "use_x_start":
                # img = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum*img + \
                #     (1-sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*x_start + \
                #     (alpha_cumsum_next-alpha_cumsum*sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*pred_res + \
                #     (delta_cumsum*sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum-delta_cumsum_next)*x_input + \
                #     sigma2.sqrt()*noise
                img = q*img + \
                    (1-q)*x_start + \
                    (alpha_cumsum_next-alpha_cumsum*q)*pred_res + \
                    (delta_cumsum*q-delta_cumsum_next)*x_input + \
                    sigma2.sqrt()*noise
            elif type == "special_eta_0":
                img = img - alpha*pred_res - \
                    (betas_cumsum-betas_cumsum_next)*pred_noise
            elif type == "special_eta_1":
                img = img - alpha*pred_res - betas2/betas_cumsum*pred_noise + \
                    betas*betas2_cumsum_next.sqrt()/betas_cumsum*noise

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
                # img_list = img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def sample(self, x_input=0, batch_size=16, last=True, task=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if self.condition:
            x_input = 2 * x_input - 1
            x_input = x_input.unsqueeze(0)

            batch_size, channels, h, w = x_input[0].shape
            size = (batch_size, channels, h, w)
        else:
            size = (batch_size, channels, image_size, image_size)
        return sample_fn(x_input, size, last=last, task=task)

    def q_sample(self, x_start, x_res, condition, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            x_start+extract(self.alphas_cumsum, t, x_start.shape) * x_res +
            extract(self.betas_cumsum, t, x_start.shape) * noise -
            extract(self.delta_cumsum, t, x_start.shape) * condition
        )

    @property
    def loss_fn(self, loss_type='l1'):
        if loss_type == 'l1':
            return F.l1_loss
        elif loss_type == 'fre':
            return mixed_frequency_loss
        elif loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def p_losses(self, imgs, t, noise=None):
        if isinstance(imgs, list):  # Condition
            x_input = 2 * imgs[1] - 1
            x_start = 2 * imgs[0] - 1  #gt:imgs[0], cond:imgs[1]
            task = imgs[2][0]


        noise = default(noise, lambda: torch.randn_like(x_start))
        x_res = x_input - x_start

        b, c, h, w = x_start.shape

        # noise sample
        x = self.q_sample(x_start, x_res, x_input, t, noise=noise)

        # predict and take gradient step
        if not self.condition:
            x_in = x
        else:
            x_in = torch.cat((x, x_input), dim=1)

        model_out = self.model(x_in,
                               [self.alphas_cumsum[t]*self.num_timesteps,
                                self.betas_cumsum[t]*self.num_timesteps])

        target = []
        if self.objective == 'pred_res_noise':
            target.append(x_res)
            target.append(noise)

            pred_res = model_out[0]
            pred_noise = model_out[1]
        elif self.objective == 'pred_x0_noise':
            target.append(x_start)
            target.append(noise)

            pred_res = x_input-model_out[0]
            pred_noise = model_out[1]
        elif self.objective == "pred_noise":
            target.append(noise)

            pred_noise = model_out[0]

        elif self.objective == "pred_res":
            target.append(x_res)

            pred_res = model_out[0]

        else:
            raise ValueError(f'unknown objective {self.objective}')

        u_loss = False
        if u_loss:
            x_u = self.q_posterior_from_res_noise(pred_res, pred_noise, x, t)
            u_gt = self.q_posterior_from_res_noise(x_res, noise, x, t)
            loss = 10000*self.loss_fn(x_u, u_gt, reduction='none')
            return [loss]
        else:
            loss_list = []
            for i in range(len(model_out)):
                loss = self.loss_fn(model_out[i], target[i], reduction='none')
                loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
                loss_list.append(loss)
            return loss_list

    def forward(self, img, *args, **kwargs):
        if isinstance(img, list):
            b, c, h, w, device, img_size, = * \
                img[0].shape, img[0].device, self.image_size
        else:
            b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(img, t, *args, **kwargs)
    # def forward(self, x_input_sample, batches, last, file_):
    #     # profile
    #     return self.sample(x_input_sample, batch_size=batches, last=last, task=file_)

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        opts,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=5,#1000
        num_unet=1,
        num_samples=25,
        results_folder='/group/yaoliang/DIFF/ckpt_universal/wavelet_pretrained',
        amp=False,
        fp16=False,
        split_batches=True,
        convert_image_to=None,
        condition=False,
        sub_dir=False,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.sub_dir = sub_dir
        self.accelerator.native_amp = amp
        self.num_unet = num_unet
        self.model = diffusion_model

        # assert has_int_squareroot(
        #     num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.condition = condition

        if self.condition:
            if opts.phase == "train":
                self.dl_fog = cycle(self.accelerator.prepare(DataLoader(dataset[0], batch_size=4, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=4)))
                self.dl_light = cycle(self.accelerator.prepare(DataLoader(dataset[1], batch_size=1, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=4)))
                self.dl_rain = cycle(self.accelerator.prepare(DataLoader(dataset[2], batch_size=2, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=4)))
                self.dl_snow = cycle(self.accelerator.prepare(DataLoader(dataset[3], batch_size=2, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=4)))
                self.dl_blur = cycle(self.accelerator.prepare(DataLoader(dataset[4], batch_size=1, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=4)))
            else:
                self.sample_dataset = dataset


        # optimizer
        self.opt0 = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay,
                           update_every=ema_update_every)

            self.set_results_folder(results_folder)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt0 = self.accelerator.prepare(self.model, self.opt0)

        device = self.accelerator.device
        self.device = device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt0': self.opt0.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    # def load(self, milestone):
    #     path = Path(self.results_folder / f'model-{milestone}.pt')
    #     if path.exists():
    #         data = torch.load(str(path), map_location=self.device,)
    #         self.model = self.accelerator.unwrap_model(self.model)



    #         self.model.load_state_dict(data['model'])
    #         self.step = data['step']

    #         self.opt0.load_state_dict(data['opt0'])
    #         self.opt0.param_groups[0]['capturable'] = True
    #         self.ema.load_state_dict(data['ema'])

    #         if exists(self.accelerator.scaler) and exists(data['scaler']):
    #             self.accelerator.scaler.load_state_dict(data['scaler'])
    #         print("load了")
    #         print("load model - "+str(path))
    
    # def load(self, milestone):
    #     path = Path(self.results_folder / f'model-{milestone}.pt')
    #     print(self.results_folder)
    #     if path.exists():
    #         data = torch.load(str(path), map_location=self.device)
    #         self.model = self.accelerator.unwrap_model(self.model)
    #
    #         # 加载模型状态字典
    #         self.model.load_state_dict(data['model'], strict=False)
    #
    #         self.step = data['step']
    #
    #         # 部分加载优化器状态字典
    #         opt0_state_dict = data['opt0']
    #         current_param_groups = self.opt0.param_groups
    #
    #         # 创建新的参数组以匹配当前优化器
    #         new_opt_state_dict = {k: v for k, v in opt0_state_dict.items() if k != 'param_groups'}
    #         new_opt_state_dict['param_groups'] = current_param_groups
    #
    #         self.opt0.load_state_dict(new_opt_state_dict)
    #         self.opt0.param_groups[0]['capturable'] = False
    #
    #         # 部分加载EMA状态字典
    #         ema_state_dict = data['ema']
    #         current_ema_state_dict = self.ema.state_dict()
    #
    #         partial_ema_state_dict = {k: v for k, v in ema_state_dict.items() if k in current_ema_state_dict}
    #         current_ema_state_dict.update(partial_ema_state_dict)
    #         self.ema.load_state_dict(current_ema_state_dict, strict=False)
    #
    #         # 加载scaler状态字典（如果存在）
    #         if exists(self.accelerator.scaler) and exists(data['scaler']):
    #             self.accelerator.scaler.load_state_dict(data['scaler'])
    #         print("load model - "+str(path))
    # def load(self, milestone):
    #     path = Path(self.results_folder / f'model-{milestone}.pt')
    #     print(f"尝试加载模型: {path}")
    #
    #     # 修改为从其他路径加载预训练模型
    #     pretrained_path = Path("/group/yaoliang/DIFF/ckpt_universal/wavelet_pretrained") / f'model-{milestone}.pt'
    #     print(f"从预训练路径加载: {pretrained_path}")
    #
    #     if pretrained_path.exists():
    #         print(f"找到预训练模型: {pretrained_path}")
    #         data = torch.load(str(pretrained_path), map_location=self.device)
    #         self.model = self.accelerator.unwrap_model(self.model)
    #
    #         # 加载模型状态字典
    #         self.model.load_state_dict(data['model'], strict=False)
    #
    #         # 重置步数从0开始
    #         # self.step = data['step']  # 注释掉这行
    #         self.step = 0  # 从0开始重新计数
    #         print(f"重置训练步数为0，将训练{self.train_num_steps}步")
    #
    #         # 部分加载优化器状态字典
    #         opt0_state_dict = data['opt0']
    #         current_param_groups = self.opt0.param_groups
    #
    #         # 创建新的参数组以匹配当前优化器
    #         new_opt_state_dict = {k: v for k, v in opt0_state_dict.items() if k != 'param_groups'}
    #         new_opt_state_dict['param_groups'] = current_param_groups
    #
    #         self.opt0.load_state_dict(new_opt_state_dict)
    #         self.opt0.param_groups[0]['capturable'] = False
    #
    #         # 部分加载EMA状态字典
    #         ema_state_dict = data['ema']
    #         current_ema_state_dict = self.ema.state_dict()
    #
    #         partial_ema_state_dict = {k: v for k, v in ema_state_dict.items() if k in current_ema_state_dict}
    #         current_ema_state_dict.update(partial_ema_state_dict)
    #         self.ema.load_state_dict(current_ema_state_dict, strict=False)
    #
    #         # 加载scaler状态字典（如果存在）
    #         if exists(self.accelerator.scaler) and exists(data['scaler']):
    #             self.accelerator.scaler.load_state_dict(data['scaler'])
    #         print(f"成功加载预训练模型: {pretrained_path}")
    #     else:
    #         print(f"未找到预训练模型: {pretrained_path}")

    def load(self, milestone):
        # path = Path("/group/yaoliang/DIFF/ckpt_universal/DiffUIR-B1224-pre4") / f'model-{milestone}.pt'
        path = Path(self.results_folder / f'model-{milestone}.pt')
        # print(self.results_folder)
        if path.exists():
            data = torch.load(str(path), map_location=self.device)
            self.model = self.accelerator.unwrap_model(self.model)

            # 加载模型状态字典
            self.model.load_state_dict(data['model'], strict=False)

            self.step = data['step']

            # 部分加载优化器状态字典
            opt0_state_dict = data['opt0']
            current_param_groups = self.opt0.param_groups

            # 创建新的参数组以匹配当前优化器
            new_opt_state_dict = {k: v for k, v in opt0_state_dict.items() if k != 'param_groups'}
            new_opt_state_dict['param_groups'] = current_param_groups

            self.opt0.load_state_dict(new_opt_state_dict)
            self.opt0.param_groups[0]['capturable'] = False

            # 部分加载EMA状态字典
            ema_state_dict = data['ema']
            current_ema_state_dict = self.ema.state_dict()

            partial_ema_state_dict = {k: v for k, v in ema_state_dict.items() if k in current_ema_state_dict}
            current_ema_state_dict.update(partial_ema_state_dict)
            self.ema.load_state_dict(current_ema_state_dict, strict=False)

            # 加载scaler状态字典（如果存在）
            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])
            print("load model - " + str(path))

    def train(self):
        if torch.cuda.device_count() > 1:  # 含有多张GPU的卡
            self.model = nn.DataParallel(self.model)
            print("使用多GPU训练")
        # self.load(1220)

        # 添加调试信息
        print(f"开始训练: 当前步数={self.step}, 目标步数={self.train_num_steps}")
        print(f"训练批次大小={self.batch_size}, 梯度累积={self.gradient_accumulate_every}")
        print(f"结果保存路径: {self.results_folder}")

        accelerator = self.accelerator
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                if self.step % 100 == 0:
                    print(f"当前训练步数: {self.step}/{self.train_num_steps}")
                total_loss = [0]

                for _ in range(self.gradient_accumulate_every):
                    if self.condition:
                        batch1 = next(self.dl_fog)
                        batch2 = next(self.dl_light)
                        batch3 = next(self.dl_rain)
                        batch4 = next(self.dl_snow)
                        batch5 = next(self.dl_blur)

                        data = {}
                        for k, v in batch1.items():
                            if 'path' in k:
                                data[k] = batch1[k] + batch2[k] + batch3[k]+ batch4[k] + batch5[k]
                            else:
                                data[k] = torch.cat([batch1[k], batch2[k], batch3[k], batch4[k], batch5[k]], dim=0)
                        gt = data["gt"].to(self.device)
                        cond_input = data["adap"].to(self.device)
                        task = data["A_paths"]
                        data = [gt, cond_input, task]
                    else:
                        data = next(self.dl)
                        data = data[0] if isinstance(data, list) else data
                        data = data.to(self.device)

                    # with self.accelerator.autocast():
                    #     loss = self.model(data)
                    #     for i in range(self.num_unet):
                    #         loss[i] = loss[i] / self.gradient_accumulate_every
                    #         total_loss[i] = total_loss[i]+ loss[i].sum()

                    # for i in range(self.num_unet):
                    #     self.accelerator.backward(loss[i].item())

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        for i in range(self.num_unet):
                            loss[i] = loss[i] / self.gradient_accumulate_every
                            loss[i] = loss[i].mean()  #并行
                            total_loss[i] = total_loss[i] + loss[i].item()

                    for i in range(self.num_unet):
                        self.accelerator.backward(loss[i])


                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                accelerator.wait_for_everyone()

                self.opt0.step()
                self.opt0.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()
                    if self.step % (self.save_and_sample_every*10) == 0:

                        milestone = self.step // self.save_and_sample_every
                        self.save(milestone)
                        print(milestone)

                pbar.set_description(f'loss_unet0: {total_loss[0]:.4f}')
                pbar.update(1)

        accelerator.print('training complete')

    def test(self, sample=False, last=True, FID=False):
        self.ema.ema_model.init()
        self.ema.to(self.device)
        print("test start")
        if self.condition:
            self.ema.ema_model.eval()
            loader = DataLoader(
                dataset=self.sample_dataset,
                batch_size=1,
                num_workers=4,
                prefetch_factor=4

            )
            i = 0
            cnt = 0
            opt_metric = {
                'psnr': {
                    'type': 'calculate_psnr',
                    'crop_border': 0,
                    'test_y_channel': True
                    },
                'ssim': {
                    'type': 'calculate_ssim',
                    'crop_border': 0,
                    'test_y_channel': True
                    }
                }
            self.metric_results = {
                metric: 0
                for metric in opt_metric.keys()
            }
            tran = transforms.ToTensor()
            for items in loader:
                if self.condition:
                    file_ = items["A_paths"][0]
                    file_name = file_.split('/')[-4]
                else:
                    file_name = f'{i}.png'

                i += 1

                start_time = time.time()
                with torch.no_grad():
                    batches = self.num_samples

                    data = items
                    x_input_sample = data["adap"].to(self.device)
                    gt = data["gt"].to(self.device)

                    all_images_list = list(self.ema.ema_model.sample(
                            x_input_sample, batch_size=batches, last=last, task=file_))
                print(time.time()-start_time)
                all_images_list = [all_images_list[-1]]
                all_images = torch.cat(all_images_list, dim=0)

                if last:
                    nrow = int(math.sqrt(self.num_samples))
                else:
                    nrow = all_images.shape[0]
                save_path = str(self.results_folder / file_name)
                os.makedirs(save_path, exist_ok=True)
                full_path = os.path.join(save_path, file_.split('/')[-1]).replace('_fake_B','')
                utils.save_image(all_images, full_path, nrow=nrow)
                print("test-save "+full_path)

            #calculate the metric

                sr_img = tensor2img(all_images, rgb2bgr=True)
                gt_img = tensor2img(gt, rgb2bgr=True)
                opt_metric_ = {
                    'psnr': {
                        'type': 'calculate_psnr',
                        'crop_border': 0,
                        'test_y_channel': True
                        },
                    'ssim': {
                        'type': 'calculate_ssim',
                        'crop_border': 0,
                        'test_y_channel': True
                        }
                    }
                for name, opt_ in opt_metric_.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)

                cnt += 1

            current_metric = {}
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric[metric] = self.metric_results[metric]
            print(current_metric['psnr'])
            print(current_metric['ssim'])

        print("test end")

    def test_one(self, sample=False, last=True, FID=False):

        self.ema.ema_model.init()
        self.ema.to(self.device)
        print("test start")
        if self.condition:
            self.ema.ema_model.eval()

            input_image = Image.open(self.sample_dataset).convert('RGB')
            
            # input_image = cv2.cvtColor(np.asarray(input_image), cv2.COLOR_RGB2BGR)
            # input_image = cv2equalizeHist(input_image) 
            # input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            
            preprocess = transforms.Compose([
                transforms.ToTensor(),              # 将图像转换为PyTorch张量
            ])
            x_input_sample = preprocess(input_image).unsqueeze(0).to(self.device)
 
            batches = self.num_samples
            temp=[]
         
            
            all_images_list = list(self.ema.ema_model.sample(x_input_sample, batch_size=batches, last=last, task=None))
            all_images_list = [all_images_list[-1]]
                        
            all_images = torch.cat(all_images_list, dim=0)

            if last:
                nrow = int(math.sqrt(self.num_samples))
            else:
                nrow = all_images.shape[0]
            utils.save_image(all_images, './temp.png', nrow=nrow)
                
        print("test end")   



   
    # def test_all(self, folder_path, sample=False, last=True):
    #     self.ema.ema_model.init()
    #     self.ema.to(self.device)
    #     print("test start")
        
    #     if self.condition:
    #         self.ema.ema_model.eval()

    #         # 获取文件夹中所有图片文件的路径
    #         image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]

    #         # 定义预处理操作
    #         preprocess = transforms.Compose([
    #             transforms.ToTensor(),
    #         ])

    #         psnr_values = []
    #         ssim_values = []
            
    #         for image_path in image_paths:
    #             # 打开并预处理图片
    #             input_image = Image.open(image_path).convert('RGB')
    #             x_input_sample = preprocess(input_image).unsqueeze(0).to(self.device)
    
    #             batches = self.num_samples
    #             temp = []
            
    #             # 生成图像
    #             all_images_list = list(self.ema.ema_model.sample(x_input_sample, batch_size=batches, last=last, task=None))
    #             all_images_list = [all_images_list[-1]]
    #             all_images = torch.cat(all_images_list, dim=0)

    #             # 保存生成的图像
    #             if last:
    #                 nrow = int(math.sqrt(self.num_samples))
    #             else:
    #                 nrow = all_images.shape[0]
                
    #             # 保存处理后的图像
    #             utils.save_image(all_images, './temp.png', nrow=nrow)
                
    #             # 计算PSNR和SSIM
    #             generated_image = all_images.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    #             original_image = x_input_sample.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                
    #             psnr = peak_signal_noise_ratio(original_image, generated_image, data_range=1.0)
                
    #             # 获取图像的大小，计算合适的win_size
    #             min_side = min(original_image.shape[0], original_image.shape[1])
    #             win_size = min(7, min_side)
                
    #             ssim = structural_similarity(original_image, generated_image, 
    #                                         multichannel=True, win_size=win_size,
    #                                         channel_axis=-1, data_range=1.0)

    #             psnr_values.append(psnr)
    #             ssim_values.append(ssim)

    #             print(f"Image: {os.path.basename(image_path)}, PSNR: {psnr}, SSIM: {ssim}")
            
    #         print(f"Average PSNR: {sum(psnr_values)/len(psnr_values)}")
    #         print(f"Average SSIM: {sum(ssim_values)/len(ssim_values)}")
                    
    #     print("test end")
   
    def test_all(self, folder_path, output_folder, sample=False, last=True):
        self.ema.ema_model.init()
        self.ema.to(self.device)
        print("test start")
        
        if self.condition:
            self.ema.ema_model.eval()

            # 创建保存去噪后图像的文件夹
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # 获取文件夹中所有图片文件的路径
            image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]

            # 定义预处理操作
            preprocess = transforms.Compose([
                transforms.ToTensor(),
            ])

            psnr_values = []
            ssim_values = []
            
            for image_path in image_paths:
                # 打开并预处理图片
                input_image = Image.open(image_path).convert('RGB')
                x_input_sample = preprocess(input_image).unsqueeze(0).to(self.device)
    
                batches = self.num_samples
                temp = []
            
                # 生成图像
                all_images_list = list(self.ema.ema_model.sample(x_input_sample, batch_size=batches, last=last, task=None))
                all_images_list = [all_images_list[-1]]
                all_images = torch.cat(all_images_list, dim=0)

                # 保存生成的图像
                if last:
                    nrow = int(math.sqrt(self.num_samples))
                else:
                    nrow = all_images.shape[0]
                
                # 生成的图像保存到指定文件夹
                output_image_path = os.path.join(output_folder, os.path.basename(image_path))
                utils.save_image(all_images, output_image_path, nrow=nrow)
                
                # 计算PSNR和SSIM
                generated_image = all_images.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                original_image = x_input_sample.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                
                psnr = peak_signal_noise_ratio(original_image, generated_image, data_range=1.0)
                
                # 获取图像的大小，计算合适的win_size
                min_side = min(original_image.shape[0], original_image.shape[1])
                win_size = min(7, min_side)
                
                ssim = structural_similarity(original_image, generated_image, 
                                            multichannel=True, win_size=win_size,
                                            channel_axis=-1, data_range=1.0)

                psnr_values.append(psnr)
                ssim_values.append(ssim)

                print(f"Image: {os.path.basename(image_path)}, PSNR: {psnr}, SSIM: {ssim}")
            
            print(f"Average PSNR: {sum(psnr_values)/len(psnr_values)}")
            print(f"Average SSIM: {sum(ssim_values)/len(ssim_values)}")
                    
        print("test end")

    # def test_true_all(self, folder_path, gt_folder_path, output_folder, sample=False, last=True):
    #     self.ema.ema_model.init()
    #     self.ema.to(self.device)
    #     print("test start")
    #
    #     if self.condition:
    #         self.ema.ema_model.eval()
    #
    #         # 创建保存去噪后图像的文件夹
    #         if not os.path.exists(output_folder):
    #             os.makedirs(output_folder)
    #
    #         # 获取文件夹中所有图片文件的路径
    #         image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if
    #                        fname.lower().endswith(('png', 'jpg', 'jpeg'))]
    #
    #         # 定义预处理操作
    #         preprocess = transforms.Compose([
    #             transforms.ToTensor(),
    #         ])
    #
    #         psnr_values = []
    #         ssim_values = []
    #
    #         for image_path in image_paths:
    #             # 打开并预处理输入图片
    #             input_image = Image.open(image_path).convert('RGB')
    #             x_input_sample = preprocess(input_image).unsqueeze(0).to(self.device)
    #
    #             # 加载对应的ground truth图像
    #             gt_image_path = os.path.join(gt_folder_path, os.path.basename(image_path))
    #             if os.path.exists(gt_image_path):
    #                 gt_image = Image.open(gt_image_path).convert('RGB')
    #                 gt_tensor = preprocess(gt_image).to(self.device)
    #             else:
    #                 print(f"警告: 未找到对应的ground truth图像: {gt_image_path}")
    #                 continue
    #
    #             batches = self.num_samples
    #             temp = []
    #
    #             # 生成图像
    #             all_images_list = list(
    #                 self.ema.ema_model.sample(x_input_sample, batch_size=batches, last=last, task=None))
    #             all_images_list = [all_images_list[-1]]
    #             all_images = torch.cat(all_images_list, dim=0)
    #
    #             # 保存生成的图像
    #             if last:
    #                 nrow = int(math.sqrt(self.num_samples))
    #             else:
    #                 nrow = all_images.shape[0]
    #
    #             # 生成的图像保存到指定文件夹
    #             output_image_path = os.path.join(output_folder, os.path.basename(image_path))
    #             utils.save_image(all_images, output_image_path, nrow=nrow)
    #
    #             # 计算PSNR和SSIM (使用ground truth和生成图像)
    #             generated_image = all_images.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    #             ground_truth = gt_tensor.cpu().numpy().transpose(1, 2, 0)
    #
    #             psnr = peak_signal_noise_ratio(ground_truth, generated_image, data_range=1.0)
    #
    #             # 获取图像的大小，计算合适的win_size
    #             min_side = min(ground_truth.shape[0], ground_truth.shape[1])
    #             win_size = min(7, min_side)
    #
    #             ssim = structural_similarity(ground_truth, generated_image,
    #                                          multichannel=True, win_size=win_size,
    #                                          channel_axis=-1, data_range=1.0)
    #
    #             psnr_values.append(psnr)
    #             ssim_values.append(ssim)
    #
    #             print(f"Image: {os.path.basename(image_path)}, PSNR: {psnr}, SSIM: {ssim}")
    #
    #         print(f"Average PSNR: {sum(psnr_values) / len(psnr_values)}")
    #         print(f"Average SSIM: {sum(ssim_values) / len(ssim_values)}")
    #
    #     print("test end")

    def test_all_csv(self, folder_path, gt_folder_path, output_folder, sample=False, last=True):
        self.ema.ema_model.init()
        self.ema.to(self.device)
        print("test start")

        if self.condition:
            self.ema.ema_model.eval()

            # 创建保存去噪后图像的文件夹
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # 获取文件夹中所有图片文件的路径
            image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if
                           fname.lower().endswith(('png', 'jpg', 'jpeg'))]

            # 定义预处理操作
            preprocess = transforms.Compose([
                transforms.ToTensor(),
            ])

            psnr_values = []
            ssim_values = []

            # 创建CSV文件并写入表头
            csv_path = os.path.join(os.path.dirname(output_folder), 'metrics_results.csv')
            with open(csv_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Image', 'PSNR', 'SSIM'])

            for image_path in image_paths:
                # 打开并预处理输入图片
                input_image = Image.open(image_path).convert('RGB')
                x_input_sample = preprocess(input_image).unsqueeze(0).to(self.device)

                # 加载对应的ground truth图像
                gt_image_path = os.path.join(gt_folder_path, os.path.basename(image_path))
                if os.path.exists(gt_image_path):
                    gt_image = Image.open(gt_image_path).convert('RGB')
                    gt_tensor = preprocess(gt_image)
                else:
                    print(f"警告: 未找到对应的ground truth图像: {gt_image_path}")
                    continue

                batches = self.num_samples

                # 生成图像
                all_images_list = list(
                    self.ema.ema_model.sample(x_input_sample, batch_size=batches, last=last, task=None))
                all_images_list = [all_images_list[-1]]
                all_images = torch.cat(all_images_list, dim=0)

                # 保存生成的图像
                if last:
                    nrow = int(math.sqrt(self.num_samples))
                else:
                    nrow = all_images.shape[0]

                # 生成的图像保存到指定文件夹
                output_image_path = os.path.join(output_folder, os.path.basename(image_path))
                utils.save_image(all_images, output_image_path, nrow=nrow)

                # 计算PSNR和SSIM (使用ground truth和生成图像)
                generated_image = all_images.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                ground_truth = gt_tensor.cpu().numpy().transpose(1, 2, 0)

                psnr = peak_signal_noise_ratio(ground_truth, generated_image, data_range=1.0)

                # 获取图像的大小，计算合适的win_size
                min_side = min(ground_truth.shape[0], ground_truth.shape[1])
                win_size = min(7, min_side)

                ssim = structural_similarity(ground_truth, generated_image,
                                             multichannel=True, win_size=win_size,
                                             channel_axis=-1, data_range=1.0)

                psnr_values.append(psnr)
                ssim_values.append(ssim)

                # 将结果写入CSV文件
                with open(csv_path, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([os.path.basename(image_path), f"{psnr:.4f}", f"{ssim:.4f}"])

                print(f"Image: {os.path.basename(image_path)}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")

            if psnr_values:
                avg_psnr = sum(psnr_values) / len(psnr_values)
                avg_ssim = sum(ssim_values) / len(ssim_values)
                print(f"Average PSNR: {avg_psnr:.4f}")
                print(f"Average SSIM: {avg_ssim:.4f}")

                # 将平均值也写入CSV
                with open(csv_path, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(['Average', f"{avg_psnr:.4f}", f"{avg_ssim:.4f}"])
            else:
                print("没有成功计算任何图像的PSNR和SSIM")

        print("test end")

    def test_all_csv_fps(self, folder_path, gt_folder_path, output_folder, sample=False, last=True):
        self.ema.ema_model.init()
        self.ema.to(self.device)
        print("test start")

        if self.condition:
            self.ema.ema_model.eval()

            # 创建保存去噪后图像的文件夹
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # 获取文件夹中所有图片文件的路径
            image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if
                           fname.lower().endswith(('png', 'jpg', 'jpeg'))]

            # 定义预处理操作
            preprocess = transforms.Compose([
                transforms.ToTensor(),
            ])

            psnr_values = []
            ssim_values = []
            fps_values = []  # 添加FPS列表

            # 创建CSV文件并写入表头
            csv_path = os.path.join(os.path.dirname(output_folder), 'metrics_results_UDC_ori_gray.csv')
            with open(csv_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Image', 'PSNR', 'SSIM', 'FPS'])  # 添加FPS列

            for image_path in image_paths:
                # 打开并预处理输入图片
                input_image = Image.open(image_path).convert('RGB')
                x_input_sample = preprocess(input_image).unsqueeze(0).to(self.device)

                # 加载对应的ground truth图像
                gt_image_path = os.path.join(gt_folder_path, os.path.basename(image_path))
                if os.path.exists(gt_image_path):
                    gt_image = Image.open(gt_image_path).convert('RGB')
                    gt_tensor = preprocess(gt_image)
                else:
                    print(f"警告: 未找到对应的ground truth图像: {gt_image_path}")
                    continue

                batches = self.num_samples

                # 测量处理时间
                torch.cuda.synchronize()  # 确保GPU操作完成
                start_time = time.time()

                # 生成图像
                all_images_list = list(
                    self.ema.ema_model.sample(x_input_sample, batch_size=batches, last=last, task=None))
                all_images_list = [all_images_list[-1]]
                all_images = torch.cat(all_images_list, dim=0)

                torch.cuda.synchronize()  # 确保GPU操作完成
                end_time = time.time()

                # 计算FPS
                process_time = end_time - start_time
                fps = 1.0 / process_time
                fps_values.append(fps)

                # 保存生成的图像
                if last:
                    nrow = int(math.sqrt(self.num_samples))
                else:
                    nrow = all_images.shape[0]

                # 生成的图像保存到指定文件夹
                output_image_path = os.path.join(output_folder, os.path.basename(image_path))
                utils.save_image(all_images, output_image_path, nrow=nrow)

                # 计算PSNR和SSIM (使用ground truth和生成图像)
                generated_image = all_images.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                ground_truth = gt_tensor.cpu().numpy().transpose(1, 2, 0)

                psnr = peak_signal_noise_ratio(ground_truth, generated_image, data_range=1.0)

                # 获取图像的大小，计算合适的win_size
                min_side = min(ground_truth.shape[0], ground_truth.shape[1])
                win_size = min(7, min_side)

                ssim = structural_similarity(ground_truth, generated_image,
                                             multichannel=True, win_size=win_size,
                                             channel_axis=-1, data_range=1.0)

                psnr_values.append(psnr)
                ssim_values.append(ssim)

                # 将结果写入CSV文件
                with open(csv_path, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([os.path.basename(image_path), f"{psnr:.4f}", f"{ssim:.4f}", f"{fps:.4f}"])

                print(f"Image: {os.path.basename(image_path)}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, FPS: {fps:.4f}")

            if psnr_values:
                avg_psnr = sum(psnr_values) / len(psnr_values)
                avg_ssim = sum(ssim_values) / len(ssim_values)
                avg_fps = sum(fps_values) / len(fps_values)  # 计算平均FPS
                print(f"Average PSNR: {avg_psnr:.4f}")
                print(f"Average SSIM: {avg_ssim:.4f}")
                print(f"Average FPS: {avg_fps:.4f}")  # 打印平均FPS

                # 将平均值也写入CSV
                with open(csv_path, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(['Average', f"{avg_psnr:.4f}", f"{avg_ssim:.4f}", f"{avg_fps:.4f}"])  # 添加平均FPS
            else:
                print("没有成功计算任何图像的PSNR和SSIM")

        print("test end")

    def set_results_folder(self, path):
        self.results_folder = Path(path)
        if not self.results_folder.exists():
            os.makedirs(self.results_folder)
