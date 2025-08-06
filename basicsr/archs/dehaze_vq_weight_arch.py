import cv2
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn as nn
import numpy as np
import math

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from .fourier_block_arch import GlobalNet_Fusion

from .network_swinir import RSTB
from .ridcp_utils import ResBlock, CombineQuantBlock 
from .vgg_arch import VGGFeatureExtractor

import torch
import torch.nn as nn

from basicsr.utils.img_util import tensor2img,img2tensor
from basicsr.utils.logger import get_root_logger


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        ffted = torch.stack((ffted.real, ffted.imag), -1)

        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        real_part = ffted[..., 0]
        imaginary_part = ffted[..., 1]
        complex_tensor = real_part + 1j * imaginary_part
        output = torch.fft.irfft2(complex_tensor, dim=(-2, -1), s=r_size[2:], norm="ortho")

        return output

##########################################################################
##---------- Fast Fourier Block-----------------------
class FFB(nn.Module):
    def __init__(self, nc):
        super(FFB, self).__init__()
        self.block = nn.Sequential(
                        nn.Conv2d(nc,nc,3,1,1),
                        nn.LeakyReLU(0.1),
                        nn.Conv2d(nc, nc, 3, 1, 1)
        )
        self.glocal = FourierUnit(nc, nc)
        self.cat = nn.Conv2d(2*nc, nc, 1, 1, 0)

    def forward(self, x):
        conv = self.block(x)
        conv = conv + x
        glocal = self.glocal(x)
        out = torch.cat([conv, glocal], 1)
        out = self.cat(out) + x
        return out


class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [FFB(nc=dim) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

##########################################################################
##---------- Resizing Modules ----------
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias)
        )

    def forward(self, x):
        return self.bot(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()

        modules_body = []
        modules_body.append(Down(in_channels, out_channels))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        )

    def forward(self, x):
        return self.bot(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()

        modules_body = []
        modules_body.append(Up(in_channels, out_channels))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out

class Global_Local_Fusion(nn.Module):
    def __init__(self, in_chans=3, feat=24, out_chans=3):
        super(Global_Local_Fusion, self).__init__()

        self.conv_in = nn.Conv2d(in_chans, feat, 1, 1)

        self.pa_att = PALayer(channel=feat)
        self.cha_att = CALayer(channel=feat*2)
        self.post = nn.Conv2d(feat*2, feat, 1, 1)
        self.conv_out = nn.Conv2d(feat, out_chans, kernel_size=1, padding=0)


    def forward(self, diff_img, global_img):

        diff_img = self.conv_in(diff_img)
        global_img = self.conv_in(global_img)
        res_img = diff_img + global_img
        pa_map = self.pa_att(res_img)
        pa_res = pa_map

        cat_f = torch.cat([diff_img, global_img], 1)
        cha_res = self.post(self.cha_att(cat_f)) + pa_res
        out_img = self.conv_out(cha_res)   #ours

        return out_img


class GlobalNet_Fusion(nn.Module):
    def __init__(self,
                 *,
                 in_chans=3, out_chans=3,
                 embed_dims=[24, 48, 96, 48, 24], #24, 48, 96, 48, 24
                 depths=[1, 1, 2, 1, 1],            #1, 1, 2, 1, 1
                 **ignore_kwargs):

        super(GlobalNet_Fusion, self).__init__()

        self.conv1 = nn.Conv2d(in_chans, embed_dims[0], kernel_size=3, padding=1)

        # backbone
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0])

        self.down1 = DownSample(embed_dims[0], embed_dims[1])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1])

        self.down2 = DownSample(embed_dims[1], embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2])

        self.up1 = UpSample(embed_dims[2], embed_dims[3])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3])

        self.up2 = UpSample(embed_dims[3], embed_dims[4])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4])

        self.conv2 = nn.Conv2d(embed_dims[4], out_chans, kernel_size=3, padding=1)

        self.global_local_fusion = Global_Local_Fusion(feat=96)

    def forward_features(self, x):
        x = self.conv1(x)   # 3*3Conv
        x = self.layer1(x)
        skip1 = x

        x = self.down1(x)
        x = self.layer2(x)
        skip2 = x

        x = self.down2(x)
        x = self.layer3(x)
        x = self.up1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.up2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.conv2(x)
        return x




    def forward(self, x,y):

        global_img = self.forward_features(x)
        out_img = self.global_local_fusion(y,global_img)


        return out_img

# class Dehaze(nn.Module):
#     def __init__(self):
#         super(Dehaze, self).__init__()
#
#         self.relu = nn.LeakyReLU(0.2, inplace=True)
#
#         self.tanh = nn.Tanh()
#
#         self.refine1 = nn.Conv2d(6, 20, kernel_size=3, stride=1, padding=1)
#         self.refine2 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)
#
#         self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
#         self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
#         self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
#         self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
#
#         self.refine3 = nn.Conv2d(20 + 4, 3, kernel_size=3, stride=1, padding=1)
#
#         self.upsample = F.upsample_nearest
#
#         self.batch1 = nn.InstanceNorm2d(100, affine=True)
#
#     def forward(self, x):
#         dehaze = self.relu((self.refine1(x)))
#         dehaze = self.relu((self.refine2(dehaze)))
#         shape_out = dehaze.data.size()
#         # print(shape_out)
#         shape_out = shape_out[2:4]
#
#         x101 = F.avg_pool2d(dehaze, 32)
#
#         x102 = F.avg_pool2d(dehaze, 16)
#
#         x103 = F.avg_pool2d(dehaze, 8)
#
#         x104 = F.avg_pool2d(dehaze, 4)
#         x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
#         x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
#         x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
#         x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)
#
#         dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
#         dehaze = self.tanh(self.refine3(dehaze))
#
#         return dehaze

class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.
    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.
    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')
        
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)

class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, weight_path='/home/lhp/code/RIDCP_dehazing/pretrained_models/weight_for_matching_dehazing_Flickr.pth', beta=0.25, LQ_stage=False, use_weight=True, weight_alpha=1.0):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(e_dim)
        self.LQ_stage = LQ_stage
        self.beta = beta
        self.use_weight = use_weight
        self.weight_alpha = weight_alpha
        if self.use_weight:
            self.weight = nn.Parameter(torch.load(weight_path))
            self.weight.requires_grad = False 
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
    
    def dist(self, x, y):
        if x.shape == y.shape:
            return (x - y) ** 2
        else:
            return torch.sum(x ** 2, dim=1, keepdim=True) + \
                    torch.sum(y**2, dim=1) - 2 * \
                    torch.matmul(x, y.t())
    
    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h*w, c)
        y = y.reshape(b, h*w, c)

        gmx = x.transpose(1, 2) @ x / (h*w)
        gmy = y.transpose(1, 2) @ y / (h*w)
    
        return (gmx - gmy).square().mean()

    def forward(self,z,gt_indices=None, current_iter=None, weight_alpha=None, input=None ):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization. 
        """
        HDP=True
        ts = None
        if HDP:
            if input is not None:

                from basicsr.transmission_map import compute_transmission_map
                input = tensor2img(input)
                input = cv2.resize(input, (z.shape[3], z.shape[2]))
                cv2.imwrite('show_reshape_lq.jpg', input)
                input = img2tensor(input)
                ts = compute_transmission_map(input).cuda(z.device)
                ts_flattened = ts.flatten()
                # print(ts_flattened)
                # print(ts_flattened.shape)
        else:
            ts = 1


        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        codebook = self.embedding.weight

        d = self.dist(z_flattened, codebook)
        if self.use_weight and self.LQ_stage:
            if weight_alpha is not None:
                self.weight_alpha = weight_alpha
            d = d * torch.exp(self.weight_alpha * self.weight)
            if weight_alpha is not None:
                # self.weight_alpha = weight_alpha*(-203.125) +80.3125        #-21.5 = 0.5
                # torch.set_printoptions(precision=4, threshold=float('inf'))  # 不截断输出
                # w = torch.exp(self.weight_alpha * self.weight)
                # w2 = 1
                # if weight_alpha <0:
                #     for i in range(d.shape[0]):
                #         ts_ref = torch.exp(1-ts_flattened)
                #         w = torch.exp((self.weight_alpha*ts_ref[i]) * self.weight)
                #         d[i] = d[i] * w
                # else:
                #     for i in range(d.shape[0]):
                #         ts_ref = torch.exp(ts_flattened-1)
                #         w = torch.exp((self.weight_alpha*ts_ref[i]) * self.weight)
                #         d[i] = d[i] * w

                #HDP_1.0
                for i in range(d.shape[0]):
                    x = ts_flattened[i]

                    if x == 0.5:
                        ts_mapping = torch.tensor(0.0, device=x.device)  # 保持在相同设备上
                    elif 0 < x < 0.5:
                        ts_mapping = 30 * torch.exp(torch.tensor(1.0, device=x.device) - 2 * (x)) - 30
                    elif 0.5 < x <= 1:
                        ts_mapping = 25 - 25 * torch.exp(2 * (x) - torch.tensor(1.0, device=x.device))
                    else:
                        raise ValueError("x必须在(0, 1)范围内")

                    # 确保所有操作都使用PyTorch函数，保持在计算图中
                    w = torch.exp((self.weight_alpha - ts_mapping) * self.weight)
                    d[i] = d[i] * w




                # shangxia fen
                # half_rows = d.shape[0] // 2
                # d[(half_rows-1):] = d[(half_rows-1):] * w2
                # d[:(half_rows-1)] = d[:(half_rows-1)] * w

                # print(w.shape)
                # print(self.weight)

                #zuoyou fen
                # for i in range(d.shape[0]) :
                #     if i%z.shape[2]<(z.shape[2]//2):
                #         d[i] = d[i] * w

                #fenduan
                # block = 10
                # block_width = z.shape[2] // block  # 计算每个块宽度 (64//4=16)
                #
                # for i in range(d.shape[0]):
                #     col_index = i % z.shape[2]  # 当前像素在行中的列位置(0-63)
                #     block_num = col_index // block_width  # 计算所属块编号(0-3)
                #
                #
                #     if block_num in (0,2,4,8):
                #         d[i] = d[i] * w

                # #masaike
                # block_size = 20  # 马赛克块大小 (8x8像素)
                # width = z.shape[2]  # 图像宽度
                #
                # for i in range(d.shape[0]):
                #     # 计算当前像素的行列位置
                #     row = i // width
                #     col = i % width
                #
                #     # 计算当前像素所属的马赛克块坐标
                #     block_row = row // block_size
                #     block_col = col // block_size
                #
                #     # 棋盘格模式：只处理行号+列号为偶数的块
                #     if (block_row + block_col) % 2 == 0:
                #         d[i] = d[i] * w



                # d = d * w
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        if gt_indices is not None:
            gt_indices = gt_indices.reshape(-1)

            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            gt_min_onehot = torch.zeros(gt_min_indices.shape[0], codebook.shape[0]).to(z)
            gt_min_onehot.scatter_(1, gt_min_indices, 1)

            z_q_gt = torch.matmul(gt_min_onehot, codebook)
            z_q_gt = z_q_gt.view(z.shape)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        e_latent_loss = torch.mean((z_q.detach() - z)**2)
        q_latent_loss = torch.mean((z_q - z.detach())**2)

        if self.LQ_stage and gt_indices is not None:
            # codebook_loss = self.dist(z_q, z_q_gt.detach()).mean() \
                            # + self.beta * self.dist(z_q_gt.detach(), z) 
            codebook_loss = self.beta * self.dist(z_q_gt.detach(), z) 
            texture_loss = self.gram_loss(z, z_q_gt.detach()) 
            # print("codebook loss:", codebook_loss.mean(), "\ntexture_loss: ", texture_loss.mean())
            codebook_loss = codebook_loss + texture_loss 
        else:
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])
    
    def get_codebook_entry(self, indices):
        b, _, h, w = indices.shape

        indices = indices.flatten().to(self.embedding.weight.device)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)        
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q

class SwinLayers(nn.Module):
    def __init__(self, input_resolution=(32, 32), embed_dim=256, 
                blk_depth=6,
                num_heads=8,
                window_size=8,
                **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(4):
            layer = RSTB(embed_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            self.swin_blks.append(layer)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2)
        for m in self.swin_blks:
            x = m(x, (h, w))
        x = x.transpose(1, 2).reshape(b, c, h, w) 
        return x


class MultiScaleEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 LQ_stage=True,
                 **swin_opts,
                 ):
        super().__init__()
        self.LQ_stage = LQ_stage
        ksz = 3

        self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.max_depth = max_depth
        res = input_res
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                ResBlock(out_ch, out_ch, norm_type, act_type),
                ResBlock(out_ch, out_ch, norm_type, act_type),
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
            res = res // 2

        if LQ_stage: 
            self.blocks.append(SwinLayers(**swin_opts))

    def forward(self, input):
        # input.requires_grad = True
        x = self.in_conv(input)
        # x = self.fourier(input)
        # visualize_average_spectrum(x)
        # if self.LQ_stage:
        #     print('input: ', input.requires_grad)
        #     for p in self.in_conv.parameters():
        #         print('conv: ', p.requires_grad)
        #     print('first output:', x.requires_grad)

        for idx, m in enumerate(self.blocks):
            with torch.backends.cudnn.flags(enabled=False):
                x = m(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super().__init__()

        self.block = []
        self.block += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            ResBlock(out_channel, out_channel, norm_type, act_type),
            ResBlock(out_channel, out_channel, norm_type, act_type),
        ]

        self.block = nn.Sequential(*self.block)

    def forward(self, input):
        return self.block(input)

class WarpBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.offset = nn.Conv2d(in_channel * 2, in_channel, 3, stride=1, padding=1)
        self.dcn = DCNv2Pack(in_channel, in_channel, 3, padding=1, deformable_groups=4)

    def forward(self, x_vq, x_residual):
        x_residual = self.offset(torch.cat([x_vq, x_residual], dim=1))
        feat_after_warp = self.dcn(x_vq, x_residual)

        return feat_after_warp


class MultiScaleDecoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 only_residual=False,
                 use_warp=True
                 ):
        super().__init__()
        self.only_residual = only_residual
        self.use_warp = use_warp
        self.upsampler = nn.ModuleList()
        self.warp = nn.ModuleList()
        res =  input_res // (2 ** max_depth)
        for i in range(max_depth):
            in_channel, out_channel = channel_query_dict[res], channel_query_dict[res * 2]
            self.upsampler.append(nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                ResBlock(out_channel, out_channel, norm_type, act_type),
                ResBlock(out_channel, out_channel, norm_type, act_type),
                )
            )
            self.warp.append(WarpBlock(out_channel))
            res = res * 2

    def forward(self, input, code_decoder_output):
        x = input
        for idx, m in enumerate(self.upsampler):
            with torch.backends.cudnn.flags(enabled=False):
                if not self.only_residual:
                    x = m(x)
                    if self.use_warp:
                        x_vq = self.warp[idx](code_decoder_output[idx], x)
                        # print(idx, x.mean(), x_vq.mean())
                        x = x + x_vq * (x.mean() / x_vq.mean())
                    else:
                        x = x + code_decoder_output[idx]
                else:
                    x = m(x)
        # print()
        return x


@ARCH_REGISTRY.register()
class VQWeightDehazeNet(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_params=None,
                 gt_resolution=256,
                 LQ_stage=False,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,
                 use_semantic_loss=False,
                 use_residual=True,
                 only_residual=False,
                 use_weight=False,
                 use_warp=True,
                 weight_alpha=1.0,
                 **ignore_kwargs):
        super().__init__()

        codebook_params = np.array(codebook_params)

        self.codebook_scale = codebook_params[:, 0]
        codebook_emb_num = codebook_params[:, 1].astype(int)
        codebook_emb_dim = codebook_params[:, 2].astype(int)

        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.use_residual = use_residual
        self.only_residual = only_residual
        self.use_weight = use_weight
        self.use_warp = use_warp
        self.weight_alpha = weight_alpha

        self.fusionNet = GlobalNet_Fusion()
        # self.dehaze = Dehaze()

        channel_query_dict = {
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }

        # build encoder 
        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale[0]))
        self.multiscale_encoder = MultiScaleEncoder(
                                in_channel,     
                                self.max_depth,  
                                self.gt_res, 
                                channel_query_dict,
                                norm_type, act_type, LQ_stage
                            )
        if self.LQ_stage and self.use_residual:
            self.multiscale_decoder = MultiScaleDecoder(
                                in_channel,     
                                self.max_depth,  
                                self.gt_res, 
                                channel_query_dict,
                                norm_type, act_type, only_residual, use_warp=self.use_warp
            )


        # build decoder
        self.decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // 2**self.max_depth * 2**i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            self.decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))

        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)
        self.residual_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)

        # build multi-scale vector quantizers 
        self.quantize_group = nn.ModuleList()
        self.before_quant_group = nn.ModuleList()
        self.after_quant_group = nn.ModuleList()

        for scale in range(0, codebook_params.shape[0]):
            quantize = VectorQuantizer(
                codebook_emb_num[scale],
                codebook_emb_dim[scale],
                LQ_stage=self.LQ_stage,
                use_weight=self.use_weight,
                weight_alpha=self.weight_alpha
            )
            self.quantize_group.append(quantize)

            scale_in_ch = channel_query_dict[self.codebook_scale[scale]]
            if scale == 0:
                quant_conv_in_ch = scale_in_ch
                comb_quant_in_ch1 = codebook_emb_dim[scale]
                comb_quant_in_ch2 = 0
            else:
                quant_conv_in_ch = scale_in_ch * 2
                comb_quant_in_ch1 = codebook_emb_dim[scale - 1]
                comb_quant_in_ch2 = codebook_emb_dim[scale]

            self.before_quant_group.append(nn.Conv2d(quant_conv_in_ch, codebook_emb_dim[scale], 1))
            self.after_quant_group.append(CombineQuantBlock(comb_quant_in_ch1, comb_quant_in_ch2, scale_in_ch))

        # semantic loss for HQ pretrain stage
        self.use_semantic_loss = use_semantic_loss
        if use_semantic_loss:
            self.conv_semantic = nn.Sequential(
                nn.Conv2d(512, 512, 1, 1, 0),
                nn.ReLU(),
                )
            self.vgg_feat_layer = 'relu4_4'
            self.vgg_feat_extractor = VGGFeatureExtractor([self.vgg_feat_layer]) 

    def encode_and_decode(self, input, gt_indices=None, current_iter=None, weight_alpha=None):
        # if self.training:
        #     for p in self.multiscale_encoder.parameters():
        #         p.requires_grad = True
        # input = self.fusionNet(input)
        enc_feats = self.multiscale_encoder(input)

        if self.use_semantic_loss:
            with torch.no_grad():
                vgg_feat = self.vgg_feat_extractor(input)[self.vgg_feat_layer]

        codebook_loss_list = []
        indices_list = []
        semantic_loss_list = []
        code_decoder_output = []

        quant_idx = 0
        prev_dec_feat = None
        prev_quant_feat = None
        out_img = None
        out_img_residual = None

        x = enc_feats
        for i in range(self.max_depth):
            cur_res = self.gt_res // 2**self.max_depth * 2**i
            if cur_res in self.codebook_scale:  # needs to perform quantize
                if prev_dec_feat is not None:
                    before_quant_feat = torch.cat((x, prev_dec_feat), dim=1)
                else:
                    before_quant_feat = x
                feat_to_quant = self.before_quant_group[quant_idx](before_quant_feat)

                if weight_alpha is not None:
                    self.weight_alpha = weight_alpha
                if gt_indices is not None:
                    z_quant, codebook_loss, indices = self.quantize_group[quant_idx](feat_to_quant, gt_indices[quant_idx],weight_alpha=self.weight_alpha,input=input)
                else:
                    z_quant, codebook_loss, indices = self.quantize_group[quant_idx](feat_to_quant, weight_alpha=self.weight_alpha,input=input)

                if self.use_semantic_loss:
                    semantic_z_quant = self.conv_semantic(z_quant)
                    semantic_loss = F.mse_loss(semantic_z_quant, vgg_feat)
                    semantic_loss_list.append(semantic_loss)
                
                if not self.use_quantize:
                    z_quant = feat_to_quant

                after_quant_feat = self.after_quant_group[quant_idx](z_quant, prev_quant_feat)

                codebook_loss_list.append(codebook_loss)
                indices_list.append(indices)

                quant_idx += 1
                prev_quant_feat = z_quant
                x = after_quant_feat

            x = self.decoder_group[i](x)
            code_decoder_output.append(x)
            prev_dec_feat = x
        
        out_img = self.out_conv(x)

        if self.LQ_stage and self.use_residual:
            if self.only_residual:
                residual_feature = self.multiscale_decoder(enc_feats, code_decoder_output)
            else:
                residual_feature = self.multiscale_decoder(enc_feats.detach(), code_decoder_output)
            out_img_residual = self.residual_conv(residual_feature)

        if len(codebook_loss_list) > 0:
            codebook_loss = sum(codebook_loss_list) 
        else:
            codebook_loss = 0
        semantic_loss = sum(semantic_loss_list) if len(semantic_loss_list) else codebook_loss * 0

        fblock = False
        if self.LQ_stage and fblock:
            out_img_residual = self.fusionNet(x=out_img_residual,y=out_img_residual).to(input)

        # self.enhancer = False
        # if self.LQ_stage and self.enhancer:
        #     # start enhancer!
        #     tmp = torch.cat((out_img_residual, input), dim=1)
        #     out_img_residual = self.dehaze(tmp)


        return out_img, out_img_residual, codebook_loss, semantic_loss, feat_to_quant, z_quant, indices_list
    
    def decode_indices(self, indices):
        assert len(indices.shape) == 4, f'shape of indices must be (b, 1, h, w), but got {indices.shape}'

        z_quant = self.quantize_group[0].get_codebook_entry(indices)
        x = self.after_quant_group[0](z_quant)

        for m in self.decoder_group:
            x = m(x)
        out_img = self.out_conv(x)
        return out_img

    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16):
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height
        output_width = width
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x
                output_end_x = input_end_x
                output_start_y = input_start_y
                output_end_y = input_end_y

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad)
                output_end_x_tile = output_start_x_tile + input_tile_width
                output_start_y_tile = (input_start_y - input_start_y_pad)
                output_end_y_tile = output_start_y_tile + input_tile_height

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                       output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                  output_start_x_tile:output_end_x_tile]
        return output

    @torch.no_grad()
    def test(self, input, weight_alpha=None):
        org_use_semantic_loss = self.use_semantic_loss
        self.use_semantic_loss = False

        # padding to multiple of window_size * 8
        wsz = 32
        _, _, h_old, w_old = input.shape
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old
        input = torch.cat([input, torch.flip(input, [2])], 2)[:, :, :h_old + h_pad, :]
        input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, :w_old + w_pad]


        output_vq, output, _, _, _, after_quant, index = self.encode_and_decode(input, None, None, weight_alpha=weight_alpha)

        if output is not None:
            output = output[..., :h_old, :w_old]
        if output_vq is not None:
            output_vq = output_vq[..., :h_old, :w_old]

        self.use_semantic_loss = org_use_semantic_loss 
        return output, index

    def forward(self, input, gt_indices=None, weight_alpha=None):

        if gt_indices is not None:
            # in LQ training stage, need to pass GT indices for supervise.
            dec, dec_residual, codebook_loss, semantic_loss, quant_before_feature, quant_after_feature, indices = self.encode_and_decode(input, gt_indices, weight_alpha=weight_alpha)
        else:
            # in HQ stage, or LQ test stage, no GT indices needed.
            dec, dec_residual, codebook_loss, semantic_loss, quant_before_feature, quant_after_feature, indices = self.encode_and_decode(input, weight_alpha=weight_alpha)

        return dec, dec_residual, codebook_loss, semantic_loss, quant_before_feature, quant_after_feature, indices
