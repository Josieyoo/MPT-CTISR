import math

import cv2
import torch
import torch.nn as nn

from einops import rearrange
from torchsummary import summary



def conv_1x1_bn(inp, oup,kernel_size=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # 深度卷积
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # 逐点卷积
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        # self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        #up conv3*3 to dwcconv3*3
        self.conv1 = MV2Block(inp=channel, oup=channel, expansion=4)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = Transformer(dim=dim, depth=depth, heads=4, dim_head=8, mlp_dim=mlp_dim, dropout=dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        #up conv_nxn_bn(2 * channel, channel, kernel_size) to conv_1x1_bn(2 * channel, channel)
        # self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
        self.conv4 = conv_1x1_bn((dim+channel), channel)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        #up add z = x.clone()
        z = x.clone()

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)

        # Fusion
        x = self.conv3(x)
        #up change torch.cat((x, y), 1) to torch.cat((x, z), 1)
        # x = torch.cat((x, y), 1)
        # print(x.shape)
        # print(z.shape)
        x = torch.cat((x, z), 1)

        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        # print(y.shape)
        x = x+y
        return x


class Upsample(nn.Sequential):
    """
    输入:
        scale (int): 缩放因子，支持: 2^n 和 3.
        num_feat (int): 中间特征的通道数
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):  # 循环n次
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))  # pixelshuffle 上采样 2 倍
                m.append(nn.PReLU())
        elif scale == 3:  # scale = 3
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))  # pixelshuffle 上采样 3 倍
        else:
            # 报错，缩放因子不对
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class MobileViT(nn.Module):
    def __init__(self, image_size, kernel_size=3, patch_size=(2, 2), scale = 2):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        self.scale = scale

        self.conv1 = nn.Conv2d(4, 16, 3, 1, 1)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(inp=16, oup=32, expansion=1))
        self.mv2.append(MV2Block(inp=32, oup=48, expansion=1))
        self.mv2.append(MV2Block(inp=48, oup=48, expansion=1))

        self.mv2.append(MV2Block(inp=48, oup=48, expansion=1))
        self.mv2.append(MV2Block(inp=48, oup=64, expansion=1))
        self.mv2.append(MV2Block(inp=64, oup=80, expansion=1))


        # dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout = 0.
        self.mvit = nn.ModuleList([])
        self.mvit.append(
            MobileViTBlock(dim=48, depth=2, channel=48, kernel_size=kernel_size, patch_size=patch_size, mlp_dim=96))
        self.mvit.append(
            MobileViTBlock(dim=64, depth=4, channel=64, kernel_size=kernel_size, patch_size=patch_size, mlp_dim=128))
        self.mvit.append(
            MobileViTBlock(dim=80, depth=3, channel=80, kernel_size=kernel_size, patch_size=patch_size, mlp_dim=160))

        self.conv2 = nn.Conv2d(80, 64, 3, 1, 1)

        self.conv_before_upsample = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),
                                                  nn.LeakyReLU(inplace=True))#inplace=True，对上层传下来的原地操作，不创建新变量，节省内存

        self.upsample = Upsample(scale, 32)
        self.conv_last = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x, y):
        x = self.conv1(x)

        x = self.mv2[0](x)
        x = self.mv2[1](x)
        x = self.mv2[2](x)

        x = self.mv2[3](x)
        x = self.mvit[0](x)

        x = self.mv2[4](x)
        x = self.mvit[1](x)


        x = self.mv2[5](x)
        x = self.mvit[2](x)

        x = self.conv2(x)
        x = self.conv_before_upsample(x)
        x = self.upsample(x)  # 上采样并降维后输出

        x = self.conv_last(x)


        return x


if __name__ == '__main__':
    model = MobileViT(image_size=(24, 48),scale=4).cuda()
    summary(model, [(3,24, 48),(3,24, 48)])
#
#     img = torch.randn(5, 3, 256, 256)
#     #
#     # vit = mobilevit_xxs()
#     # out = vit(img)
#     # print(out.shape)
#     # print(count_parameters(vit))
#     #
#     # vit = mobilevit_xs()
#     # out = vit(img)
#     # print(out.shape)
#     # print(count_parameters(vit))
#     #
#     # vit = mobilevit_s()
#     # out = vit(img)
#     # print(out.shape)
#     # print(count_parameters(vit))