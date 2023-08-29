import ptflops
import torch
from torch import nn
import torchvision
import math
import torch.nn.functional as F
from torchsummary import summary

from dcn import DSTA, DeformFuser
from gatedfusion import GatedFusion
# from language_correction import BCNLanguage
from mobilevit import MobileViTBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class ConvolutionalBlock(nn.Module):
    """
    卷积模块,由卷积层, BN归一化层, 激活层构成.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        """
        :参数 in_channels: 输入通道数
        :参数 out_channels: 输出通道数
        :参数 kernel_size: 核大小
        :参数 stride: 步长
        :参数 batch_norm: 是否包含BN层
        :参数 activation: 激活层类型; 如果没有则为None
        """

        #super()：解决多重继承时父类查找的问题，一般在子类中需要调用父类的方法时
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # 层列表
        layers = list()

        # 1个卷积层
        layers.append(
            nn.Conv2d(in_channels=in_channels,#输入通道数
                      out_channels=out_channels,#输出通道数
                      kernel_size=kernel_size, #卷积核大小
                      stride=stride,#步长，默认1
                      padding=kernel_size // 2)#输入数据最边缘补0的个数，默认为0
        )

        # 1个BN归一化层
        if batch_norm is True:
            layers.append(
                nn.BatchNorm2d(num_features=out_channels))#特征的数量

        # 1个激活层
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # 合并层
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        前向传播

        :参数 input: 输入图像集，张量表示，大小为 (N, in_channels, w, h)
        :返回: 输出图像集，张量表示，大小为(N, out_channels, w, h)
        """
        output = self.conv_block(input)

        return output


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
            m.append(nn.PReLU())
        else:
            # 报错，缩放因子不对
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class GruBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x

class MobileViTResidualBlock(nn.Module):
    """
    残差模块, 包含一个卷积模块、一个MobileViT块和一个跳连.
    """
    def __init__(self, kernel_size=3, n_channels=64, dim=64, depth=2, patch_size=(2,2), mlp_dim=192 ):
        """
        :参数 kernel_size: 核大小
        :参数 n_channels: 输入和输出通道数（由于是ResNet网络，需要做跳连，因此输入和输出通道数是一致的）
        """
        super(MobileViTResidualBlock, self).__init__()
        # 第一个卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')
        # 第二个卷积块
        # self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
        #                                       batch_norm=True, activation=None)
        self.mvit_block = MobileViTBlock(dim=dim, depth=depth, channel=n_channels, patch_size=patch_size, mlp_dim=mlp_dim)

    def forward(self, input):
        """
        前向传播.
        :参数 input: 输入图像集，张量表示，大小为 (N, n_channels, w, h)
        :返回: 输出图像集，张量表示，大小为 (N, n_channels, w, h)
        """
        residual = input  # (N, n_channels, w, h)
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.mvit_block(output)  # (N, n_channels, w, h)

        output = output + residual  # (N, n_channels, w, h)
        return output

class InfoGen(nn.Module):
    def __init__(
                self,
                t_emb,
                output_size
                 ):
        super(InfoGen, self).__init__()

        self.tconv1 = nn.ConvTranspose2d(t_emb, 512, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 128, 3, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 3, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, output_size, 3, (2, 1), padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(output_size)

    def forward(self, t_embedding):

        # t_embedding += noise.to(t_embedding.device)
        # print("t_embedding.shape:",t_embedding.shape)
        x = F.relu(self.bn1(self.tconv1(t_embedding)))
        # print(x.shape)
        x = F.relu(self.bn2(self.tconv2(x)))
        # print(x.shape)
        x = F.relu(self.bn3(self.tconv3(x)))
        # print(x.shape)
        x = F.relu(self.bn4(self.tconv4(x)))
        # print(x.shape)

        return x, torch.zeros((x.shape[0], 1024, t_embedding.shape[-1])).to(x.device)


class MyNet(nn.Module):
    """
    我的模型
    """
    def __init__(self, in_channels=3,
                 kernel_size=3,
                 n_channels=32,
                 n_blocks=6,
                 scale=2,
                 hidden_units=32,
                 text_emb=7935,
                 out_text_channels=32,
                 triple_clues=True
                 ):
        """
        :参数 in_channels: 输入图像通道数。若加了二值图，就为4，否则为3
        :参数 kernel_size: 中间层卷积核大小
        :参数 n_channels: 中间层通道数
        :参数 n_blocks: 残差模块数
        :参数 scale: 放大比例
        """
        super(MyNet, self).__init__()

        self.triple_clues = triple_clues
        self.infoGen = InfoGen(text_emb, out_text_channels)
        # self.lm = BCNLanguage()
        # self.lm.load_state_dict(torch.load('ckpt/BCN_correct_model.pt'))
        self.dsta_rec = DSTA(hidden_units)
        self.dsta_vis = DSTA(hidden_units)
        self.dsta_ling = DSTA(hidden_units)
        # self.vis_rec_fuser = DeformFuser(16,hidden_units,hidden_units,4)
        # self.gated = gated(hidden_units)
        self.gated = GatedFusion(hidden_units)
        self.down_conv = nn.Conv2d(hidden_units, hidden_units, 1, padding=0)
        self.infoGen_ling = InfoGen(text_emb, out_text_channels)
        self.infoGen_visual = InfoGen(text_emb, 10)
        # self.correction_model = BCNLanguage()
        self.vis_rec_fuser = DeformFuser(16, hidden_units, hidden_units, 4)

        # 第一个卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=in_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=False, activation='PReLu')

        # 一系列残差模块, 每个残差模块包含一个跳连接
        self.residual_blocks = nn.Sequential(
            *[MobileViTResidualBlock(kernel_size=3, n_channels=32, dim=96, depth=2, patch_size=(2,2), mlp_dim=192) for i in range(n_blocks)])

        # 第二个卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=kernel_size,
                                              batch_norm=True, activation=None)
        # 放大通过子像素卷积模块实现, 每个模块放大两倍
        self.subpixel_convolutional_blocks = Upsample(scale=scale, num_feat=n_channels)

        # 最后一个卷积模块
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=kernel_size,
                                              batch_norm=False, activation='Tanh')



    def forward(self, lr_imgs,tp_net=None):
        """
        前向传播.

        :参数 lr_imgs: 低分辨率输入图像集, 张量表示，大小为 (N, 3, w, h)
        :返回: 高分辨率输出图像集, 张量表示， 大小为 (N, 3, w * scaling factor, h * scaling factor)
        """
        output = self.conv_block1(lr_imgs)  # (16, 3, 24, 24)
        residual = output  # (16, 64, 24, 24)
        # print(output.shape)
        output = self.residual_blocks(output)  #
        output = self.conv_block2(output)
        output = output + residual  #
        # print("output.shape:",output.shape)
        output = self.subpixel_convolutional_blocks(output)  # (16, 64, 24 * 4, 24 * 4)
        sr_imgs = self.conv_block3(output)  # (16, 3, 24 * 4, 24 * 4)

        preds_sr = tp_net(sr_imgs)

        label_vecs = torch.nn.functional.softmax(preds_sr, -1)
        label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
        text_emb = label_vecs_final
        print(text_emb.shape)
        exit(0)
        spatial_t_emb_, pr_weights = self.infoGen(text_emb)  # # ,block['1'], block['1'],
        spatial_t_emb = F.interpolate(spatial_t_emb_, (sr_imgs.shape[2], sr_imgs.shape[3]), mode='bilinear',
                                      align_corners=True)
        if self.triple_clues:
            hint_rec = self.dsta_rec(spatial_t_emb)
            hint = self.gated(self.down_conv(self.conv_block1(sr_imgs)), hint_rec)

        sr_imgs = self.conv_block3(hint)  # (16, 3, 24 * 4, 24 * 4)

        return sr_imgs

class TruncatedVGG19(nn.Module):
    """
    truncated VGG19网络，用于计算VGG特征空间的MSE损失
    """

    def __init__(self, i, j):
        """
        :参数 i: 第 i 个池化层
        :参数 j: 第 j 个卷积层
        """
        super(TruncatedVGG19, self).__init__()

        # 加载预训练的VGG模型
        vgg19 = torchvision.models.vgg19(pretrained=True)  # C:\Users\Administrator/.cache\torch\checkpoints\vgg19-dcbb9e9d.pth

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        # 迭代搜索
        for layer in vgg19.features.children():
            truncate_at += 1

            # 统计
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # 截断位置在第(i-1)个池化层之后（第 i 个池化层之前）的第 j 个卷积层
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # 检查是否满足条件
        assert maxpool_counter == i - 1 and conv_counter == j, "当前 i=%d 、 j=%d 不满足 VGG19 模型结构" % (
            i, j)

        # 截取网络
        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

    def forward(self, input):
        """
        前向传播
        参数 input: 高清原始图或超分重建图，张量表示，大小为 (N, 3, w * scaling factor, h * scaling factor)
        返回: VGG19特征图，张量表示，大小为 (N, feature_map_channels, feature_map_w, feature_map_h)
        """
        output = self.truncated_vgg19(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)

        return output

def parse_crnn_data(imgs_input):
    # print(imgs_input.shape)

    in_width = 128
    imgs_input = torch.nn.functional.interpolate(imgs_input[:, :3, ...], (32, in_width), mode='bicubic')#插值和上采样

    # imgs_input = torch.nn.functional.interpolate(imgs_input, (32, in_width), mode='bicubic')
    R = imgs_input[:, 0:1, :, :]
    G = imgs_input[:, 1:2, :, :]
    B = imgs_input[:, 2:3, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B
    return tensor

if __name__ == '__main__':
    model = MyNet(scale=2).cuda()
    summary(model, (3,24, 96))