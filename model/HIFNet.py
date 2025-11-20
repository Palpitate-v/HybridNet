
import torch
from torchvision import models as resnet_model
from torch import nn
from model.transformer import TransformerModel
from model.swin_transformer import SwinTransformerBlock
from model.UnetV2 import *


class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(in_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class HIFNet(nn.Module):
    def __init__(self, n_channels=3, num_classes=2, heads=8, dim=64, depth=(3, 3, 3), patch_size=2):
        super(HIFNet, self).__init__()
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.heads = heads
        self.depth = depth
        self.dim = dim
        # mlp_dim = [2 * dim, 4 * dim, 8 * dim, 16 * dim] #原transformer的
        embed_dim = [dim, 2 * dim, 4 * dim]  # ^^^^
        resnet = resnet_model.resnet34(weights=resnet_model.ResNet34_Weights.DEFAULT)  # pretrained = True

        self.vit_1 = SwinTransformerBlock(dim=embed_dim[0], num_heads=heads, H=56, W=56)  # 224 / 4 分成四个块
        self.vit_2 = SwinTransformerBlock(dim=embed_dim[1], num_heads=heads, H=28, W=28)
        self.vit_3 = SwinTransformerBlock(dim=embed_dim[2], num_heads=heads, H=14, W=14)

        self.patch_embed_1 = nn.Conv2d(n_channels, embed_dim[0], kernel_size=2 * patch_size, stride=2 * patch_size)
        self.patch_embed_2 = nn.Conv2d(embed_dim[0], embed_dim[1], kernel_size=patch_size, stride=patch_size)
        self.patch_embed_3 = nn.Conv2d(embed_dim[1], embed_dim[2], kernel_size=patch_size, stride=patch_size)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1  # 64
        self.encoder2 = resnet.layer2  # 128
        self.encoder3 = resnet.layer3  # 256
        self.encoder4 = resnet.layer4

        self.ca_1 = ChannelAttention(128 + dim)
        self.sa_1 = SpatialAttention()

        self.ca_2 = ChannelAttention(256 + 2 * dim)
        self.sa_2 = SpatialAttention()

        self.ca_3 = ChannelAttention(512 + 4 * dim)
        self.sa_3 = SpatialAttention()

        # BasicConv2d进行 conv + BN 操作
        self.Translayer_1 = BasicConv2d(128 + dim, dim, 1)
        self.Translayer_2 = BasicConv2d(256 + 2 * dim, dim, 1)
        self.Translayer_3 = BasicConv2d(512 + 4 * dim, dim, 1)

        self.dropoutVit = nn.Dropout(0.5)
        self.dropoutLast = nn.Dropout(0.5)

        # self.sdi_1 = SDI(dim)  # 最底下
        self.sdi_2 = SDI(dim)
        self.sdi_3 = SDI(dim)
        self.sdi_4 = SDI(dim)

        self.decoder1 = DecoderBottleneckLayer(dim)
        self.decoder2 = DecoderBottleneckLayer(dim+dim//2)
        self.decoder3 = DecoderBottleneckLayer((dim+dim//2)//2+dim)

        self.up4_1 = nn.ConvTranspose2d(dim, dim//2, kernel_size=2, stride=2)  # 最底下的一层，通道數最多
        self.up3_1 = nn.ConvTranspose2d(dim+dim//2, (dim+dim//2)//2, kernel_size=2, stride=2)
        self.up2_1 = nn.ConvTranspose2d((dim+dim//2)//2+dim, dim, kernel_size=4, stride=4)

        self.out = nn.Conv2d(dim, num_classes, kernel_size=1)


    def forward(self, x):
        b, c, h, w = x.shape
        patch_size = self.patch_size
        dim = self.dim


        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)  # H/2 * W/2  torch.Size([1, 64, 112, 112])
        # e0 = self.maxpool(e0)  # torch.Size([1, 64, 56, 56])

        e1 = self.encoder1(e0)  # H/4  *  W/4      torch.Size([1, 64, 56, 56])
        e2 = self.encoder2(e1)  # H/8  *  W/8      torch.Size([1, 128, 28, 28])
        e3 = self.encoder3(e2)  # H/16  *  W/16    torch.Size([1, 256, 14, 14])
        e4 = self.encoder4(e3)  # H/32  *  W/32    torch.Size([1, 512, 7, 7])


        v1 = self.patch_embed_1(x)
        v1 = v1.permute(0, 2, 3, 1).contiguous()
        v1 = v1.view(b, -1, dim)
        v1 = self.vit_1(v1)
        v1=self.dropoutVit(v1)
        v1_cnn = v1.view(b, int(h / (2 * patch_size)), int(w / (2 * patch_size)), dim)
        v1_cnn = v1_cnn.permute(0, 3, 1, 2).contiguous()  # ↑  torch.Size([1, 64, 56, 56])

        v2 = self.patch_embed_2(v1_cnn)
        v2 = v2.permute(0, 2, 3, 1).contiguous()
        v2 = v2.view(b, -1, 2 * dim)
        v2 = self.vit_2(v2)
        v2 = self.dropoutVit(v2)
        v2_cnn = v2.view(b, int(h / (patch_size * 2 * 2)), int(w / (2 * 2 * patch_size)), dim * 2)
        v2_cnn = v2_cnn.permute(0, 3, 1, 2).contiguous() #torch.Size([1, 128, 28, 28])

        v3 = self.patch_embed_3(v2_cnn)
        v3 = v3.permute(0, 2, 3, 1).contiguous()
        v3 = v3.view(b, -1, 4 * dim)
        v3 = self.vit_3(v3)
        v3 = self.dropoutVit(v3)
        v3_cnn = v3.view(b, int(h / (patch_size * 2 * 2 * 2)), int(w / (2 * 2 * 2 * patch_size)), dim * 2 * 2)
        v3_cnn = v3_cnn.permute(0, 3, 1, 2).contiguous() #torch.Size([1, 256, 14, 14])

        cat_3 = torch.cat([v3_cnn, e4], dim=1)
        cat_3 = self.ca_3(cat_3) * cat_3
        cat_3 = self.sa_3(cat_3) * cat_3
        cat_3 = self.Translayer_3(cat_3) #torch.Size([1, 64, 14, 14])

        cat_2 = torch.cat([v2_cnn, e3], dim=1)
        cat_2 = self.ca_2(cat_2) * cat_2
        cat_2 = self.sa_2(cat_2) * cat_2
        cat_2 = self.Translayer_2(cat_2) #torch.Size([1, 64, 28, 28])

        cat_1 = torch.cat([v1_cnn, e2], dim=1)
        cat_1 = self.ca_1(cat_1) * cat_1
        cat_1 = self.sa_1(cat_1) * cat_1
        cat_1 = self.Translayer_1(cat_1) #torch.Size([1, 64, 56, 56])

        f41 = self.sdi_4([cat_1, cat_2, cat_3], cat_1)  # 残差连接部分，从上面数的  第一层
        f31 = self.sdi_3([cat_1, cat_2, cat_3], cat_2)
        f21 = self.sdi_2([cat_1, cat_2, cat_3], cat_3)  # 第3层


        cat_2 = self.decoder1(f21) #torch.Size([1, 96, 14, 14])
        cat_2 = self.up4_1(cat_2) #torch.Size([1, 48, 28, 28])

        cat_3 = torch.cat([f31, cat_2], dim=1)#torch.Size([1, 112, 28, 28])
        cat_3 = self.decoder2(cat_3)
        cat_3 = self.up3_1(cat_3) #torch.Size([1, 56, 56, 56])

        cat_4 = torch.cat([f41, cat_3], dim=1) #torch.Size([1, 120, 56, 56])
        cat_4 = self.decoder3(cat_4)
        cat_4 = self.up2_1(cat_4) #torch.Size([1, 64, 224, 224])


        cat_4 = self.dropoutLast(cat_4)

        out = self.out(cat_4)
        return out
