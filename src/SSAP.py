import torch
import torch.nn as nn
from torchvision import models


class DoubleConv(nn.Module):
    """DoubleConv

     Conv→BN→ReLuを２回繰り返す.

    """
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    """Down

     U-NetにおけるDown処理.解像度を縦横半分にする.

    """
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    """Up

     U-NetにおけるUp処理.解像度を縦横2倍にする.

    """
    def __init__(self, in_ch, out_ch):
        """__init__

         in_ch　>　out_ch として、次元数を減らし画像の解像度を大きくする

        Args:
            in_ch (int): 入力データのchannel数.
                x1(forwardの引数)と同じchannel数の必要がある.
            out_ch (int): 出力データのchannel数.
                x2(forwardの引数)と同じchannel数の必要がある.

        """
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        """forward

        Args:
            x1(Tensor): 解像度を小さくした特徴量.
                行列の形式は(n_batch, in_ch, height, width).
            x2(Tensor): 浅い層で保存しておいた解像度の大きい特徴量.
                行列の形式は(n_batch, out_ch, height, width).

         Returns:
            Tensor: 解像度が低い特徴量と保存しておいた特徴量を組み合わせた特徴量.
                行列の形式は(n_batch, out_ch, height, width).

        """
        x1 = self.up(x1)
        x = torch.cat((x2, x1), dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """OutConv

     Conv→Sigmoidを通し、最終結果を出力.

    """
    def __init__(self, in_ch, out_ch, activation):
        """__init__

        Args:
            in_ch (int): 入力データのchannel数.
            out_ch (int): 最終出力のchannel数.
                Segmentationの出力のchannel数はclass数.
                Affinityの出力のchannel数はAffinityのWindow size＊＊２.
            activation (str): 活性化関数の指定.

        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class SSAP(nn.Module):
    """SSAP

     `SSAP <https://arxiv.org/abs/1909.01616>`_ は、
     U-Netを基準として、各層のSegmentationとAffinityを出力する.
     出力結果を階層的に利用することで、Instance Segmentationを作成する.
     SSAPのEncoder部分に
     `Resnet34 <https://arxiv.org/abs/1512.03385>`_ を使用している.

    """

    def __init__(self, n_channels, n_classes, aff_r):
        """__init__

        Args:
            n_channels (int): 入力画像のchannel数.
            n_classes (int): Segmentationのクラス数.
            aff_r (int): AffinityのWindow size.

        """
        super(SSAP, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.inc = DoubleConv(n_channels, 64)
        self.down = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.out_c0 = OutConv(1024, n_classes, "softmax")
        self.out_aff0 = OutConv(1024, aff_r**2, "sigmoid")
        self.out_c1 = OutConv(512, n_classes, "softmax")
        self.out_aff1 = OutConv(512, aff_r**2, "sigmoid")
        self.out_c2 = OutConv(256, n_classes, "softmax")
        self.out_aff2 = OutConv(256, aff_r**2, "sigmoid")
        self.out_c3 = OutConv(128, n_classes, "softmax")
        self.out_aff3 = OutConv(128, aff_r**2, "sigmoid")
        self.out_c4 = OutConv(64, n_classes, "softmax")
        self.out_aff4 = OutConv(64, aff_r**2, "sigmoid")

    def forward(self, x):
        """forward

        Args:
            x (Tensor): 入力画像をBatchにまとめたもの.
                行列の形式は(n_batch, ch, height, width).
                通常channel=3.

        Returns:
            out_c0~out_c4 (Tensor): 各層のSegmentation.
                行列の形式は(n_batch, class数, height, width).
            out_aff0~out_aff4 (Tensor): 各層のAffinity.
                行列の形式は(n_batch, AffinityのWindow size**2, height, width).

        """
        x = self.inc(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        x5 = self.down(x4)
        out_c0 = self.out_c0(x5)
        out_aff0 = self.out_aff0(x5)
        x = self.up1(x5, x4)
        out_c1 = self.out_c1(x)
        out_aff1 = self.out_aff1(x)
        x = self.up2(x, x3)
        out_c2 = self.out_c2(x)
        out_aff2 = self.out_aff2(x)
        x = self.up3(x, x2)
        out_c3 = self.out_c3(x)
        out_aff3 = self.out_aff3(x)
        x = self.up4(x, x1)
        out_c4 = self.out_c4(x)
        out_aff4 = self.out_aff4(x)
        return out_c0, out_c1, out_c2, out_c3, out_c4, \
            out_aff0, out_aff1, out_aff2, out_aff3, out_aff4
