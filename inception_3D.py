from __future__ import division

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F



__all__ = ['Inception3', 'inception_3']

# Script annotations failed with _GoogleNetOutputs = namedtuple ...

def inception_3(pretrained=False, progress=True, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the 3D inception_v3 expects tensors with a size of
        N x 3 x 43 x 149 x 149, so ensure your images are sized accordingly.
        Batch Size = N
        channel = 3

    Args:
        # pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """

    model = Inception3(**kwargs)
    
    return model


class Inception3(nn.Module):

    def __init__(self, num_classes=10,
                 inception_blocks=None):
        super(Inception3, self).__init__()
        
        conv_block = BasicConv3d
        
        inception_a = InceptionA
        inception_b = InceptionB
        inception_c = InceptionC
        inception_d = InceptionD
        inception_e = InceptionE
        #inception_aux = InceptionAux

        # self.Conv3d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=1, padding=(0, 1, 1))

        self.Conv3d_2a_3x3 = conv_block(3, 16, kernel_size=3)
        self.Conv3d_2b_3x3 = conv_block(16, 32, kernel_size=3, padding=1)

        self.Conv3d_3a_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.Conv3d_3b_1x1 = conv_block(64, 128, kernel_size=1)
        self.Conv3d_4a_3x3 = conv_block(128, 192, kernel_size=3)

        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(384, channels_7x7=64)
        self.Mixed_6c = inception_c(384, channels_7x7=80)
        self.Mixed_6d = inception_c(384, channels_7x7=80)
        self.Mixed_6e = inception_c(384, channels_7x7=96)
        self.Mixed_7a = inception_d(384)
        self.Mixed_7b = inception_e(640)
        self.Mixed_7c = inception_e(1024)
        self.fc = nn.Linear(1024, num_classes)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def init_weight_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # N x 3 x 299 x 299 for 2D
        # N x 3 x 43 x 149 x 149 
        # x = self.Conv3d_1a_3x3(x)
        # N x 3 x 43 x 81 x 81
        x = self.Conv3d_2a_3x3(x)
        # N x 16 x 41 x 79 x 79
        x = self.Conv3d_2b_3x3(x)
        # N x 32 x 41 x 79 x 79
        x = F.max_pool3d(x, kernel_size=3, stride=(1, 2, 2))

        # N x 32 x 39 x 39 x 39
        x = self.Conv3d_3a_3x3(x)
        # N x 64 x 39 x 39 x 39  scale=1/2
        x = self.Conv3d_3b_1x1(x)
        # N x 128 x 39 x 39 x 39
        x = self.Conv3d_4a_3x3(x)
        # N x 192 x 37 x 37 x 37
        x = F.max_pool3d(x, kernel_size=3, stride=2, padding=1)

        # N x 192 x 19 x 19 x 19 下同
        x = self.Mixed_5b(x)
        # N x 256 x 19 x 19 x 19
        x = self.Mixed_5c(x)
        # N x 288 x 19 x 19 x 19
        x = self.Mixed_5d(x)
        # N x 288 x 19 x 19 x 19
        x = self.Mixed_6a(x)
        # N x 384 x 9 x 9 x 9
        x = self.Mixed_6b(x)
        # N x 384 x 9 x 9 x 9
        x = self.Mixed_6c(x)
        # N x 384 x 9 x 9 x 9
        x = self.Mixed_6d(x)
        # N x 384 x 9 x 9 x 9
        x = self.Mixed_6e(x)
        # N x 384 x 9 x 9 x 9
        x = self.Mixed_7a(x)
        # N x 640 x 4 x 4 x 4
        x = self.Mixed_7b(x)
        # N x 1024 x 4 x 4 x 4
        x = self.Mixed_7c(x)
        # N x 1024 x 4 x 4 x 4
        # Adaptive average pooling
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        # N x 1024 x 1 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 1024 x 1 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.fc(x)
        # N x 1000 (num_classes)

        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv3d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv3d
        self.branch3x3 = conv_block(in_channels, 192, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 32, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(32, 48, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(48, 48, kernel_size=3, stride=2)

        self.branch_conv = conv_block(in_channels, 144, kernel_size=3, padding=1)

    def _forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_conv(x)
        branch_pool = F.max_pool3d(branch_pool, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7, conv_block=None):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv3d
        self.branch1x1 = conv_block(in_channels, 96, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 1, 7), padding=(0, 0, 3))
        self.branch7x7_3 = conv_block(c7, c7, kernel_size=(1, 7, 1), padding=(0, 3, 0))
        self.branch7x7_4 = conv_block(c7, 96, kernel_size=(7, 1, 1), padding=(3, 0, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(1, 1, 7), padding=(0, 0, 3))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7, 1), padding=(0, 3, 0))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1, 1), padding=(3, 0, 0))
        self.branch7x7dbl_5 = conv_block(c7, c7, kernel_size=(1, 1, 7), padding=(0, 0, 3))
        self.branch7x7dbl_6 = conv_block(c7, c7, kernel_size=(1, 7, 1), padding=(0, 3, 0))
        self.branch7x7dbl_7 = conv_block(c7, 96, kernel_size=(7, 1, 1), padding=(3, 0, 0))

        self.branch_pool = conv_block(in_channels, 96, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7 = self.branch7x7_4(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_6(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_7(branch7x7dbl)

        branch_pool = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv3d
        self.branch3x3_1 = conv_block(in_channels, 96, kernel_size=1)
        self.branch3x3_2 = conv_block(96, 160, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 96, kernel_size=1)
        self.branch7x7x3_2 = conv_block(96, 96, kernel_size=(1, 1, 7), padding=(0, 0, 3))
        self.branch7x7x3_3 = conv_block(96, 96, kernel_size=(1, 7, 1), padding=(0, 3, 0))
        self.branch7x7x3_4 = conv_block(96, 96, kernel_size=(7, 1, 1), padding=(3, 0, 0))
        self.branch7x7x3_5 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_5(branch7x7x3)

        branch_pool = F.max_pool3d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv3d
        self.branch1x1 = conv_block(in_channels, 160, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 128, kernel_size=1)
        self.branch3x3_2a = conv_block(128, 128, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.branch3x3_2b = conv_block(128, 128, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.branch3x3_2c = conv_block(128, 128, kernel_size=(3, 1, 1), padding=(1, 0, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 224, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(224, 128, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(128, 128, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.branch3x3dbl_3b = conv_block(128, 128, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.branch3x3dbl_3c = conv_block(128, 128, kernel_size=(3, 1, 1), padding=(1, 0, 0))

        self.branch_pool = conv_block(in_channels, 96, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
            self.branch3x3_2c(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
            self.branch3x3dbl_3c(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv3d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool3d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


if __name__=='__main__':
    model = inception_3()
    model = model.cuda()
    # print(model)

    in_ = torch.randn(10, 3, 43, 81, 81)
    in_ = in_.cuda()
    out_ = model(in_)
    print(out_.shape)
