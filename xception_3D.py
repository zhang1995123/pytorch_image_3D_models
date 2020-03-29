import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv3d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv3d, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.pointwise = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=bias)
        self.bn2 = nn.BatchNorm3d(planes)

    def forward(self, x):
        # print('s_in', x.shape)
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        # print('s_out', x.shape)

        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, 
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv3d(inplanes, planes, 1, stride=stride, bias=False)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv3d(inplanes, planes, 3, 1, dilation))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv3d(filters, filters, 3, 1, dilation))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv3d(inplanes, planes, 3, 1, dilation))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv3d(planes, planes, 3, 2))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv3d(planes, planes, 3, 1))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
        else:
            skip = inp

        x = x + skip

        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Xception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, num_classes=10):
        super(Xception, self).__init__()

        middle_block_dilation = 1
        exit_block_dilations = (1, 2)
    
        # exit_block_dilations = (2, 4)

        self.relu = nn.ReLU(inplace=True)
        # Entry flow
        self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=1, bias=False)

        self.conv2 = ConvBlock(32, 64, kernel_size=4, stride=1, bias=False)

        self.conv3 = ConvBlock(64, 64, kernel_size=3, stride=1, bias=False)

        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=False, grow_first=True)
        self.block3 = Block(256, 384, reps=2, stride=1,
                            start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        self.block4  = Block(384, 384, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block5  = Block(384, 384, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block6  = Block(384, 384, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block7  = Block(384, 384, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block8  = Block(384, 384, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block9  = Block(384, 384, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block10 = Block(384, 384, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block11 = Block(384, 384, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block12 = Block(384, 384, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block13 = Block(384, 384, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block14 = Block(384, 384, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block15 = Block(384, 384, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block16 = Block(384, 384, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block17 = Block(384, 384, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block18 = Block(384, 384, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block19 = Block(384, 384, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(384, 512, reps=2, stride=1, dilation=exit_block_dilations[0],
                             start_with_relu=True, grow_first=False, is_last=True)

        self.conv4 = SeparableConv3d(512, 768, 3, stride=1, dilation=exit_block_dilations[1])

        self.conv5 = SeparableConv3d(768, 768, 3, stride=1, dilation=exit_block_dilations[1])

        self.conv6 = SeparableConv3d(768, 1024, 3, stride=1, dilation=exit_block_dilations[1])

        self.fc = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)

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
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        ## N x 3 x 512 x 512  --  3 x 43 x 81 x 81
        # Entry flow
        x = self.conv1(x)
        ## N x 32 x 41 x 79 x 79
        x = F.max_pool3d(x, kernel_size=3, stride=(1, 2, 2))
        ## N x 32 x 39 x 39 x 39
        x = self.conv2(x)
        ## N x 64 x 36 x 36 x 36
        x = F.avg_pool3d(x, kernel_size=3, stride=1)
        ## N x 64 x 34 x 34 x 34
        x = self.conv3(x)
        # print('c3', x.shape)
        ## N x 64 256 x 256  --  64 x 32 x 32 x 32
        x = self.block1(x)
        # print('b1', x.shape)
        # can add relu here (low-level-feature)
        
        ## N x 128 x 128 x 128  --  128 x 32 x 32 x 32
        x = self.block2(x)
        ## N x 256 x 64 x 64  --  256 x 16 x 16 x 16
        x = self.block3(x)

        # Middle flow
        ## N x 728 x 32 x 32  --  384 x 8 x 8 x 8
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        ## N x 728 x 32 x 32

        # Exit flow
        x = self.block20(x)
        ## N x 1024 x 32 x 32  -- 512 x 8 x 8 x 8
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        ## N x 1536 x 32 x 32  --  768 x 8 x 8 x 8
        x = self.conv5(x)
        x = self.relu(x)
        ## N x 1536 x 32 x 32  --  786 x 8 x 8 x 8
        x = self.conv6(x)
        x = self.relu(x)       
        ## N x 2048 x 32 x 32  --  1024 x 8 x 8 x 8
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = self.dropout(x)
        ## N x 1024 x 1 x 1 x 1
        x = torch.flatten(x, 1)
        ## N x 1024
        x = self.fc(x)
        ## N x 1000 (num_classes)

        return x



if __name__ == "__main__":
    model = Xception().cuda()
    in_ = torch.rand(9, 3, 43, 81, 81).cuda()
    out_ = model(in_)
    print(out_.size())
