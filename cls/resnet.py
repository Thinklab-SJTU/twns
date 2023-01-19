import torch
import torch.nn as nn
from util import BinaryConv2d, TernaryConv2d

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, conv_op, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            conv_op(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv_op(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                conv_op(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, conv_op, num_block, wscale=1.0, num_classes=1000):
        super().__init__()
        self.conv_op = conv_op
        
        self.in_channels = int(64*wscale)
        
        self.conv1 = nn.Sequential(
            self.conv_op(3, self.in_channels, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True))
        
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2_x = self._make_layer(block, self.in_channels, num_block[0], 2)
        self.conv3_x = self._make_layer(block, self.in_channels*2, num_block[1], 2)
        self.conv4_x = self._make_layer(block, self.in_channels*4, num_block[2], 2)
        self.conv5_x = self._make_layer(block, self.in_channels*8, num_block[3], 2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(self.in_channels*8*block.expansion, num_classes, kernel_size=1, bias=False)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.conv_op, self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mp(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x

def ResNet18_TWNs():
    return ResNet(BasicBlock, TernaryConv2d, [2, 2, 2, 2])

def ResNet18_BPWNs():
    return ResNet(BasicBlock, BinaryConv2d, [2, 2, 2, 2])

def ResNet18_FPWNs():
    return ResNet(BasicBlock, nn.Conv2d, [2, 2, 2, 2])

def ResNet18B_TWNs():
    return ResNet(BasicBlock, TernaryConv2d, [2, 2, 2, 2], wscale=1.5)

def ResNet18B_BPWNs():
    return ResNet(BasicBlock, BinaryConv2d, [2, 2, 2, 2], wscale=1.5)

def ResNet18B_FPWNs():
    return ResNet(BasicBlock, nn.Conv2d, [2, 2, 2, 2], wscale=1.5)
