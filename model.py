import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np

class Block(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1):
        super(Block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=stride,padding="same")
        self.in1 = nn.InstanceNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=1,padding="same")
        self.in2 = nn.InstanceNorm2d(out_channel)
        self.dropout = nn.Dropout(0.5)
    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += identity
        out = self.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self,block,blocks_num):
        super(ResNet, self).__init__()
        self.in_channel = 2
        self.conv1 = nn.Conv2d(self.in_channel,out_channels=64,kernel_size=3,padding="same")
        self.in1 = nn.InstanceNorm2d(self.in_channel)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.linear1 = nn.Linear(64,1)
        self.in2 = nn.InstanceNorm2d(self.in_channel)
        self.sigmoid = nn.Sigmoid()
        #self.layer2 = self._make_layer(block, 64, blocks_num[1])
        #self.layer3 = self._make_layer(block, 64, blocks_num[2])
        #self.layer4 = self._make_layer(block, 64, blocks_num[3])
    def _make_layer(self,block,channel,block_num):
        layers = []
        for _ in range(block_num):
            layers.append(block(channel,channel))
        return nn.Sequential(*layers)
    def forward(self,x):
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = x.transpose(1,3)
        x = self.in2(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = (x+x.transpose(1,2))/2
        return x.transpose(1,3)
