#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from util import BinaryConv2d, TernaryConv2d

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.conv1 = nn.Conv2d(1, 32,kernel_size = 5)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64,kernel_size = 5)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10, bias=False)
        
    def forward(self,x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn_conv1(x)), 2)
        x = self.conv2(x)
        x = F.max_pool2d(F.relu(self.bn_conv2(x)), 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x  
    
class LeNet5_TWNs(nn.Module):
    def __init__(self):
        super(LeNet5_TWNs, self).__init__()
        print ("TERNARY VERSION...")
        self.conv1 = TernaryConv2d(1, 32,kernel_size = 5)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = TernaryConv2d(32, 64,kernel_size = 5)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.fc1 = TernaryConv2d(1024, 512,kernel_size = 1)
        self.bn_fc1 = nn.BatchNorm2d(512)
        self.fc2 = nn.Conv2d(512, 10,kernel_size = 1, bias=False)
        #self.bn_fc2 = nn.BatchNorm2d(10)
        
    def forward(self,x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn_conv1(x)), 2)
        x = self.conv2(x)
        x = F.max_pool2d(F.relu(self.bn_conv2(x)), 2)
        
        x = x.view(-1, 1024, 1, 1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        
        x = self.fc2(x)
        x = x.view(-1, 10)
    
        return x  
  
    
class LeNet5_BPWNs(nn.Module):
    def __init__(self):
        super(LeNet5_BPWNs, self).__init__()
        print ("BINARY VERSION...")
        self.conv1 = BinaryConv2d(1, 32,kernel_size = 5)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = BinaryConv2d(32, 64,kernel_size = 5)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.fc1 = BinaryConv2d(1024, 512,kernel_size = 1)
        self.bn_fc1 = nn.BatchNorm2d(512)
        self.fc2 = nn.Conv2d(512, 10,kernel_size = 1, bias=False)
        #self.bn_fc2 = nn.BatchNorm2d(10)
        
    def forward(self,x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn_conv1(x)), 2)
        x = self.conv2(x)
        x = F.max_pool2d(F.relu(self.bn_conv2(x)), 2)
        
        x = x.view(-1, 1024, 1, 1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        
        x = self.fc2(x)
        x = x.view(-1, 10)
    
        return x  
 
 
class VGG7_BPWNs(nn.Module):
    def __init__(self, class_num=10):
        super(VGG7_BPWNs, self).__init__()
        self.class_num = class_num
        self.conv1_1 = BinaryConv2d(3, 128, kernel_size = 3, padding=1)
        self.bn_conv1_1 = nn.BatchNorm2d(128)
        self.conv1_2 = BinaryConv2d(128, 128, kernel_size = 3, padding=1)
        self.bn_conv1_2 = nn.BatchNorm2d(128)
        
        self.conv2_1 = BinaryConv2d(128, 256, kernel_size = 3, padding=1)
        self.bn_conv2_1 = nn.BatchNorm2d(256)
        self.conv2_2 = BinaryConv2d(256, 256, kernel_size = 3, padding=1)
        self.bn_conv2_2 = nn.BatchNorm2d(256)
        
        self.conv3_1 = BinaryConv2d(256, 512, kernel_size = 3, padding=1)
        self.bn_conv3_1 = nn.BatchNorm2d(512)
        self.conv3_2 = BinaryConv2d(512, 512, kernel_size = 3, padding=1)
        self.bn_conv3_2 = nn.BatchNorm2d(512)
        
        self.fc1 = BinaryConv2d(512*4*4, 1024, kernel_size = 1)
        self.bn_fc1 = nn.BatchNorm2d(1024)
        
        self.fc2 = nn.Conv2d(1024, self.class_num, kernel_size = 1, bias=False)
        
    def forward(self,x):
        x = F.relu(self.bn_conv1_1(self.conv1_1(x)))
        x = F.relu(self.bn_conv1_2(self.conv1_2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn_conv2_1(self.conv2_1(x)))
        x = F.relu(self.bn_conv2_2(self.conv2_2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn_conv3_1(self.conv3_1(x)))
        x = F.relu(self.bn_conv3_2(self.conv3_2(x)))
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, 512*4*4, 1, 1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        
        x = self.fc2(x)
        x = x.view(-1, self.class_num)
    
        return x  


class VGG7_TWNs(nn.Module):
    def __init__(self, class_num=10):
        super(VGG7_TWNs, self).__init__()
        self.class_num = class_num
        self.conv1_1 = TernaryConv2d(3, 128, kernel_size = 3, padding=1)
        self.bn_conv1_1 = nn.BatchNorm2d(128)
        self.conv1_2 = TernaryConv2d(128, 128, kernel_size = 3, padding=1)
        self.bn_conv1_2 = nn.BatchNorm2d(128)
        
        self.conv2_1 = TernaryConv2d(128, 256, kernel_size = 3, padding=1)
        self.bn_conv2_1 = nn.BatchNorm2d(256)
        self.conv2_2 = TernaryConv2d(256, 256, kernel_size = 3, padding=1)
        self.bn_conv2_2 = nn.BatchNorm2d(256)
        
        self.conv3_1 = TernaryConv2d(256, 512, kernel_size = 3, padding=1)
        self.bn_conv3_1 = nn.BatchNorm2d(512)
        self.conv3_2 = TernaryConv2d(512, 512, kernel_size = 3, padding=1)
        self.bn_conv3_2 = nn.BatchNorm2d(512)
        
        self.fc1 = TernaryConv2d(512*4*4, 1024, kernel_size = 1)
        self.bn_fc1 = nn.BatchNorm2d(1024)
        
        self.fc2 = nn.Conv2d(1024, self.class_num, kernel_size = 1, bias=False)
        
    def forward(self,x):
        x = F.relu(self.bn_conv1_1(self.conv1_1(x)))
        x = F.relu(self.bn_conv1_2(self.conv1_2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn_conv2_1(self.conv2_1(x)))
        x = F.relu(self.bn_conv2_2(self.conv2_2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn_conv3_1(self.conv3_1(x)))
        x = F.relu(self.bn_conv3_2(self.conv3_2(x)))
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, 512*4*4, 1, 1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        
        x = self.fc2(x)
        x = x.view(-1, self.class_num)
    
        return x  


class VGG7(nn.Module):
    def __init__(self, class_num=10):
        super(VGG7, self).__init__()
        self.class_num = class_num
        self.conv1_1 = nn.Conv2d(3, 128, kernel_size = 3, padding=1)
        self.bn_conv1_1 = nn.BatchNorm2d(128)
        self.conv1_2 = nn.Conv2d(128, 128, kernel_size = 3, padding=1)
        self.bn_conv1_2 = nn.BatchNorm2d(128)
        
        self.conv2_1 = nn.Conv2d(128, 256, kernel_size = 3, padding=1)
        self.bn_conv2_1 = nn.BatchNorm2d(256)
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size = 3, padding=1)
        self.bn_conv2_2 = nn.BatchNorm2d(256)
        
        self.conv3_1 = nn.Conv2d(256, 512, kernel_size = 3, padding=1)
        self.bn_conv3_1 = nn.BatchNorm2d(512)
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size = 3, padding=1)
        self.bn_conv3_2 = nn.BatchNorm2d(512)
        
        self.fc1 = nn.Conv2d(512*4*4, 1024, kernel_size = 1)
        self.bn_fc1 = nn.BatchNorm2d(1024)
        
        self.fc2 = nn.Conv2d(1024, self.class_num, kernel_size = 1, bias=False)
        
    def forward(self,x):
        x = F.relu(self.bn_conv1_1(self.conv1_1(x)))
        x = F.relu(self.bn_conv1_2(self.conv1_2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn_conv2_1(self.conv2_1(x)))
        x = F.relu(self.bn_conv2_2(self.conv2_2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn_conv3_1(self.conv3_1(x)))
        x = F.relu(self.bn_conv3_2(self.conv3_2(x)))
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, 512*4*4, 1, 1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        
        x = self.fc2(x)
        x = x.view(-1, self.class_num)
    
        return x  


        
