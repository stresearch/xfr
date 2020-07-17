'''
    implement Light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04

    MIT License: 
    https://github.com/AlfredXiangWu/LightCNN/blob/master/LICENSE
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
from skimage.color import rgb2gray

def prepare_lightCNN_image(img):
    """
    Convert a grayscale byte image to a FloatTensor suitable for processing with the network
    This function assumes the image has already been resized, cropped, jittered, etc.
    """
    img_gray = rgb2gray(np.array(img))  # RGB uint8 -> Luminance float32 [0,1]
    return torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0)   # 1x1x128x128

def lightcnn_preprocess():
    transform_list = [transforms.Resize(144),]
    transform_list.append(transforms.CenterCrop((128,128)))
    transform_list.append(transforms.Lambda(prepare_lightCNN_image))
    return transforms.Compose(transform_list)

class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()
    def forward(self, x, y):
        return x+y

class Split(nn.Module):
    def __init__(self, split_size, dim):
        super(Split, self).__init__()
        self.split_size = split_size
        self.dim = dim
    def forward(self, x):
        return torch.split(x, self.split_size, self.dim)


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)
        self.split = Split(out_channels, 1)
            
    def forward(self, x):
        x = self.filter(x)
        #out = torch.split(x, self.out_channels, 1)
        out = self.split(x)
        return torch.max(out[0], out[1])

class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.add = Add()

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        #out = out + res
        out = self.add(out, res)
        return out

class network_9layers(nn.Module):
    def __init__(self, num_classes=79077):
        super(network_9layers, self).__init__()
        self.features = nn.Sequential(
            mfm(1, 48, 5, 1, 2), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(48, 96, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        self.fc1 = mfm(8*8*128, 256, type=0)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out, x


class network_9layers_Custom(network_9layers):

    def forward(self, x, nrm=True):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        out = self.fc2(x)

        if nrm is False:
            return x

        xnorm = F.normalize(x, p=2, dim=1)

        return xnorm

class network_29layers(nn.Module):
    def __init__(self, block, layers, num_classes=79077):
        super(network_29layers, self).__init__()
        self.conv1  = mfm(1, 48, 5, 1, 2)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc     = mfm(8*8*128, 256, type=0)
        self.fc2    = nn.Linear(256, num_classes)
            
    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.group1(x)
        x = self.pool2(x)

        x = self.block2(x)
        x = self.group2(x)
        x = self.pool3(x)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        fc = F.dropout(fc, training=self.training)
        out = self.fc2(fc)
        return out, fc


class network_29layers_Custom(network_29layers):

    def forward(self, x, nrm=True):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.group1(x)
        x = self.pool2(x)

        x = self.block2(x)
        x = self.group2(x)
        x = self.pool3(x)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        fc = F.dropout(fc, training=self.training)

        if nrm is False:
            return fc

        xnorm = F.normalize(fc, p=2, dim=1)

        #out = self.fc2(fc)
        #return out, fc
        return xnorm

class network_29layers_v2(nn.Module):
    def __init__(self, block, layers, num_classes=79077):
        super(network_29layers_v2, self).__init__()
        self.conv1    = mfm(1, 48, 5, 1, 2)
        self.block1   = self._make_layer(block, layers[0], 48, 48)
        self.group1   = group(48, 96, 3, 1, 1)
        self.block2   = self._make_layer(block, layers[1], 96, 96)
        self.group2   = group(96, 192, 3, 1, 1)
        self.block3   = self._make_layer(block, layers[2], 192, 192)
        self.group3   = group(192, 128, 3, 1, 1)
        self.block4   = self._make_layer(block, layers[3], 128, 128)
        self.group4   = group(128, 128, 3, 1, 1)
        self.fc       = nn.Linear(8*8*128, 256)
        self.fc2 = nn.Linear(256, num_classes, bias=False)

        # Expose modules for whitebox EBP
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.maxpool4 = nn.MaxPool2d(2)

        self.avgpool1 = nn.AvgPool2d(2)
        self.avgpool2 = nn.AvgPool2d(2)
        self.avgpool3 = nn.AvgPool2d(2)
        self.avgpool4 = nn.AvgPool2d(2)


    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        #x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        x = self.maxpool1(x) + self.avgpool1(x)

        x = self.block1(x)
        x = self.group1(x)
        #x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        x = self.maxpool2(x) + self.avgpool2(x)

        x = self.block2(x)
        x = self.group2(x)
        #x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        x = self.maxpool3(x) + self.avgpool3(x)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        #x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        x = self.maxpool4(x) + self.avgpool4(x)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        x = F.dropout(fc, training=self.training)
        out = self.fc2(x)
        return out, fc

def LightCNN_9Layers(weights_path=None):
    model = network_9layers_Custom()

    if weights_path is not None:
        state_dict = Load_Checkpoint(weights_path)
        model.load_state_dict(state_dict)

    return model

def LightCNN_29Layers(weights_path=None):
    model = network_29layers_Custom(resblock, [1, 2, 3, 4])

    if weights_path is not None:
        state_dict = Load_Checkpoint(weights_path)
        model.load_state_dict(state_dict)

    return model

def LightCNN_29Layers_v2(**kwargs):
    model = network_29layers_v2(resblock, [1, 2, 3, 4], **kwargs)
    model.training = False
    return model

def Load_Checkpoint(weights_path):
    checkpoint = torch.load(weights_path, map_location='cpu')
    restore_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    return restore_dict

