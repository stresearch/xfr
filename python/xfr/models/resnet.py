# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.



import os.path
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import PIL
from PIL import Image, ImageFilter
import numpy as np
import pdb
import uuid
from heapq import nlargest
import skimage.transform
import warnings


MEAN_RGB = np.array(([122.782, 117.001, 104.298]))

def convert_resnet101v4_image(img, mean_rgb=MEAN_RGB):
    """
    Convert an RGB byte image to a FloatTensor suitable for processing with the network.
    This function assumes the image has already been resized, cropped, jittered, etc.
    """
    # Subtract mean pixel value
    if isinstance(img, np.ndarray):
        img_fp = img - MEAN_RGB
    else:
        img_fp = img.convert('RGB') - MEAN_RGB
    # Permute dimensions so output is 3xHxW
    img_fp = np.moveaxis(img_fp, 2, 0)
    return torch.from_numpy(img_fp).float()


def pil_to_tensor(img):
    return convert_resnet101v4_image(img)

def convert_from_numpy(img):
    assert img.max() <= 1.00001
    assert img.min() >= -0.00001
    img = skimage.transform.resize(img, (224, 224), preserve_range=True)
    img = (img * 255).astype(np.uint8)
    img = PIL.Image.fromarray(img).convert('RGB')
    img = convert_resnet101v4_image(img).unsqueeze(0)
    return img


def unconvert_resnet101v4_image(img_in, mean_rgb=MEAN_RGB):
    """
    Convert a FloatTensor to an RGB byte Image
    """
    if img_in.dim() == 4:
        assert img_in.shape[0] == 1, "Expecting a single image as input"
        img_in = img_in.squeeze()
    img_fp = np.moveaxis(img_in.numpy(), 0, 2)
    img_fp += MEAN_RGB
    img_fp[img_fp < 0] = 0
    img_fp[img_fp > 255] = 255
    return Image.fromarray(img_fp.astype(np.uint8))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()
    def forward(self, x, y):
        return x+y


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               bias=True)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.add = Add()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # Need a function with forward for hooks.  Replace:
        #    out += residual
        #    out = self.relu(out)
        return self.relu(self.add(out, residual))


class ConcatChannels(nn.Module):
    def __init__(self, channels):
        super(ConcatChannels, self).__init__()
        self.channels = int(channels)
    def forward(self, x):
        return torch.cat((x, Variable(torch.zeros(x.size()).type_as(x.data).repeat(1,self.channels,1,1))), dim=1)


class Multiply(nn.Module): # ugh
    def __init__(self, n):
        super(Multiply, self).__init__()
        self.n = n
    def forward(self, x):
        return x*self.n


class ResNet(nn.Module):

    def __init__(self, block, layers, mode, num_classes):
        super(ResNet, self).__init__()
        valid_modes = {'encode','classify','both'}
        if mode not in valid_modes:
            raise Exception('mode should be one of ' + str(valid_modes))
        self.mode = mode
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=7)
        self.fc1 = nn.Linear(512 * block.expansion, 512)
        self.multiply = Multiply(50.0)
        self.fc2 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None: m.bias.data.normal_(0, math.sqrt(2. / n)) # REVIEW
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, use_conv=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if use_conv:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion, track_running_stats=True),
                )
            else:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride), # VSI HACK: tying kernel_size and stride
                    ConcatChannels(planes*block.expansion//self.inplanes - 1)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, mode=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        # FIXME: every operation needs a .forward() method
        if isinstance(x,dict):
            x['An'] = x['An'].view(x['An'].size(0), -1)
            x['Xn'] = x['Xn'].view(x['Xn'].size(0), -1)
        else:
            x = x.view(x.size(0), -1)
        x = self.fc1(x)

        # FIXME:  every operation needs a .forward() method
        if isinstance(x, dict):
            xnorm = {'An':F.normalize(x['An'], p=2, dim=1),
                     'Xn':F.normalize(x['Xn'], p=2, dim=1)}
        else:
            xnorm = F.normalize(x, p=2, dim=1)

        xnorm = self.multiply(xnorm)

        mode = self.mode if None else mode
        if mode == 'encode':
            return xnorm
        else:
            scores = self.fc2(xnorm)
            if mode == 'classify':
                return scores
            elif mode == 'both':
                # else return the encoding and classification scores
                return xnorm, scores
            else:
                raise Exception('Invalid mode: ' + mode)


def resnet101v6(pthfile, device=None):
    if device is None:
        if torch.cuda.is_available():
            warnings.warn('device not defined in resnet101v6, assigning to '
                          'first CUDA visible device.')
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    model = ResNet(Bottleneck, [3, 4, 23, 3], mode='encode', num_classes=65359)
    model.load_state_dict(torch.load(pthfile, map_location=device))
    return model

def stresnet101(pthfile, device=None):
    return resnet101v6(pthfile, device)


