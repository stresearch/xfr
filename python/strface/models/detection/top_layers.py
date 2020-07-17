# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.res5a_branch1 = self.__conv(2, name='res5a_branch1', in_channels=1024, out_channels=2048, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.res5a_branch2a = self.__conv(2, name='res5a_branch2a', in_channels=1024, out_channels=512, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.bn5a_branch1 = self.__batch_normalization(2, 'bn5a_branch1', num_features=2048, eps=9.99999974738e-06, momentum=0.0)
        self.bn5a_branch2a = self.__batch_normalization(2, 'bn5a_branch2a', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.res5a_branch2b = self.__conv(2, name='res5a_branch2b', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn5a_branch2b = self.__batch_normalization(2, 'bn5a_branch2b', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.res5a_branch2c = self.__conv(2, name='res5a_branch2c', in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5a_branch2c = self.__batch_normalization(2, 'bn5a_branch2c', num_features=2048, eps=9.99999974738e-06, momentum=0.0)
        self.res5b_branch2a = self.__conv(2, name='res5b_branch2a', in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5b_branch2a = self.__batch_normalization(2, 'bn5b_branch2a', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.res5b_branch2b = self.__conv(2, name='res5b_branch2b', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn5b_branch2b = self.__batch_normalization(2, 'bn5b_branch2b', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.res5b_branch2c = self.__conv(2, name='res5b_branch2c', in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5b_branch2c = self.__batch_normalization(2, 'bn5b_branch2c', num_features=2048, eps=9.99999974738e-06, momentum=0.0)
        self.res5c_branch2a = self.__conv(2, name='res5c_branch2a', in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5c_branch2a = self.__batch_normalization(2, 'bn5c_branch2a', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.res5c_branch2b = self.__conv(2, name='res5c_branch2b', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn5c_branch2b = self.__batch_normalization(2, 'bn5c_branch2b', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.res5c_branch2c = self.__conv(2, name='res5c_branch2c', in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5c_branch2c = self.__batch_normalization(2, 'bn5c_branch2c', num_features=2048, eps=9.99999974738e-06, momentum=0.0)
        self.bbox_pred_1 = self.__dense(name = 'bbox_pred_1', in_features = 2048, out_features = 8, bias = True)
        self.cls_score_1 = self.__dense(name = 'cls_score_1', in_features = 2048, out_features = 2, bias = True)

    def forward(self, x):
        res5a_branch1   = self.res5a_branch1(x)
        res5a_branch2a  = self.res5a_branch2a(x)
        bn5a_branch1    = self.bn5a_branch1(res5a_branch1)
        bn5a_branch2a   = self.bn5a_branch2a(res5a_branch2a)
        res5a_branch2a_relu = F.relu(bn5a_branch2a)

        # Fix MMDNN dilated convolution bug
        #res5a_branch2b_pad = F.pad(res5a_branch2a_relu, (0, 0, 0, 0))
        #res5a_branch2b  = self.res5a_branch2b(res5a_branch2b_pad)
        # Fix broken dilated convolutions on MMDNN conversion, and roll in padding
        res5a_branch2b = F.conv2d(res5a_branch2a_relu, weight=self.res5a_branch2b.weight, bias=self.res5a_branch2b.bias, 
                                  stride=self.res5a_branch2b.stride, padding=(2,2), dilation=2, groups=self.res5a_branch2b.groups)

        bn5a_branch2b   = self.bn5a_branch2b(res5a_branch2b)
        res5a_branch2b_relu = F.relu(bn5a_branch2b)
        res5a_branch2c  = self.res5a_branch2c(res5a_branch2b_relu)
        bn5a_branch2c   = self.bn5a_branch2c(res5a_branch2c)
        res5a           = bn5a_branch1 + bn5a_branch2c
        res5a_relu      = F.relu(res5a)
        res5b_branch2a  = self.res5b_branch2a(res5a_relu)
        bn5b_branch2a   = self.bn5b_branch2a(res5b_branch2a)
        res5b_branch2a_relu = F.relu(bn5b_branch2a)

        # Fix MMDNN dilated convolution bug
        #res5b_branch2b_pad = F.pad(res5b_branch2a_relu, (0, 0, 0, 0))
        #res5b_branch2b  = self.res5b_branch2b(res5b_branch2b_pad)
        res5b_branch2b = F.conv2d(res5b_branch2a_relu, weight=self.res5b_branch2b.weight, bias=self.res5b_branch2b.bias, 
                                  stride=self.res5b_branch2b.stride, padding=(2,2), dilation=2, groups=self.res5b_branch2b.groups)

        bn5b_branch2b   = self.bn5b_branch2b(res5b_branch2b)
        res5b_branch2b_relu = F.relu(bn5b_branch2b)
        res5b_branch2c  = self.res5b_branch2c(res5b_branch2b_relu)
        bn5b_branch2c   = self.bn5b_branch2c(res5b_branch2c)
        res5b           = res5a_relu + bn5b_branch2c
        res5b_relu      = F.relu(res5b)
        res5c_branch2a  = self.res5c_branch2a(res5b_relu)
        bn5c_branch2a   = self.bn5c_branch2a(res5c_branch2a)
        res5c_branch2a_relu = F.relu(bn5c_branch2a)

        # Fix MMDNN dilated convolution bug
        #res5c_branch2b_pad = F.pad(res5c_branch2a_relu, (1, 1, 1, 1))
        #res5c_branch2b  = self.res5c_branch2b(res5c_branch2b_pad)
        res5c_branch2b = F.conv2d(res5c_branch2a_relu, weight=self.res5c_branch2b.weight, bias=self.res5c_branch2b.bias, 
                                  stride=self.res5c_branch2b.stride, padding=(2,2), dilation=2, groups=self.res5c_branch2b.groups)
        bn5c_branch2b   = self.bn5c_branch2b(res5c_branch2b)
        res5c_branch2b_relu = F.relu(bn5c_branch2b)
        res5c_branch2c  = self.res5c_branch2c(res5c_branch2b_relu)
        bn5c_branch2c   = self.bn5c_branch2c(res5c_branch2c)
        res5c           = res5b_relu + bn5c_branch2c
        res5c_relu      = F.relu(res5c)
        pool5           = F.avg_pool2d(res5c_relu, kernel_size=(7, 7), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False)
        bbox_pred_0     = pool5.view(pool5.size(0), -1)
        cls_score_0     = pool5.view(pool5.size(0), -1)
        bbox_pred_1     = self.bbox_pred_1(bbox_pred_0)
        cls_score_1     = self.cls_score_1(cls_score_0)
        # import pdb; pdb.set_trace()
        cls_prob        = F.softmax(cls_score_1, dim=1)
        # Returning pre-softmax score to be consistent with Caffe implementation
        return bbox_pred_1, cls_prob, cls_score_1

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in __weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(__weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(__weights_dict[name]['var']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        print(name)
        print(kwargs)
        if   dim == 1:  
            layer = nn.Conv1d(**kwargs)
        elif dim == 2:  
            layer = nn.Conv2d(**kwargs)
        elif dim == 3:  
            layer = nn.Conv3d(**kwargs)
        else:           
            raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

