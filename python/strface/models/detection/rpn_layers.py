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

        self.rpn_conv_3x3 = self.__conv(2, name='rpn_conv/3x3', in_channels=1024, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.rpn_cls_score = self.__conv(2, name='rpn_cls_score', in_channels=512, out_channels=18, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.rpn_bbox_pred = self.__conv(2, name='rpn_bbox_pred', in_channels=512, out_channels=36, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)

    def forward(self, x):
        rpn_conv_3x3_pad = F.pad(x, (1, 1, 1, 1))
        rpn_conv_3x3    = self.rpn_conv_3x3(rpn_conv_3x3_pad)
        rpn_relu_3x3    = F.relu(rpn_conv_3x3)
        rpn_cls_score   = self.rpn_cls_score(rpn_relu_3x3)
        rpn_bbox_pred   = self.rpn_bbox_pred(rpn_relu_3x3)
        return rpn_cls_score, rpn_bbox_pred


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

