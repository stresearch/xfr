# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.



import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import ImageFilter


def prepare_vggface_image(img):
    """
    Convert an RGB byte image to a FloatTensor suitable for processing with the network.
    This function assumes the image has already been resized, cropped, jittered, etc.
    """
    # Convert to BGR
    img_bgr = np.array(img)[...,[2,1,0]]
    # Subtract mean pixel value
    img_bgr_fp = img_bgr - np.array((93.5940, 104.7624, 129.1863))
    # Permute dimensions so output is 3xRxC
    img_bgr_fp = np.rollaxis(img_bgr_fp, 2, 0)
    return torch.from_numpy(img_bgr_fp).float()


def generate_random_blur(blur_radius, blur_prob):
    def random_blur(img):
        if np.random.random() < blur_prob:
            return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        else:
            return img
    return random_blur


""" Function suitable for transform argument of datasets.ImageFolder """
def vggface_preprocess(jitter=False, blur_radius=None, blur_prob=1.0):
    transform_list = [transforms.Resize(256),]
    if jitter:
        transform_list.append(transforms.RandomCrop((224,224)))
        transform_list.append(transforms.RandomHorizontalFlip())
        #transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    else:
        transform_list.append(transforms.CenterCrop((224,224)))
    if blur_radius is not None and blur_prob > 0:
        transform_list.append(transforms.Lambda(generate_random_blur(blur_radius, blur_prob)))
    # finally, convert PIL RGB image to FloatTensor
    transform_list.append(transforms.Lambda(prepare_vggface_image))
    return transforms.Compose(transform_list)


class VGGFace(nn.Module):
    """
    The VGGFace network (VGG_VD_16)
    mode can be one of ['encode', 'classify', 'both']
    """
    def __init__(self, mode='encode', num_classes=2622):
        super(VGGFace, self).__init__()
        valid_modes = {'encode','classify','both'}
        if mode not in valid_modes:
            raise Exception('mode should be one of ' + str(valid_modes))
        self.mode = mode
        self.fc_outputs = num_classes
        # layers with stored weights
        self.conv1_1 = nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1))
        self.conv1_2 = nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1))

        self.conv2_1 = nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1))
        self.conv2_2 = nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1))

        self.conv3_1 = nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1))
        self.conv3_2 = nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1))
        self.conv3_3 = nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1))

        self.conv4_1 = nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1))
        self.conv4_2 = nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1))
        self.conv4_3 = nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1))

        self.conv5_1 = nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1))
        self.conv5_2 = nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1))
        self.conv5_3 = nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1))

        self.fc6 = nn.Linear(25088,4096)
        self.fc7 = nn.Linear(4096,4096)
        self.fc8 = nn.Linear(4096, self.fc_outputs)

        # layers with no weights
        self.nonlin = nn.ReLU()
        self.maxpool = nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True)
        self.dropout = nn.Dropout(0.5)


    def forward(self, input):
        """
        Run the network.
        Input should be Nx3x224x224.
        Based on self.mode, return output of fc7, fc8, or both.
        """
        assert len(input.size()) == 4

        e1_1 = self.nonlin(self.conv1_1(input))
        e1_2 = self.maxpool(self.nonlin(self.conv1_2(e1_1)))

        e2_1 = self.nonlin(self.conv2_1(e1_2))
        e2_2 = self.maxpool(self.nonlin(self.conv2_2(e2_1)))

        e3_1 = self.nonlin(self.conv3_1(e2_2))
        e3_2 = self.nonlin(self.conv3_2(e3_1))
        e3_3 = self.maxpool(self.nonlin(self.conv3_3(e3_2)))

        e4_1 = self.nonlin(self.conv4_1(e3_3))
        e4_2 = self.nonlin(self.conv4_2(e4_1))
        e4_3 = self.maxpool(self.nonlin(self.conv4_3(e4_2)))

        e5_1 = self.nonlin(self.conv5_1(e4_3))
        e5_2 = self.nonlin(self.conv5_2(e5_1))
        e5_3 = self.maxpool(self.nonlin(self.conv5_3(e5_2)))

        e5_3_flat = e5_3.view(e5_3.size(0), -1)

        e6 = self.nonlin(self.fc6(e5_3_flat))
        # use encoding prior to nonlinearity
        e7_pre = self.fc7(self.dropout(e6))
        e7 = self.nonlin(e7_pre)

        # return e7, e8, or both depending on self.mode
        if self.mode == 'encode':
            return e7
        else:
            e8 = self.fc8(self.dropout(e7))
            if self.mode == 'classify':
                return e8
            elif self.mode == 'both':
                return e7,e8
            else:
                raise Exception('Invalid mode: ' + mode)

    def set_fc_outputs(self, new_fc_outputs):
        self.fc_outputs = new_fc_outputs
        self.fc8 = nn.Linear(4096, self.fc_outputs)


class VGGFace_Custom(VGGFace):
    """Inherit VGGFace() and override the forward pass to
    normalize the output. Don't care about classification
    """

    def forward(self, input, nrm=True):
        """
        Run the network.
        Input should be Nx3x224x224.
        Based on self.mode, return output of fc7, fc8, or both.
        """
        assert len(input.size()) == 4

        e1_1 = self.nonlin(self.conv1_1(input))
        e1_2 = self.maxpool(self.nonlin(self.conv1_2(e1_1)))

        e2_1 = self.nonlin(self.conv2_1(e1_2))
        e2_2 = self.maxpool(self.nonlin(self.conv2_2(e2_1)))

        e3_1 = self.nonlin(self.conv3_1(e2_2))
        e3_2 = self.nonlin(self.conv3_2(e3_1))
        e3_3 = self.maxpool(self.nonlin(self.conv3_3(e3_2)))

        e4_1 = self.nonlin(self.conv4_1(e3_3))
        e4_2 = self.nonlin(self.conv4_2(e4_1))
        e4_3 = self.maxpool(self.nonlin(self.conv4_3(e4_2)))

        e5_1 = self.nonlin(self.conv5_1(e4_3))
        e5_2 = self.nonlin(self.conv5_2(e5_1))
        e5_3 = self.maxpool(self.nonlin(self.conv5_3(e5_2)))

        e5_3_flat = e5_3.view(e5_3.size(0), -1)

        e6 = self.nonlin(self.fc6(e5_3_flat))
        # use encoding prior to nonlinearity
        e7_pre = self.fc7(self.dropout(e6))
        e7 = self.nonlin(e7_pre)

        """Override code here: Want to normalize the output and
        return the encoding. Don't care about classification.
        """

        if nrm is False:
            return e7

        #print torch.div(e7,torch.norm(e7))
        #print e7.size()
        xnorm = F.normalize(e7, p=2, dim=1)
        return xnorm

        #return torch.div(e7,torch.norm(e7))

        
def vgg16(model_filename=None):
    """
    Constructs a VGG-16 model
    """
    model = VGGFace_Custom()
    if model_filename is not None:
        model.load_state_dict(torch.load(model_filename))
    return model
