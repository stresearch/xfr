# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.



import sys
import strface.recognition
import PIL
import numpy as np
import torch


def encode_centercrop():
    im = np.array(PIL.Image.open('ak.png'))
    im = (im - strface.recognition.MEAN_RGB).astype(np.float32) # RGB
    x = torch.from_numpy(im[np.newaxis, :].transpose(0,3,1,2))
    net = strface.recognition.resnet101v6().eval()
    X_torch = net(x).detach().numpy()
    print(X_torch)

    
def encode_centertwocrop_multiscale():
    imlist = []
    for s in [224,256,281]:
        im = np.array(PIL.Image.open('ak.png').resize( (s,s), PIL.Image.BILINEAR))   # resize minimum dimension, this assumes input image is square!
        (dx,dy) = (int(round((im.shape[0]-224)/2.0)), int(round((im.shape[1]-224)/2.0)))  # center crop offset
        im = (im - strface.recognition.MEAN_RGB).astype(np.float32) # RGB image from PIL, remove mean 
        im = im[dx:dx+224, dy:dy+224, :]  # 224x224 center crop
        imlist.append(im)  # scale
        imlist.append(np.fliplr(im))  # mirrored

    imlist = [im[np.newaxis, :].transpose(0,3,1,2) for im in imlist]  # WxHxC -> 1xCxWxH tensor
    x = torch.from_numpy(np.concatenate(imlist, axis=0))  # list of N 1xCxWxH tensors -> NxCxWxH tensor
    net = strface.recognition.resnet101v6().eval()
    X_torch = torch.mean(net(x), dim=0).detach().numpy()  # mean over multiscale crops
    print(X_torch)
