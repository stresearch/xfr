# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import skimage
import skimage.filters
import pandas as pd
import pdb
import copy

from xfr.models.resnet import Bottleneck
from xfr.models.resnet import convert_resnet101v4_image
from xfr.models.lightcnn import lightcnn_preprocess
import xfr.models.lightcnn
from xfr import utils


class WhiteboxNetwork(object):
    def __init__(self, net):
        """
        A WhiteboxNetwork() is the class wrapper for a torch network to be used with the Whitebox() class
        The input is a torch network which satisfies the assumptions outlined in the README
        """
        self.net = net
        self.net.eval()

    def _layer_visitor(self, f_forward=None, f_preforward=None, net=None):
        """Recursively assign forward hooks and pre-forward hooks for all layers within containers"""
        """Returns list in order of forward visit with {'name'n:, 'hook':h} dictionary"""
        """Signature of hook functions must match register_forward_hook and register_pre_forward_hook in torch docs"""
        layerlist = []
        net = self.net if net is None else net
        for name, layer in net._modules.items():
            hooks = []
            if isinstance(layer, nn.Sequential) or isinstance(layer, Bottleneck):
                layerlist.append( {'name':str(layer), 'hooks':[None]})
                layerlist = layerlist + self._layer_visitor(f_forward, f_preforward, layer)
            else:
                if f_forward is not None:
                    hooks.append(layer.register_forward_hook(f_forward))
                if f_preforward is not None:
                    hooks.append(layer.register_forward_pre_hook(f_preforward))
                layerlist.append( {'name':str(layer), 'hooks':hooks} )
            if isinstance(layer, nn.BatchNorm2d):
                #layer.eval()
                #layer.track_running_stats = False  (do not disable, this screws things up)
                #layer.affine = False
                pass
        return layerlist

    def encode(self, x):
        """Given an Nx3xUxV input tensor x, return a D dimensional vector encoding of shape NxD, one per image"""
        raise

    def classify(self, x):
        """Given an Nx3xUxV input tensor x, and a network with C classes, return NxC pre-softmax classification output for the network"""
        raise

    def clear(self):
        """Clear gradients, multiple calls to backwards should not accumulate"""
        """This function should not need to be overloaded"""
        for p in self.net.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def set_triplet_classifier(self, x_mate, x_nonmate):
        """Given two D-dimensional encodings x_mate and x_nonmate, construct a 2xD classifier layer which will be output for classify() with C=2."""
        """Replace the output of the encoding layer with a 2xD fully connected layer which will compute the inner product of the two encodings and the probe"""
        raise

    def num_classes(self):
        """Return the number of classes for the current network"""
        raise

    def preprocess(self, im):
        """Given a PIL image im, preprocess this image to return a tensor that can be suitably input to the network for forward()"""
        raise


class WhiteboxSTResnet(WhiteboxNetwork):
    def __init__(self, net):
        """A subclass of WhiteboxNetwork() which implements the whitebox API for a resnet-101 topology"""
        self.net = net
        self.net.eval()

    def set_triplet_classifier(self, x_mate, x_nonmate):
        """Replace classifier with binary mate and non-mate classifier, must be set prior to first call to forward for callbacks to be set properly"""
        self.net.fc2 = nn.Linear(512, 2, bias=False)
        self.net.fc2.weight = nn.Parameter(torch.cat( (x_mate, x_nonmate), dim=0))

    def encode(self, x):
        """Return 512d normalized forward encoding for input tensor x"""
        return self.net.forward(x, mode='encode')

    def classify(self, x):
        return self.net.forward(x, mode='classify')

    def num_classes(self):
        return self.net.fc2.out_features

    def preprocess(self, im):
        """PIL image to input tensor"""
        return convert_resnet101v4_image(im.resize( (224,224) )).unsqueeze(0)


class WhiteboxLightCNN(WhiteboxNetwork):
    def __init__(self, net):
        """A subclass of WhiteboxNetwork() which implements the whitebox API for a Light CNN Topology"""
        """https://github.com/AlfredXiangWu/LightCNN"""
        self.net = net
        self.net.eval()
        self.f_preprocess = lightcnn_preprocess()

    def set_triplet_classifier(self, x_mate, x_nonmate):
        """Replace classifier with binary mate and non-mate classifier, must be set prior to first call to forward for callbacks to be set properly"""
        self.net.fc2 = nn.Linear(256, 2, bias=False)
        self.net.fc2.weight = nn.Parameter(torch.cat( (x_mate, x_nonmate), dim=0))

    def encode(self, x):
        """Return 512d normalized forward encoding for input tensor x"""
        p, features = self.net(x)
        return features

    def classify(self, x):
        p, features = self.net(x)
        return p

    def num_classes(self):
        return self.net.fc2.out_features

    def preprocess(self, im):
        """PIL image to input tensor"""
        return self.f_preprocess(im)

    def _layer_visitor(self, f_forward=None, f_preforward=None, net=None):
        """Recursively assign forward hooks and pre-forward hooks for all layers within containers"""
        """Returns list in order of forward visit with {'name'n:, 'hook':h} dictionary"""
        """Signature of hook functions must match register_forward_hook and register_pre_forward_hook in torch docs"""
        layerlist = []
        net = self.net if net is None else net
        for name, layer in net._modules.items():
            hooks = []
            if isinstance(layer, nn.Sequential) or isinstance(layer, Bottleneck) or isinstance(layer, xfr.models.lightcnn.mfm) or isinstance(layer, xfr.models.lightcnn.group) or isinstance(layer, xfr.models.lightcnn.resblock):
                layerlist.append( {'name':str(layer), 'hooks':[None]})
                layerlist = layerlist + self._layer_visitor(f_forward, f_preforward, layer)
            else:
                if f_forward is not None:
                    hooks.append(layer.register_forward_hook(f_forward))
                if f_preforward is not None:
                    hooks.append(layer.register_forward_pre_hook(f_preforward))
                layerlist.append( {'name':str(layer), 'hooks':hooks} )
        return layerlist



class Whitebox_senet50_256(WhiteboxNetwork):

    def __init__(self, net):
        """https://github.com/ox-vgg/vgg_face2"""
        self.net = net
        self.net.eval()
        self.fc1 = nn.Linear(256, 2, bias=False)

    def set_triplet_classifier(self, x_mate, x_nonmate):
        """Replace classifier with binary mate and non-mate classifier, must be set prior to first call to forward for callbacks to be set properly"""
        self.fc1.weight = nn.Parameter(torch.cat( (x_mate, x_nonmate), dim=0))

    def encode(self, x):
        """Return 256d normalized forward encoding for input tensor x"""
        return self.net(x)[0]

    def classify(self, x):
        return self.fc1(self.net(x)[0])

    def num_classes(self):
        return 256 if self.fc1 is None else self.fc1.out_features

    def preprocess(self, img):
        """PIL image to input tensor"""
        """https://github.com/ox-vgg/vgg_face2/blob/master/standard_evaluation/pytorch_feature_extractor.py"""

        mean = (131.0912, 103.8827, 91.4953)

        short_size = 224.0
        crop_size = (224,224,3)
        im_shape = np.array(img.size)    # in the format of (width, height, *)
        img = img.convert('RGB')

        ratio = float(short_size) / np.min(im_shape)
        img = img.resize(size=(int(np.ceil(im_shape[0] * ratio)),   # width
                               int(np.ceil(im_shape[1] * ratio))),  # height
                         resample=PIL.Image.BILINEAR)

        x = np.array(img)  # image has been transposed into (height, width)
        newshape = x.shape[:2]
        h_start = (newshape[0] - crop_size[0])//2
        w_start = (newshape[1] - crop_size[1])//2
        x = x[h_start:h_start+crop_size[0], w_start:w_start+crop_size[1]]
        x = x - mean
        x = torch.from_numpy(x.transpose(2,0,1).astype(np.float32)).unsqueeze(0)
        return x

class Whitebox_resnet50_128(WhiteboxNetwork):

    def __init__(self, net):
        """https://github.com/ox-vgg/vgg_face2"""
        self.net = net
        self.net.eval()
        self.fc1 = nn.Linear(128, 2, bias=False)

    def set_triplet_classifier(self, x_mate, x_nonmate):
        """Replace classifier with binary mate and non-mate classifier, must be set prior to first call to forward for callbacks to be set properly"""
        self.fc1.weight = nn.Parameter(torch.cat( (x_mate, x_nonmate), dim=0))

    def encode(self, x):
        """Return 128d normalized forward encoding for input tensor x"""
        return self.net(x)[0]

    def classify(self, x):
        fc1_device = next(self.fc1.parameters()).device
        if fc1_device != x.device:
            self.fc1 = self.fc1.to(x.device)
        return self.fc1(self.net(x)[0])

    def num_classes(self):
        return 128 if self.fc1 is None else self.fc1.out_features

    def preprocess(self, img):
        """PIL image to input tensor"""
        """https://github.com/ox-vgg/vgg_face2/blob/master/standard_evaluation/pytorch_feature_extractor.py"""

        mean = (131.0912, 103.8827, 91.4953)

        short_size = 224.0
        crop_size = (224,224,3)
        im_shape = np.array(img.size)    # in the format of (width, height, *)
        img = img.convert('RGB')

        ratio = float(short_size) / np.min(im_shape)
        img = img.resize(size=(int(np.ceil(im_shape[0] * ratio)),   # width
                               int(np.ceil(im_shape[1] * ratio))),  # height
                         resample=PIL.Image.BILINEAR)

        x = np.array(img)  # image has been transposed into (height, width)
        newshape = x.shape[:2]
        h_start = (newshape[0] - crop_size[0])//2
        w_start = (newshape[1] - crop_size[1])//2
        x = x[h_start:h_start+crop_size[0], w_start:w_start+crop_size[1]]
        x = x - mean
        x = torch.from_numpy(x.transpose(2,0,1).astype(np.float32)).unsqueeze(0)
        return x


class Whitebox(nn.Module):
    def __init__(self, net, ebp_version=None, with_bias=None, eps=1E-16,
                 ebp_subtree_mode='affineonly_with_prior'):
        """
        Net must be WhiteboxNetwork object.

        ebp_version=7: Whitebox(..., eps=1E-12).weighted_subtree_ebp(..., do_max_subtree=True, subtree_mode='all', do_mated_similarity_gating=True)
        ebp_version=8: Whitebox(..., eps=1E-12).weighted_subtree_ebp(..., do_max_subtree=False, subtree_mode='all', do_mated_similarity_gating=True)
        ebp_version=9: Whitebox(..., eps=1E-12).weighted_subtree_ebp(..., do_max_subtree=True, subtree_mode='all', do_mated_similarity_gating=False)
        ebp_version=10: Whitebox(..., eps=1E-12).weighted_subtree_ebp(..., do_max_subtree=True, subtree_mode='norelu', do_mated_similarity_gating=True)
        ebp_version=11: Whitebox(..., eps=1E-12, with_bias=False).weighted_subtree_ebp (..., do_max_subtree=True, subtree_mode='all', do_mated_similarity_gating=True)

        """
        super(Whitebox, self).__init__()
        assert(isinstance(net, WhiteboxNetwork))
        self.net = net

        self.eps = eps
        self.layerlist = None
        self.ebp_ver = ebp_version
        if self.ebp_ver is None:
            self.ebp_ver = 6 # set to the latest
        elif self.ebp_ver < 4:
            raise RuntimeError('ebp version, if set, must be at least 4')
        self.convert_saliency_uint8 = (self.ebp_ver != 6)
        if with_bias is not None:
            self._ebp_with_bias = with_bias
        else:
            self._ebp_with_bias = self.ebp_ver == 11

        self.dA = []   # layer activation gradient
        self.A = []  # layer activations
        self.X = []  # W^{+T}*A, equivalent to Apox
        self.P = []  # MWP layer outputs
        self.P_prior = []  # MWP layer priors
        self.P_layername = []  # MWP layer names in recursive layer order

        # batch size is not applied to all functions, just embeddings
        self.batch_size = 32

        # Create layer visitor
        self._ebp_mode = 'disable'
        self.layerlist = self.net._layer_visitor(f_preforward=self._preforward_hook, f_forward=self._forward_hook)
        self._ebp_subtree_mode = ebp_subtree_mode

    def _preforward_hook(self, module, x_input):
        if self._ebp_mode == 'activation':
            if hasattr(module, 'orig_weight') and module.orig_weight is not None:
                module.weight.data.copy_(module.orig_weight.data)  # Restore original weights
                module.orig_weight = None
            if hasattr(module, 'orig_bias') and module.orig_bias is not None:
                module.bias.data.copy_(module.orig_bias.data)  # Restore original bias
                module.orig_bias = None
            return None
        elif self._ebp_mode == 'positive_activation':
            assert(len(self.A) > 0)  # must have called activation first
            if hasattr(module, 'weight'):
                module.orig_weight = module.weight.detach().clone()
                module.pos_weight = F.relu(module.orig_weight.data)
                module.weight.data.copy_( module.pos_weight.data )  # module backwards is on positive weights
            if self._ebp_with_bias and hasattr(module, 'bias') and module.bias is not None:
                module.orig_bias = module.bias.detach().clone()
                module.pos_bias = F.relu(module.orig_bias.data)
                module.bias.data.copy_( module.pos_bias.data )

            # Save layerwise positive activations (self.X = W^{+T}*A) in recursive layer visitor order
            self.X.append(tuple([F.relu(x.detach().clone()) for x in x_input]))
            A = self.A.pop(0)  # override forward input -> activation (A)
            self.A.append(A) # Reappend for EBP
            return A
        elif self._ebp_mode == 'ebp':
            assert(len(self.X) > 0 and len(self.A) > 0 and len(self.X)==len(self.A))  # must have called forward in activation and positive_activation mode first
            if hasattr(module, 'orig_weight') and module.orig_weight is not None:
                module.weight.data.copy_(module.orig_weight.data)  # restore
                module.orig_weight = None
            if self._ebp_with_bias and hasattr(module, 'orig_bias') and module.orig_bias is not None:
                module.bias.data.copy_(module.orig_bias.data)  # Restore original bias
                module.orig_bias = None
            return None
        elif self._ebp_mode == 'disable':
            if hasattr(module, 'orig_weight') and module.orig_weight is not None:
                module.weight.data.copy_(module.orig_weight.data)  # Restore original weights
                module.orig_weight = None
            if self._ebp_with_bias and hasattr(module, 'orig_bias') and module.orig_bias is not None:
                module.bias.data.copy_(module.orig_bias.data)  # Restore original bias
                module.orig_bias = None
            return None
        else:
            raise ValueError('invalid mode "%s"' % self._ebp_mode)

    def _forward_hook(self, module, x_input, x_output):
        """Forward hook called after forward is called for each layer to save off layer inputs and gradients, or compute EBP"""
        if self._ebp_mode == 'activation':
            # Save layerwise activations (self.A) and layerwise gradients (self.dA) in recursive layer visitor order
            for x in x_input:
                def _savegrad(x):
                    self.dA.append(x) # save layer gradient on backward
                x.register_hook(_savegrad)  # tensor hook to call _savegrad
            self.A.append(tuple([F.relu(x.detach().clone()) for x in x_input]))  # save non-negative layer activation on forward
            return None  # no change to forward output

        elif self._ebp_mode == 'positive_activation':
            return None

        elif self._ebp_mode == 'ebp':
            # Excitation backprop: https://arxiv.org/pdf/1608.00507.pdf, Algorithm 1, pg 9
            A = self.A.pop(0)   # An input, pre-computed in "activation" mode
            X = self.X.pop(0)   # X, (Alg. 1, step 2), pre-computed in "positive activation" mode

            # Affine layers only
            if hasattr(module, 'pos_weight'):
                # Step 1, W^{+}
                module.orig_weight = module.weight.detach().clone()
                module.weight.data.copy_( module.pos_weight.data )
            if self._ebp_with_bias and hasattr(module, 'pos_bias'):
                module.orig_bias = module.bias.detach().clone()
                module.bias.data.copy_( module.pos_bias.data )

            for (g, a, x) in zip(x_input, A, X):
                assert(g.shape == a.shape and g.shape == x.shape)
                def _backward_ebp(z):
                    # Tensor hooks are broken but "it's a feature", need operations in same scope to avoid memory leak
                    #   https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555
                    #   https://github.com/pytorch/pytorch/issues/12863
                    #   https://github.com/pytorch/pytorch/issues/25723

                    # Implement equation 10 (algorithm 1)
                    zh = F.relu(z)
                    p = torch.mul(a, zh)  # step 5, closure (a)
                    p_prior = self.P_prior.pop(0) if len(self.P_prior) > 0 else None
                    if p_prior is not None:
                        p.data.copy_(p_prior)  # override with prior
                    self.P_layername.append(str(module))  # MWP layer, closure (self.P_layername)
                    self.P.append(p) # marginal winning probability, closure (self.P)

                    # Subtree EBP modes (for analysis purposes)
                    if self._ebp_subtree_mode == 'affineonly':
                        # This mode sucks
                        if 'Conv' in str(module) or 'Linear' in str(module) or 'AvgPool' in str(module) or 'BatchNorm' in str(module):
                            y = torch.div(p, x + self.eps)  # step 3, closure (x)
                            return y  # step 4 (Z=W^{+}*Y), on next layer call to _backward_ebp, input to this lambda is Z
                        elif 'Sigmoid' in str(module) or 'ELU' in str(module) or 'Tanh' in str(module):
                            raise ValueError('layer "%s" is a special case (https://arxiv.org/pdf/1608.00507.pdf, eq 5), and is not yet supported' % str(module))
                        else:
                            return None
                    elif self._ebp_subtree_mode == 'affineonly_with_prior':
                        zh = torch.mul(p_prior > 0, z) if p_prior is not None else zh
                        p = torch.mul(p_prior > 0, p) if p_prior is not None else p
                        if 'Conv' in str(module) or 'Linear' in str(module) or 'AvgPool' in str(module) or 'BatchNorm' in str(module):
                            y = torch.div(p, x + self.eps)  # step 3, closure (x)
                            return y  # step 4 (Z=W^{+}*Y), on next layer call to _backward_ebp, input to this lambda is Z
                        elif 'Sigmoid' in str(module) or 'ELU' in str(module) or 'Tanh' in str(module):
                            raise ValueError('layer "%s" is a special case (https://arxiv.org/pdf/1608.00507.pdf, eq 5), and is not yet supported' % str(module))
                        else:
                            return zh  
                    elif self._ebp_subtree_mode == 'norelu':
                        # This mode is necessary for visualization of other networks, without backprop -> inf
                        if ('MaxPool' in str(module) or 'ReLU' in str(module)) and p_prior is not None:
                            return None
                        elif 'Sigmoid' in str(module) or 'ELU' in str(module) or 'Tanh' in str(module):
                            raise ValueError('layer "%s" is a special case (https://arxiv.org/pdf/1608.00507.pdf, eq 5), and is not yet supported' % str(module))
                        else:
                            y = torch.div(p, x + self.eps)  # step 3, closure (x)
                            return y  # step 4 (Z=W^{+}*Y), on next layer call to _backward_ebp, input to this lambda is Z
                    elif self._ebp_subtree_mode == 'all':
                        # This mode is best for weighted subtree on STR-Janus network.  Why?
                        y = torch.div(p, x + self.eps)  # step 3, closure (x)
                        return y  # step 4 (Z=W^{+}*Y), on next layer call to _backward_ebp, input to this lambda is Z
                    else:
                        raise ValueError('Invalid subtree mode "%s"' % self._ebp_subtree_mode)

                g.register_hook(_backward_ebp)
            return None
        elif self._ebp_mode == 'disable':
            return None
        else:
            raise ValueError('Invalid mode "%s"' % self._ebp_mode)

    def _float32_to_uint8(self, img):
        # float32 [0,1] rescaled to [0,255] uint8
        return np.uint8(255*((img - np.min(img))/(self.eps+(np.max(img)-np.min(img)))))

    def _scale_normalized(self, img):
        # float32 [0,1] 
        img = np.float32(img)
        return (img - np.min(img))/(self.eps+(np.max(img)-np.min(img)))

    def _mwp_to_saliency(self, P, blur_radius=2):
        """Convert marginal winning probability (MWP) output from EBP to uint8 saliency map, with pooling, normalization and blurring"""
        img = P  # pooled over channels
        if self.convert_saliency_uint8:
            img = np.uint8(255*((img - np.min(img))/(self.eps+(np.max(img)-np.min(img)))))  # normalized [0,255]
            img = np.array(PIL.Image.fromarray(img).filter(PIL.ImageFilter.GaussianBlur(radius=blur_radius)))  # smoothed
            img = np.uint8(255*((img - np.min(img))/(self.eps+(np.max(img)-np.min(img)))))  # renormalized [0,255]
        else:
            # version 6, avoid converting to 8 bit
            img = skimage.filters.gaussian(img, blur_radius)
            img = np.maximum(0, img)
            img /= (max(img.sum(), self.eps))
        return img

    # def _layers(self):
    #     """Random input EBP just to get layer order on forward.  This sets hooks."""
    #     img = self._float32_to_uint8(np.random.rand( 224,224,3 ))
    #     P = torch.zeros( (1, self.net.num_classes()), dtype=torch.float32 )
    #     x = self.net.preprocess(PIL.Image.fromarray(img))
    #     if next(self.net.net.parameters()).is_cuda:
        #         the following only works for single-gpu systems:
    #         P = P.cuda()
    #         x = x.cuda()
    #     self.ebp(x, P)
    #     P_layername = self.P_layername
    #     self._clear()
    #     return P_layername


    def _clear(self):
        (self.P, self.P_layername, self.dA, self.A, self.X) = ([], [], [], [], [])
        self.net.clear()


    def ebp(self, x, Pn, mwp=False):
        """Excitation backprop: forward operation to compute activations (An, Xn) and backward to compute Pn following equation (10)"""

        # Pre-processing
        x = x.detach().clone()   # if we do not clone, then the backward graph grows
        self._clear()  # if we do do not clear, then forward will accumulate self.A and self.dA

        # Forward activations
        self._ebp_mode = 'activation'
        y = self.net.classify(x.requires_grad_(True))
        self._ebp_mode = 'positive_activation'
        y = self.net.classify(x.requires_grad_(True))

        # EBP
        self._ebp_mode = 'ebp'
        Xn = self.net.classify(x.requires_grad_(True))
        Xn.backward(Pn, retain_graph=True)
        P = np.squeeze(np.sum(self.P[-2].detach().cpu().numpy(), axis=1)).astype(np.float32)  # pool over channels
        self._ebp_mode = 'disable'

        # Marginal winning probability or saliency map
        P = self._mwp_to_saliency(P) if not mwp else P
        return P

    def contrastive_ebp(self, img_probe, k_poschannel, k_negchannel):
        """Contrastive excitation backprop"""
        assert(k_poschannel >= 0 and k_poschannel < self.net.num_classes())
        assert(k_negchannel >= 0 and k_negchannel < self.net.num_classes())

        # Mated EBP
        P0 = torch.zeros( (1, self.net.num_classes()) );  P0[0][k_poschannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self.ebp(img_probe, P0)
        P_mate = self.P

        # Non-mated EBP
        P0 = torch.zeros( (1, self.net.num_classes()) );  P0[0][k_negchannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self.ebp(img_probe, P0)
        P_nonmate = self.P
        
        # Contrastive EBP
        mwp_mate = P_mate[-2] / torch.sum(P_mate[-2])
        mwp_nonmate = P_nonmate[-2] / torch.sum(P_nonmate[-2])
        mwp_contrastive = np.squeeze(np.sum(F.relu(mwp_mate - mwp_nonmate).detach().cpu().numpy(), axis=1).astype(np.float32))  # pool over channels
        return self._mwp_to_saliency(mwp_contrastive)

    def truncated_contrastive_ebp(self, img_probe, k_poschannel, k_negchannel, percentile=20):
        """Truncated contrastive excitation backprop"""
        assert(k_poschannel >= 0 and k_poschannel < self.net.num_classes())
        assert(k_negchannel >= 0 and k_negchannel < self.net.num_classes())

        # Mated EBP
        P0 = torch.zeros( (1, self.net.num_classes()) );  P0[0][k_poschannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self.ebp(img_probe, P0)
        P_mate = self.P

        # Non-mated EBP
        P0 = torch.zeros( (1, self.net.num_classes()) );  P0[0][k_negchannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self.ebp(img_probe, P0)
        P_nonmate = self.P
        
        # Truncated contrastive EBP
        mwp_mate = P_mate[-2] / torch.sum(P_mate[-2])
        mwp_nonmate = P_nonmate[-2] / torch.sum(P_nonmate[-2])

        (mwp_sorted, mwp_sorted_indices) = torch.sort(torch.flatten(mwp_mate.clone()))  # ascending
        mwp_sorted_cumsum = torch.cumsum(mwp_sorted, 0)  # for percentile
        percentile_mask = torch.zeros(mwp_sorted.shape)
        percentile_mask[mwp_sorted_indices] = (mwp_sorted_cumsum >= (percentile/100.0)*mwp_sorted_cumsum[-1]).type(torch.FloatTensor)
        percentile_mask = percentile_mask.reshape(mwp_mate.shape)
        percentile_mask = percentile_mask.to(img_probe.device)
        tcebp = F.relu(torch.mul(percentile_mask, mwp_mate) - torch.mul(percentile_mask, mwp_nonmate))
        mwp_truncated_contrastive = np.squeeze(np.sum(tcebp.detach().cpu().numpy(), axis=1).astype(np.float32))  # pool over channels
        return self._mwp_to_saliency(mwp_truncated_contrastive)


    def layerwise_ebp(self, img_probe, k_layer, mode='argmax', k_element=None, k_poschannel=0, mwp=True):
        """Layerwise excitation backprop"""
        """For a given layer, select the starting node according to a provided element or the provided mode"""
        assert(k_poschannel >= 0 and k_poschannel < self.net.num_classes())
        P0 = torch.zeros( (1, self.net.num_classes()) );  P0[0][k_poschannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self.ebp(img_probe, P0)
        P_mate = self.P

        self.P_prior = [None for x in self.P]  # all other layers propagate
        if mode == 'argmax':
            self.P_prior[k_layer] = torch.mul(P_mate[k_layer], 1.0 - torch.ne(P_mate[k_layer], torch.max(P_mate[k_layer])).type(torch.FloatTensor))
        elif mode == 'elementwise':
            assert(k_element is not None)  
            P = (0*(P_mate[k_layer].detach().clone())).flatten()
            P[k_element] = P_mate[k_layer].flatten()[k_element]
            self.P_prior[k_layer] = P.reshape(P_mate[k_layer].shape)
        else:
            raise ValueError('invalid layerwise EBP mode "%s"' % mode)

        return self.ebp(img_probe, 0.0*P0, mwp=mwp) 


    def layerwise_contrastive_ebp(self, img_probe, k_poschannel, k_negchannel, k_layer, mode='copy', percentile=80, k_element=None, gradlayer=None, mwp=False):
        """Layerwise contrastive excitation backprop"""

        import warnings
        warnings.warn("layerwise_contrastive_ebp is deprecated, use weighted_subtree_ebp instead")

        # Mated EBP
        assert(k_poschannel >= 0 and k_poschannel < self.net.num_classes())
        assert(k_negchannel >= 0 and k_negchannel < self.net.num_classes())
        P0 = torch.zeros( (1, self.net.num_classes()) );  P0[0][k_poschannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self.ebp(img_probe, P0)
        P_mate = self.P

        # Non-mated EBP
        P0 = torch.zeros( (1, self.net.num_classes()) );  P0[0][k_negchannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self.ebp(img_probe, P0)
        P_nonmate = self.P

        # Contrastive EBP
        self.P_prior = [None for x in self.P]  # all other layers propagate
        if mode == 'copy':
            # Pn is replaced with contrastive difference at layer k
            self.P_prior[k_layer] = F.relu(P_mate[k_layer] - P_nonmate[k_layer])
        elif mode == 'mean':
            # Pn is replaced with mean of contrastive difference and EBP
            self.P_prior[k_layer] = 0.5 * (P_mate[k_layer] + (F.relu(P_mate[k_layer] - P_nonmate[k_layer])))
        elif mode == 'product':
            # Product of EBP and contrast
            self.P_prior[k_layer] = torch.sqrt(torch.mul(P_mate[k_layer].type(torch.DoubleTensor), F.relu(P_mate[k_layer] - P_nonmate[k_layer]).type(torch.DoubleTensor))).type(torch.FloatTensor)
        elif mode == 'argmax':
            Pn_prior = F.relu(P_mate[k_layer] - P_nonmate[k_layer])
            self.P_prior[k_layer] = torch.mul(Pn_prior, 1.0 - torch.ne(Pn_prior, torch.max(Pn_prior)).type(torch.FloatTensor))
        elif mode == 'argmax_product':
            Pn_prior = torch.sqrt(torch.mul(P_mate[k_layer].type(torch.DoubleTensor), F.relu(P_mate[k_layer] - P_nonmate[k_layer]).type(torch.DoubleTensor))).type(torch.FloatTensor)
            self.P_prior[k_layer] = torch.mul(Pn_prior, 1.0 - torch.ne(Pn_prior, torch.max(Pn_prior)).type(torch.FloatTensor))
        elif mode == 'percentile' or mode == 'percentile_argmax':
            assert(percentile >= 0 and percentile <= 100)
            Pn = P_mate[k_layer]
            Pn_prior = F.relu(P_mate[k_layer] - P_nonmate[k_layer])
            (Pn_sorted, Pn_sorted_indices) = torch.sort(torch.flatten(Pn.clone()))  # ascending
            Pn_sorted_cumsum = torch.cumsum(Pn_sorted, 0)  # for percentile
            Pn_mask = torch.zeros(Pn_sorted.shape)
            Pn_mask[Pn_sorted_indices] = (Pn_sorted_cumsum >= (percentile/100.0)*Pn_sorted_cumsum[-1]).type(torch.FloatTensor)
            Pn_mask = Pn_mask.reshape(Pn.shape)
            self.P_prior[k_layer] = torch.mul(Pn_mask, Pn_prior.type(torch.FloatTensor)).clone()
            if mode == 'percentile_argmax':
                Pn = self.P_prior[k_layer]
                Pn_argmax = torch.mul(Pn, 1.0 - torch.ne(Pn, torch.max(Pn)).type(torch.FloatTensor))
                self.P_prior[k_layer] = Pn_argmax
        elif mode == 'elementwise':
            assert(gradlayer[k_layer].shape == P_mate[k_layer].shape)
            C = F.relu(P_mate[k_layer] - P_nonmate[k_layer])
            P = (0*C.detach().clone()).flatten()
            P[k_element] = C.flatten()[k_element]
            self.P_prior[k_layer] = P.reshape(C.shape)
        else:
            raise ValueError('unknown contrastive ebp mode "%s"' % mode)

        return self.ebp(img_probe, 0.0*P0, mwp=mwp)


    def weighted_subtree_ebp(self, img_probe, k_poschannel, k_negchannel, topk=1, verbose=True, do_max_subtree=False, do_mated_similarity_gating=True, subtree_mode='norelu', do_mwp_to_saliency=True): 
        """Weighted subtree EBP"""

        # Forward and backward to save triplet loss data gradients in layer visitor order
        self._ebp_subtree_mode = subtree_mode
        self._ebp_mode = 'activation'
        x = img_probe.detach().clone()   # if we do not clone, then the backward graph grows for some reason
        y = self.net.classify(x.requires_grad_(True))
        y0 = torch.tensor([0]).to(y.device)
        # y1 = torch.tensor([1]).cuda() if next(self.net.net.parameters()).is_cuda else torch.tensor([1])
        F.cross_entropy(y, y0).backward(retain_graph=True)  # Binary classes = {mated, nonmated}
        gradlist = self.dA
        self._clear()

        # Forward and backward to save mate and non-mate data gradients in layer visitor order
        if not do_mated_similarity_gating:
            y = self.net.classify(x.requires_grad_(True))
            F.cross_entropy(y,y0).backward(retain_graph=True)
            gradlist_ce = self.dA
            self._clear()

        y[0][0].backward(retain_graph=True)
        # gradlist_mated = copy.deepcopy(self.dA)  # TESTING deepcopy
        gradlist_mated = self.dA
        self._clear()

        y[0][1].backward(retain_graph=True)
        # gradlist_nonmated = copy.deepcopy(self.dA)  # TESTING deepcopy
        gradlist_nonmated = self.dA
        self._clear()

        # Select subtrees using loss weighting
        P_img = []
        P_subtree = []
        P_subtree_idx = []
        n_layers = len(gradlist_mated)
        assert len(gradlist_mated) == len(gradlist_nonmated)
        for k in range(0, n_layers-1):  # not including image layer
            # Given loss function L, a positive gradient (dL/dx) means that x is DECREASED to DECREASE the loss
            # Given loss function L, a negative gradient (-dL/dx) means that x is INCREASED to DECREASE the loss (more excitory -> reduces loss)
            if do_mated_similarity_gating:
                # Mated similarity  must increase (positive gradient), non-mated similarity decrease (negative gradient, more excitory)
                p = torch.max(torch.mul(gradlist_mated[k]>=0, -gradlist_nonmated[k]))
                k_argmax = torch.argmax(torch.mul(gradlist_mated[k]>=0, -gradlist_nonmated[k]))                
            else:
                # Triplet ebp loss must decrease (negative gradient, more excitory), non-mated similarity decrease (negative gradient, more excitory)
                p = torch.max(torch.mul(gradlist_ce[k]<0, -gradlist_nonmated[k]))
                k_argmax = torch.argmax(torch.mul(gradlist_ce[k]<0, -gradlist_nonmated[k]))                
            P_subtree.append(float(p))
            P_subtree_idx.append(k_argmax)
        k_subtree = np.argsort(np.array(P_subtree))  # ascending, one per layer

        # Generate layerwise EBP for each selected subtree
        for k in k_subtree:
            P_img.append(self.layerwise_ebp(x, k_layer=k, k_poschannel=k_poschannel, k_element=P_subtree_idx[k], mode='elementwise'))
            if verbose:
                print('[weighted_subtree_ebp][%d]: layername=%s, grad=%f' % (k, self.P_layername[k], P_subtree[k]))

        # Merge MWP from each subtree, weighting by convex combination of subtrees, weights proportional to loss gradient
        k_valid = [np.max(P) > 0 for P in P_img]
        k_subtree_valid = [k for (k,v) in zip(k_subtree, k_valid) if v == True and k != 1][-topk:] # FIXME: k==1 is for STR-Janus Multiply() layer
        if len(k_subtree_valid) == 0:
            # assert(len(k_subtree_valid)>0)  # Should never be empty
            raise RuntimeError(
                'Failed to calculate valid subtrees. The ebp subtree mode '
                '(%s) may not support by this type of network. You may want '
                'to try the "affineonly_with_prior" ebp subtree mode.' %
                self._ebp_subtree_mode
            )
        P_img_valid = [p for (p,k,v) in zip(P_img, k_subtree, k_valid) if v == True and k != 1][-topk:] 
        P_subtree_valid = [P_subtree[k] for k in k_subtree_valid]
        P_subtree_valid_norm = self._scale_normalized(P_subtree_valid) if not np.sum(self._scale_normalized(P_subtree_valid)) == 0  else np.ones_like(P_subtree_valid)
        if do_max_subtree:
            smap = np.max(np.dstack([float(p_subtree_valid_norm)*np.array(P)*(1.0 / (np.max(P)+1E-12)) for (p_subtree_valid_norm, P) in zip(P_subtree_valid_norm, P_img_valid)]), axis=2)
        else:
            if len(P_subtree_valid_norm) > 0:
                smap = np.sum(np.dstack([float(p_subtree_valid_norm)*np.array(P)*(1.0 / (np.max(P)+1E-12)) for (p_subtree_valid_norm, P) in zip(P_subtree_valid_norm, P_img_valid)]), axis=2)
            else:
                smap = 0*P_img[0]  # empty saliency map

        # Generate output saliency map
        if self.convert_saliency_uint8:
            smap = self._float32_to_uint8(smap)
        else:
            smap /= max(smap.sum(), self.eps)

        return (
            self._mwp_to_saliency(smap) if do_mwp_to_saliency else smap,
            [self._mwp_to_saliency(P) if do_mwp_to_saliency else P for P in P_img_valid],
            P_subtree_valid,
            k_subtree_valid)

    def ebp_subtree_mode(self):
        return self._ebp_subtree_mode

    def encode(self, x):
        """ Expose wbnet encode function.
        """
        return self.net.encode(x)

    def embeddings(self, images, norm=True):
        """ Calculate embeddings from numpy float images.

            A wrapper to help fit into existing API.
        """
        if isinstance(images, pd.DataFrame):
            imagesT = [self.convert_from_numpy(im)[0]
                       for im in utils.image_loader(images)]
        elif isinstance(images[0], torch.Tensor):
            assert images[0].ndim == 3  # [3, spatial dims]
            imagesT = images
        elif isinstance(images[0], np.ndarray):
            # currently only handle np arrays that are in network format
            # already
            assert images[0].shape[0] in (1,3)  # grayscale or RGB
            imagesT = [torch.from_numpy(im).float() for im in images]
        else:
            imagesT = [self.convert_from_numpy(im)[0]
                       for im in utils.image_loader(images)]

        if not isinstance(imagesT, torch.Tensor):
            # imagesT = torch.cat(imagesT)
            imagesT = torch.stack(imagesT)

        batches = torch.split(imagesT, self.batch_size, dim=0)
        embeds = []
        for k, batch in enumerate(batches):
            batch = batch.to(next(self.net.net.parameters()).device)
            embeds.append(self.encode(batch).detach().cpu().numpy())
        embeds = np.concatenate(embeds)

        if norm:
            embeds = (
                embeds.reshape((embeds.shape[0], -1)) /
                np.linalg.norm(embeds.reshape((embeds.shape[0], -1)),
                               axis=1, keepdims=True)
                ).reshape(embeds.shape)

        return embeds

    def convert_from_numpy(self, img):
        """ Converts float RGB image (WxHx3) with range 0 to 1 or uint8 image
            to tensor (1x3XWxH).
        """
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255

        if img.max() > 1 + 1e-6 and img.min() > 0 - 1e-6:
            img = img / 255

        if img.max() > 1 + 1e-6 or img.min() < 0 - 1e-6:
            import pdb
            pdb.set_trace()
            img = (img - img.min()) / (img.max() - img.max() + 1e-6)

        img = skimage.transform.resize(img, (224, 224), preserve_range=True)
        img = (img * 255).astype(np.uint8)
        img = PIL.Image.fromarray(img).convert('RGB')
        img = self.net.preprocess(img)
        return img

    def preprocess_loader(self, images, returnImageIndex=False, repeats=1):
        """ Iterates over tuple: (displayable image, tensor, fn)

            Tensor should have 3 dimensions (for a single image).

            Included to match snet interface.

            Preprocessing depends on specific network.
        """
        for im, fn in utils.image_loader(
            images,
            returnFileName=True,
            returnImageIndex=returnImageIndex,
            repeats=repeats,
        ):
            imT = self.convert_from_numpy(im)
            yield im, imT[0], fn
