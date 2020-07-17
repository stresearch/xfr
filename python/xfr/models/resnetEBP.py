# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.



import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image, ImageFilter
import numpy as np
import pdb
from heapq import nlargest
import skimage
import pandas as pd
import six
import imageio

from xfr.models.resnet import Bottleneck
from xfr.models.resnet import convert_resnet101v4_image

from xfr import utils

MEAN_RGB = np.array(([122.782, 117.001, 104.298]))

import warnings
warnings.warn('WARNING - xfr.models.resnetEBP is deprecated.  Use xfr.models.whitebox instead') 

class Resnet101_EBP(nn.Module):
    def __init__(self, net, ebp_version=1):
        """Net should not include a loss function

            ebp_versions:
            1 - original
            2 - ignore subtrees with zero saliency maps
            3 - return floats
            None - set it to the most recent version
        """
        super(Resnet101_EBP, self).__init__()
        self.net = net
        self.net.eval()
        self.Pn = []  # MWP layer outputs
        self.Pn_prior = []
        self.Pn_layername = []
        self.eps = 1E-9
        self.layerlist = None
        self.ebp_ver = ebp_version
        if self.ebp_ver is None:
            self.ebp_ver = 3 # set to the latest
        self.dx = []

        # batch size is not applied to all functions, just embeddings
        self.batch_size = 32


    def _forward_savegrad(self, module, x_input, x_output):
        for (k, x) in enumerate(x_input):
            def _savegrad(x):
                self.dx.append(x)
            x.register_hook(_savegrad)
        return None

        
    def _preforward_ebp(self, module, x_inputs):
        """x_input = {'An'An, 'Xn':Xn} -> return An, save module.Xn"""
        x_inputs_split = []
        module.Xn = []  # cleanup
        module.orig_weight = None  # cleanup
        for x_input in x_inputs:
            assert(isinstance(x_input, dict))  # must be output of _forward
            module.Xn.append(x_input['Xn'])  # save for use in forward
            x_inputs_split.append(x_input['An'])  # propagated in forward, must be positive
        return tuple(x_inputs_split)

    def _forward_ebp(self, module, x_input, x_output):
        """An = x_input, Xn = module.Xn, returns x_output={'An'x_output, 'Xn':Xnm1}"""
        Xn = [F.relu(x) if x is not None else x for x in module.Xn]  # non-negative
        An = [F.relu(x) for x in x_input]  # non-negative
        if 'Add' in str(module):
            assert(len(An) == 2)
            Xnm1 = module.forward(An[0], An[1])  # for next layer, disable hooks (do not use __call__)
        elif 'Conv2d' in str(module) or 'Linear' in str(module) or 'BatchNorm2d' in str(module):
            assert(len(An) == 1)
            module.orig_weight = module.weight.clone()  # to be restored in backward, without clone results explode to infinity
            module.weight.data.copy_(F.relu(module.weight.data))  # module backwards is on positive weights
            #module.bias.data.copy_(F.relu(module.bias.data))  # FIXME: do we update bias too?
            Xnm1 = module.forward(An[0])  # Input and weights non-negative
        else:
            assert(len(An) == 1)
            Xnm1 = module.forward(An[0])  # for next layer, disable hooks (do not use __call__)
        Xnm1 = F.relu(Xnm1)  # non-negative

        # Backward hooks are fundamentally broken, do not use - use tensor hooks on gradient instead
        #   https://github.com/pytorch/pytorch/issues/25723
        # But, tensor hooks are also broken but "it's a feature", need function in same scope to avoid memory leak
        #   https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555
        #   https://github.com/pytorch/pytorch/issues/12863
        # but if we do this, we are freeing variables before they are used by using in place layers?  Calling the same tensor twice on Add,
        # which means we catch the exception to handle this.  Yuck.
        for (k, (x, an, xn)) in enumerate(zip(x_input, An, Xn)):
            def _backward_ebp_local(x):
                try:
                    nonlocal an, xn  # closure variables
                    x_out = self._backward_ebp(module, x, an, xn)
                    del an, xn  # free closure variables
                    return x_out
                except:
                    if "Add" not in str(module):
                        print(module)  # Add() screws things up
                        # FIXME: assertions in backward will fail silently.  Not good. Raise here.
                        raise ValueError('uncaught exception in backward')
                    return None  # no change to gradient
            x.register_hook(_backward_ebp_local)
        return {'An':x_output, 'Xn':Xnm1.clone()}

    def _backward_ebp(self, module, x, An, Xn):
        """Compute equation (10) using tensor hooks"""
        # inputs:
        #   x:  The gradient tensor to be updated.  This contains Zn
        #
        # notation:
        #   P_{n-1}:  MWP at output of forward for this layer, same shape as input to backward (grad_output)
        #   P_{n}:  MWP at input of forward for this layer
        #   X_{n-1}:  W^{+}^{T} * An.  Same shape as P_{n-1}
        #   Y_{n-1}:  P_{n-1} ./ X_{n-1}, same shape as P_{n-1}
        #   Z_{n}:  W^{+} * Y_{n-1}.  Same shape as P_n
        #   An:  Activations at input of forward for this layer
        #   A_{n-1}:  Activations at output of forward for this layer
        #
        # backward:
        #   P_n = An .* grad_output (saved)
        #   returns: Y_n = P_n / Xn
        #
        # init: what is passed backwards is not P_n, but rather Y=(P_n / X).
        assert(len(An) <= 1)

        Zn = F.relu(x.data)  # assumed non-negative, but this is not true always.  why?
        Pn = torch.mul(torch.reshape(An[0], x.shape), Zn)
        Pn_prior = self.Pn_prior.pop(0) if len(self.Pn_prior) > 0 else None
        if Pn_prior is not None:
            Pn.data.copy_(Pn_prior)  # Pn override with prior
        self.Pn.append(Pn)   # MWP save
        self.Pn_layername.append(str(module)) # MWP layer name for traceability
        Yn = torch.div(Pn, torch.reshape(Xn + self.eps, x.shape)) if Xn is not None  else Pn
        if module.orig_weight is not None:
            module.weight.data.copy_(module.orig_weight.data)  # Restore original weights
        return Yn


    def _layer_visitor(self, f_forward=None, f_preforward=None, net=None):
        """Recursively assign forward hooks for all layers into Sequential and Bottleneck containers that are not ReLU (in-place causes problems)"""
        """Returns list in order of forward visit with {'name'n:, 'hook':h} dictionary"""
        layerlist = []
        net = self.net if net is None else net
        for name, layer in net._modules.items():
            if isinstance(layer, nn.Sequential) or isinstance(layer, Bottleneck):
                layerlist.append( {'name':str(layer), 'hooks':[None]})
                layerlist = layerlist + self._layer_visitor(f_forward, f_preforward, layer)
            elif f_forward is not None and f_preforward is not None:
                layerlist.append( {'name':str(layer), 'hooks':[layer.register_forward_hook(f_forward),
                                                               layer.register_forward_pre_hook(f_preforward)]})
            if isinstance(layer, nn.BatchNorm2d):
                #layer.eval()
                #layer.track_running_stats = False  (do not disable, this screws things up)
                #layer.affine = False
                pass

        return layerlist


    def num_layers(self):
        return len(self._layer_visitor())

    def encode(self, x):
        """Return 512d normalized forward encoding for input tensor x"""
        return self.net.forward(x, mode='encode')

    def normalize(self, img):
        ret = np.uint8(255*((img - np.min(img))/(self.eps+(np.max(img)-np.min(img)))))  # normalized [0,255]
        return ret

    def mwp_to_saliency(self, P, blur_radius=2):
        """Convert MWP to uint8 saliency map, with pooling, normalization and blurring"""
        if self.ebp_ver <= 2:
            img = np.squeeze(np.sum(P.detach().cpu().numpy(), axis=1)).astype(np.float32)  # pool over channels
            img = np.uint8(255*((img - np.min(img))/(self.eps+(np.max(img)-np.min(img)))))  # normalized [0,255]
            img = np.array(PIL.Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=blur_radius)))  # smoothed
            img = np.uint8(255*((img - np.min(img))/(self.eps+(np.max(img)-np.min(img)))))  # renormalized [0,255]
        else:
            # version 3, avoid converting to 8 bit
            img = np.squeeze(np.sum(P.detach().cpu().numpy(), axis=1)).astype(np.float32)  # pool over channels
            img = skimage.filters.gaussian(img, blur_radius)
            img = np.maximum(0, img)
            img /= (max(img.sum(), self.eps))
        return img

    def clear(self):
        # Clear gradients, multiple calls to backwards should not accumulate
        (self.Pn, self.Pn_layername, self.dx) = ([], [], [])
        for p in self.net.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def ebp(self, x, Pn):
        """Excitation backprop: forward operation to compute activations (An, Xn) and backward to compute Pn following equation (10)"""

        # Pre-processing
        x = x.detach().clone()   # if we do not clone, then the backward graph grows for some reason
        self.clear()

        # Callbacks
        if self.layerlist is None:
            # Layer visitors can only be added once, and cannot be removed after forward, even using hook.remove().  Yuck..
            self.layerlist = self._layer_visitor(f_preforward=self._preforward_ebp, f_forward=self._forward_ebp)

        # Forward activations
        x_input = {'An':x.requires_grad_(True), 'Xn':None}
        y = self.net.forward(x_input, mode='classify')
        (An, Xn) = (y['An'], y['Xn'])

        # EBP backward MWP
        Yn = torch.div(Pn, Xn+self.eps) if Pn is not None else None
        An.backward(Yn, retain_graph=True)
        return self.mwp_to_saliency(self.Pn[-2])

    def set_triplet_classifier(self, x_mate, x_nonmate):
        """Replace classifier with binary mate and non-mate classifier, must be set prior to first call to forward for callbacks to be set properly"""
        self.net.fc2 = nn.Linear(512, 2, bias=False)
        self.net.fc2.weight = nn.Parameter(torch.cat( (x_mate, x_nonmate), dim=0))

    def negate_classifier(self):
        self.net.fc2.weight = torch.neg(self.net.fc2.weight)
        self.net.fc2.bias = torch.neg(self.net.fc2.bias)

    def contrastive_ebp(self, img_probe, k_poschannel, k_negchannel, k_layer, mode='copy', percentile=50):
        # Mated EBP
        P0 = torch.zeros( (1, self.net.fc2.out_features) );  P0[0][k_poschannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self.ebp(img_probe, P0)
        P_mate = self.Pn

        # Non-mated EBP
        P0 = torch.zeros( (1, self.net.fc2.out_features) );  P0[0][k_negchannel] = 1.0;  # one-hot
        P0 = P0.to(img_probe.device)
        self.ebp(img_probe, P0)
        P_nonmate = self.Pn

        # Contrastive EBP
        self.Pn_prior = [None for x in self.Pn]  # all other layers propagate
        if mode == 'copy':
            # Pn is replaced with contrastive difference at layer k
            self.Pn_prior[k_layer] = F.relu(P_mate[k_layer] - P_nonmate[k_layer])
        elif mode == 'mean':
            # Pn is replaced with mean of contrastive difference and EBP
            self.Pn_prior[k_layer] = 0.5 * (P_mate[k_layer] + (F.relu(P_mate[k_layer] - P_nonmate[k_layer])))
        elif mode == 'product':
            # Product of EBP and contrast
            self.Pn_prior[k_layer] = torch.sqrt(torch.mul(P_mate[k_layer].type(torch.DoubleTensor), F.relu(P_mate[k_layer] - P_nonmate[k_layer]).type(torch.DoubleTensor))).type(torch.FloatTensor)
        elif mode == 'argmax':
            Pn_prior = F.relu(P_mate[k_layer] - P_nonmate[k_layer])
            self.Pn_prior[k_layer] = torch.mul(Pn_prior, 1.0 - torch.ne(Pn_prior, torch.max(Pn_prior)).type(torch.FloatTensor))
        elif mode == 'argmax_product':
            Pn_prior = torch.sqrt(torch.mul(P_mate[k_layer].type(torch.DoubleTensor), F.relu(P_mate[k_layer] - P_nonmate[k_layer]).type(torch.DoubleTensor))).type(torch.FloatTensor)
            self.Pn_prior[k_layer] = torch.mul(Pn_prior, 1.0 - torch.ne(Pn_prior, torch.max(Pn_prior)).type(torch.FloatTensor))
        elif mode == 'percentile' or mode == 'percentile_argmax':
            assert(percentile >= 0 and percentile <= 100)
            Pn = P_mate[k_layer]
            Pn_prior = F.relu(P_mate[k_layer] - P_nonmate[k_layer])
            (Pn_sorted, Pn_sorted_indices) = torch.sort(torch.flatten(Pn.clone()))  # ascending
            Pn_sorted_cumsum = torch.cumsum(Pn_sorted, 0)  # for percentile
            Pn_mask = torch.zeros(Pn_sorted.shape)
            Pn_mask[Pn_sorted_indices] = (Pn_sorted_cumsum > (percentile/100.0)*Pn_sorted_cumsum[-1]).type(torch.FloatTensor)
            Pn_mask = Pn_mask.reshape(Pn.shape)
            self.Pn_prior[k_layer] = torch.mul(Pn_mask, Pn_prior.type(torch.FloatTensor)).clone()
            if mode == 'percentile_argmax':
                Pn = self.Pn_prior[k_layer]
                Pn_argmax = torch.mul(Pn, 1.0 - torch.ne(Pn, torch.max(Pn)).type(torch.FloatTensor))
                self.Pn_prior[k_layer] = Pn_argmax
        else:
            raise ValueError('unknown contrastive ebp mode')

        ebp = self.ebp(img_probe, 0.0*P0)
        return ebp

    def subtree_ebp(self, img_probe, k_poschannel, k_negchannel,
                    percentile=20, mode='percentile_argmax', topk=1):
        """Return truncated contrastive EBP at the selected layer with the
           maximum truncated contrastive MWP over all layers
        """
        assert('percentile' in mode)
        P_img = []
        P_subtree = []
        for k in range(0, self.num_layers()):
            P_img.append(self.contrastive_ebp(
                img_probe, k_poschannel, k_negchannel,
                k_layer=k, mode=mode, percentile=percentile))
            P_subtree.append(torch.max(self.Pn[k] / (1E-12+torch.sum(self.Pn[k]))) * self.Pn[k].numel())

        if self.ebp_ver > 1:
            # Zero P_subtree that have a zero saliency map
            mask = np.stack([pim.max() > 0 for pim in P_img])
            P_subtree = [arr * torch.tensor(incl) for arr, incl in
                         zip(P_subtree, mask)]

        k_subtree = np.argsort(np.array(P_subtree))[-topk:]  # ascending

        smap = np.sum(np.dstack([P_img[k] for k in k_subtree]), axis=2)

        if self.ebp_ver <= 2:
            smap = self.normalize(smap)
        else:
            smap /= max(smap.sum(), self.eps)
        # if self.ebp_ver > 1:
        #     assert smap.max() > 0

        return (
            smap,
            [P_subtree[k] for k in k_subtree],
            k_subtree)


    def weighted_subtree_ebp(self, img_probe, k_poschannel, k_negchannel, percentile=20, mode='percentile_argmax', topk=1):

        # Forward and backward to save triplet loss data gradients in layer visitor order
        self.clear()
        x = img_probe.detach().clone()   # if we do not clone, then the backward graph grows for some reason
        self._layer_visitor(f_preforward=None, f_forward=self._forward_savegrad)  # FIXME: layer visitors twice?
        y = self.net.forward(x.requires_grad_(True), mode='classify')  # FIXME: general network?
        F.cross_entropy(y, 0).backward()  # Binary classes = {mated, nonmated}
        gradlist = self.dx

        # FIXME: this needs to remove layer visitors before calling constrastive EBP

        assert('percentile' in mode)
        P_img = []
        P_subtree = []
        for k in range(0, self.num_layers()):
            P_img.append(self.contrastive_ebp(
                img_probe, k_poschannel, k_negchannel,
                k_layer=k, mode=mode, percentile=percentile))
            p = torch.max(self.Pn[k] / (1E-12+torch.sum(self.Pn[k]))) * self.Pn[k].numel()
            P_subtree.append(torch.mul(p, gradlist[k]))  # gradient weighted

        if self.ebp_ver > 1:
            # Zero P_subtree that have a zero saliency map
            mask = np.stack([pim.max() > 0 for pim in P_img])
            P_subtree = [arr * torch.tensor(incl) for arr, incl in
                         zip(P_subtree, mask)]

        k_subtree = np.argsort(np.array(P_subtree))[-topk:]  # ascending

        smap = np.sum(np.dstack([P_img[k] for k in k_subtree]), axis=2)

        if self.ebp_ver <= 2:
            smap = self.normalize(smap)
        else:
            smap /= max(smap.sum(), self.eps)
        # if self.ebp_ver > 1:
        #     assert smap.max() > 0

        return (
            smap,
            [P_subtree[k] for k in k_subtree],
            k_subtree)

    def embeddings(self, images, norm=True):
        """ Calculate embeddings from numpy float images.

            A wrapper to help fit into existing API.
        """
        if isinstance(images, pd.DataFrame):
            imagesT = [Resnet101_EBP.convert_from_numpy(im)[0]
                       for im in utils.image_loader(images)]
        elif isinstance(images[0], torch.Tensor):
            assert images[0].ndim == 3  # [3, spatial dims]
            imagesT = images
        elif isinstance(images[0], np.ndarray):
            # If third channel equals 3, assume images need preprocessing
            if images[0].shape[2] == 3:
                imagesT = [convert_resnet101v4_image(im) for im in images]
            else:
                assert images[0].shape[0] == 3
                imagesT = [torch.from_numpy(im).float() for im in images]
        else:
            imagesT = [Resnet101_EBP.convert_from_numpy(im)[0]
                       for im in utils.image_loader(images)]

        if not isinstance(imagesT, torch.Tensor):
            # imagesT = torch.cat(imagesT)
            imagesT = torch.stack(imagesT)

        batches = torch.split(imagesT, self.batch_size, dim=0)
        embeds = []
        for k, batch in enumerate(batches):
            batch = batch.to(next(self.net.parameters()).device)
            embeds.append(self.encode(batch).detach().cpu().numpy())
        embeds = np.concatenate(embeds)

        if norm:
            embeds = (
                embeds.reshape((embeds.shape[0], -1)) /
                np.linalg.norm(embeds.reshape((embeds.shape[0], -1)),
                               axis=1, keepdims=True)
                ).reshape(embeds.shape)

        return embeds

    @staticmethod
    def convert_from_numpy(img):
        assert img.max() <= 1.00001
        assert img.min() >= -0.00001
        img = skimage.transform.resize(img, (224, 224), preserve_range=True)
        img = (img * 255).astype(np.uint8)
        img = PIL.Image.fromarray(img).convert('RGB')
        img = convert_resnet101v4_image(img).unsqueeze(0)
        return img

    @staticmethod
    def preprocess_loader(images, returnImageIndex=False, repeats=1):
        """ Iterates over tuple: (displayable image, tensor, fn)

            Tensor should have 3 dimensions (for a single image).

            Included to match snet interface.
        """
        for im, fn in utils.image_loader(
            images,
            returnFileName=True,
            returnImageIndex=returnImageIndex,
            repeats=repeats,
        ):
            imT = Resnet101_EBP.convert_from_numpy(im)
            yield im, imT[0], fn

#    @staticmethod
#    def image_loader(images, returnImageIndex=False, returnFileName=False, repeats=1):
#        """ Iterates over tuple: (displayable image, fn)
#
#            This function is a modification of preprocess_loader from
#            engine/experimental.py which has been modified to remove caffe
#            dependencies.
#
#            fn will be None if images is a numpy array
#
#            Args:
#                repeats: number of times each image is returned. If it is set to a
#                    number higher than 1, then the repeat number is added to return
#                    tuple. Used for channel-wise EBP.
#
#            Raises:
#                ValueError: might be raised when attempting to load image
#        """
#        if isinstance(images, pd.DataFrame):
#            for i, (_, imginfo) in enumerate(images.iterrows()):
#                img, sid, fn, _ = (
#                    Resnet101_EBP.crop_example_no_name(imginfo))
#                assert img.max() <= 1.0 and img.min() >= 0.0
#
#                # img = (img * 255).astype(np.uint8)
#                # img = PIL.Image.fromarray(img).convert('RGB')
#
#                ret = [img]
#
#                if returnImageIndex:
#                    ret.append(i)
#                if returnFileName:
#                    ret.append(fn)
#
#                if repeats == 1:
#                    if len(ret) == 1:
#                        yield ret[0]
#                    else:
#                        yield tuple(ret)
#                else:
#                    for repeat_num in range(repeats):
#                        yield tuple(ret + [repeat_num])
#        else:
#            for i, img in enumerate(images):
#                if isinstance(img, np.ndarray):
#                    assert img.ndim == 3 and img.shape[2] == 3
#                    fn = None
#                    cropped = img
#                elif isinstance(img, six.string_types):
#                    fn = img
#                    img = imageio.imread(fn)
#                    img = img.astype(float) / 255
#                    cropped = Resnet101_EBP.center_crop(img, convert_uint8=False)
#                else:
#                    raise NotImplementedError('Unhandled type %s' %
#                                              type(img))
#
#                ret = [cropped]
#
#                if returnImageIndex:
#                    ret.append(i)
#                if returnFileName:
#                    ret.append(fn)
#
#                if repeats == 1:
#                    if len(ret) == 1:
#                        yield ret[0]
#                    else:
#                        yield tuple(ret)
#                else:
#                    for repeat_num in range(repeats):
#                        yield tuple(ret + [repeat_num])
#
#    @staticmethod
#    def crop_image(img, crop_xywh=None, crop_tblr=None, roi_method='expand'):
#        """ Copy of max_tracker function, without caffe and related dependencies.
#        """
#        if crop_xywh is not None:
#            x = int(round(crop_xywh[0]))
#            y = int(round(crop_xywh[1]))
#            w = int(round(crop_xywh[2]))
#            h = int(round(crop_xywh[3]))
#        if crop_tblr is not None:
#            y = int(round(crop_tblr[0]))
#            y2 = int(round(crop_tblr[1]))
#            x = int(round(crop_tblr[2]))
#            x2 = int(round(crop_tblr[3]))
#            w = y2 - y
#            h = x2 - x
#
#        center_x = x + w // 2
#        center_y = y + h // 2
#
#        if roi_method == 'constrict':
#            cropDim = int(min(w, h))
#        elif roi_method == 'constrict80':
#            cropDim = int(min(w, h) * 0.8)
#        elif roi_method == 'constrict50':
#            cropDim = int(min(w, h) * 0.5)
#        else:
#            assert roi_method == 'expand'
#            # the crop dimensions can't be larger than the image
#            cropDim = min(
#                    max(w, h),
#                    min(img.shape[0], img.shape[1]))
#        top = max(0, center_y - cropDim // 2)
#        left = max(0, center_x - cropDim // 2)
#        # make bottom and right relative to (potentially shifted) top and left
#        bottom = min(img.shape[0], top + cropDim)
#        right = min(img.shape[1], left + cropDim)
#        # if hit bottom or right border, shift top or left
#        top = max(0, min(top, bottom - cropDim))
#        left = max(0, min(left, right - cropDim))
#
#        cropped = img[top : bottom,
#                left : right,
#                :]
#        # Image.fromarray((cropped*255).astype(np.uint8)).show()
#        return (cropped, (top, bottom, left, right))
#
#    @staticmethod
#    def crop_example_no_name(ex, data_root=''):
#        '''
#        Copy of function from engine/experimental.py without caffe dependencies.
#
#        Raises:
#            ValueError: imageio.imread might throw this exception if image invalid.
#        '''
#        img = imageio.imread(os.path.join(data_root, ex['Filename']))
#        img = img.astype(float) / 255
#        if img.ndim == 2:
#            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
#
#        try:
#            cropped_img, roi_tblr = Resnet101_EBP.crop_image(
#                img, crop_xywh=(ex['XMin'], ex['YMin'], ex['Width'], ex['Height']))
#        except KeyError:
#            cropped_img = img
#        return (cropped_img, ex['SubjectID'], ex['Filename'], ex['SubjectID'])
#
#    @staticmethod
#    def center_crop(img, convert_uint8=True):
#        if isinstance(img, six.string_types):
#            fn = img
#            img = imageio.imread(fn)
#
#        if convert_uint8 and img.dtype != np.uint8:
#            if img.max() <= 1:  # make sure scale is correct ...
#                # make copy to avoid side effects on img parameter
#                img = img.copy() * 255
#            img = img.astype(np.uint8)
#            assert img.max() > 1
#
#        #Pre-Process the Image
#        imgScale = 224
#        minDim = min(img.shape[:2])
#        yx = (np.asarray(img.shape[:2]) - minDim) // 2
#        # img = img[:minDim,:minDim,:] - we want to perform center cropping
#        img = img[yx[0]:yx[0] + minDim,
#                    yx[1]:yx[1] + minDim]
#
#        newSize = (int(img.shape[0]*imgScale/float(minDim)), int(img.shape[1]*imgScale/float(minDim)))
#
#        imgS = skimage.transform.resize(img, (224, 224))
#
#        return imgS
