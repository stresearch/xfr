# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.

import math
import shutil
import subprocess
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

import skimage
from skimage.transform import resize
import skimage.filters

import torch
from xfr.utils import create_net
from xfr.utils import center_crop
from xfr.models.resnet import convert_resnet101v4_image


def print_flush(str, file=sys.stdout, flush=True):
    file.write(str + '\n')
    if flush:
        file.flush()

have_gpu_state = False
can_use_gpu = False
def check_gpu(use_gpu=True):
    # taa: This is pretty massively incorrect.
    # nvidia-smi does not honor CUDA_VISIBLE_DEVICES (it's not a CUDA app,
    # why should it?).
    # Add call to cuda.is_available, which at least covers the case
    # where CUDA_VISIBLE_DEVICES is empty.
    global have_gpu_state
    global can_use_gpu
    if not torch.cuda.is_available():
        have_gpu_state = True
        can_use_gpu = False
        return False
    if have_gpu_state:
        return can_use_gpu
    have_gpu_state = True
    if not use_gpu:
        can_use_gpu = False
    elif 'STR_USE_GPU' in os.environ and len(os.environ['STR_USE_GPU']) > 0:
        try:
            gpu = int(os.environ['STR_USE_GPU'])
            if gpu >= 0:
                can_use_gpu = True
            else:
                can_use_gpu = False
        except ValueError as e:
            can_use_gpu = False
    else:
        import subprocess
        # Ask nvidia-smi for list of available GPUs
        try:
            proc = subprocess.Popen(["nvidia-smi", "--list-gpus"],
                                    stdout=subprocess.PIPE, stdin=subprocess.PIPE)
            gpu_list = []
            if proc:
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    linelist = line.split()
                    if linelist[0] == b'GPU':
                        gpu_id = linelist[1].decode("utf-8")[0]
                        gpu_list.append(gpu_id)
            if len(gpu_list) > 0:
                can_use_gpu = True
        except:
            can_use_gpu = False
    return can_use_gpu


def custom_black_box_fn(probes, gallery):
    """
    User defined black box scoring function.

    This function should compute pairwise similarity scores between all images
    in the 'probes' and 'gallery'. Within this function you should pass image
    data to your black box system and read back in the scores it returns. Note
    the data type requirements for the args and returns below.
    
    To use your custom black box function, instantiate the STRise object by
    setting the 'black_box_fn' parameter to be the name of your custom black box
    function, i.e. STRise(black_box_fn=custom_black_box_fn). A specific example
    illustrating this for the PittPatt system can be found in the following 
    Jupyter notebook: xfr/demo/blackbox_demo_pittpatt.ipynb

    Args:
        probes: A list of numpy arrays of images
        gallery: A list of filepaths to or numpy arrays of images

    Returns:
        A numpy array of size [len(probes) len(gallery)]. Contains
        the similarity score between the ith probe and the jth gallery
        image in the ith row and jth column.
    """
    pass


class STRise:
    def __init__(self,
            probe=None,
            refs=None,
            ref_sids = None,
            potential_gallery=None,
            gallery=None, 
            gallery_size=50,
            black_box=None,
            black_box_fn=None,
            prior_type='mean_ebp',
            mask_type='sparse',
            num_mask_elements=1,
            num_masks=6500,
            mask_scale=12,
            mask_fill_type='blur',
            blur_fill_sigma_percent=4,
            triplet_score_type='cts',
            use_gpu=True,
            device=None,
        ):

        self.mean_ebp_net = None
        self.priors = {'mean_ebp': self.mean_ebp_prior,
                       'uniform': self.uniform_prior}

        self.resnet_net = None
        self.black_boxes = {'resnetv4_pytorch': self.resnet_bb_fn,
                            'resnetv6_pytorch': self.resnet_bb_fn}

        self.mask_types = {'sparse': self.generate_sparse_masks}

        self.mask_fill_types = {'gray': self.mask_fill_gray,
                                'blur': self.mask_fill_blur}

        # Gaussian sigma (as a percent of saliency map size)
        self.blur_fill_sigma_percent = blur_fill_sigma_percent
       
        self.triplet_scoring_fns = {'cts': self.contrastive_triplet_similarity}

        # Validate inputs
        self.use_gpu = check_gpu(use_gpu=use_gpu)
        if not self.use_gpu:
            self.device = torch.device("cpu")
        else:
            if device is None:
                warnings.warn('use_gpu of STRise will be deprecated, use device',
                              PendingDeprecationWarning)
                self.device = torch.device('cuda')
            else:
                self.use_gpu = None
                self.device = device
        
        # Setup probe and reference
        if (probe is not None and refs is not None):
            if isinstance(probe, str) or isinstance(probe, np.ndarray):
                self.probe = center_crop(probe, convert_uint8=True)
            else:
                raise ValueError('Probe must be a filepath to an image or a NumPy array')

            if isinstance(refs, list) or isinstance(refs, np.ndarray) or isinstance(refs, pd.DataFrame):
                self.refs = refs 
            else:
                raise ValueError('Refs must be a list of filepaths, NumPy arrays, or a Pandas dataframe') 
            self.ref_sids = ref_sids
        else:
            raise ValueError('Probe and reference must be specified')
        
        # Setup prior
        if (prior_type is not None):
            if (prior_type not in self.priors):
                raise ValueError('Specified prior "{}" is not supported'.format(prior_type))
            else:
                self.prior_type = prior_type
        else:
            raise ValueError('Prior must be specified')

        # Setup potential gallery
        if (potential_gallery is not None):
            self.potential_gallery = potential_gallery
            if isinstance(potential_gallery, list):
                # List
                self.potential_gallery_size = len(potential_gallery)
            elif isinstance(potential_gallery, np.ndarray):
                # NumPy array
                self.potential_gallery_size = potential_gallery.shape[0]
            elif isinstance(potential_gallery, pd.DataFrame):
                # Pandas dataframe
                self.potential_gallery_size = len(potential_gallery.index)
            else:
                raise TypeError('Potential gallery must be a list of filepaths, NumPy arrays, or a Pandas dataframe')
        else:
            self.potential_gallery = potential_gallery

        # Setup gallery
        if (gallery is not None):
            self.gallery = gallery
            if isinstance(gallery, list):
                # List
                self.gallery_size = len(gallery)
            elif isinstance(gallery, np.ndarray):
                # NumPy array
                self.gallery_size = gallery.shape[0]
            elif isinstance(gallery, pd.DataFrame):
                # Pandas dataframe
                self.gallery_size = len(gallery.index)
            else:
                raise TypeError('Gallery must be a list of filepaths, NumPy arrays, or a Pandas dataframe')
        else:
            self.gallery = gallery
            self.gallery_size = gallery_size

        # Setup black box
        if (black_box):
            self.set_black_box(black_box)
        elif(black_box_fn):
            self.black_box_fn = black_box_fn
        else:
            raise ValueError('Black box name or function must be specified')

        # Setup masks
        if (mask_type is not None):
            if (mask_type not in self.mask_types):
                raise ValueError('Specified mask type "{}" is not supported'.format(mask_type))
            else:
                self.mask_type = mask_type
                self.generate_masks = self.mask_types[mask_type]
        else:
            raise ValueError('Mask type must be specified')

        if (mask_fill_type is not None):
            if (mask_fill_type not in self.mask_fill_types):
                raise ValueError('Specified mask fill type "{}" is not supported'.format(mask_fill_type))
            else:
                self.mask_fill_type = mask_fill_type
                self.apply_masks = self.mask_fill_types[mask_fill_type]
        else:
            raise ValueError('Mask fill type must be specified')
            
        self.num_mask_elements = num_mask_elements
        self.num_masks = num_masks
        self.mask_scale = mask_scale

        # Setup triplet scoring function
        if (triplet_score_type is not None):
            if (triplet_score_type not in self.triplet_scoring_fns):
                raise ValueError('Specified triplet score type "{}" is not supported.'.format(triplet_score_type))
            else:
                self.triplet_score_type = triplet_score_type
                self.triplet_scoring_fn = self.triplet_scoring_fns[triplet_score_type]
        else:
            raise ValueError('Triplet score type must be specified')
    
    def set_probe(self, probe):
        if isinstance(probe, str) or isinstance(probe, np.ndarray):
            self.probe = center_crop(probe, convert_uint8=False)
        else:
            raise ValueError('Probe must be a filepath to an image or a NumPy array')

        # Reset probe gallery scores if necessary
        if (hasattr(self, 'original_probe_gallery_scores')):
            self.original_probe_gallery_scores = None 

    def set_black_box(self, black_box):
        if (black_box not in self.black_boxes):
            raise ValueError('Specified black box "{}" is not supported'.format(black_box))
        else:
            self.black_box = black_box
            self.black_box_fn = self.black_boxes[black_box]

    def mean_ebp_prior(self):
        if (not self.mean_ebp_net):
            self.mean_ebp_net = create_net(
                'resnetv4_pytorch', ebp_version=None, device=self.device)
        
        probe = np.copy(self.probe)
        probe = convert_resnet101v4_image(probe).unsqueeze(0)
        probe = probe.to(self.device)
        
        #Pn = torch.zeros( (1,65359), dtype=torch.float32, device=self.device)
        #Pn[0][0] = 1.0
        Pn = torch.ones( (1,65359), dtype=torch.float32, device=self.device)
        Pn /= 65359.0
        P = self.mean_ebp_net.ebp(probe, Pn)
        self.prior = skimage.transform.resize(P, (224,224), anti_aliasing=True)

    def uniform_prior(self):
        pass

    def generate_sparse_masks(self, random_shift=True, order=1):
        # Assume this divides evenly for now?
        input_size = self.prior.shape[0:2]
        mask_size = tuple(np.ceil(np.divide(input_size, self.mask_scale)).astype(np.int))

        # Rescale prior
        prior_scaled = resize(self.prior, mask_size, anti_aliasing=True)
        
        # Set clipping threshold
        pct = 50.0
        threshold = np.percentile(prior_scaled, pct)
        prior_scaled[prior_scaled<threshold] = 0.0

        # TODO: Get rid of this
        if (self.prior_type == 'uniform'):
            prior_scaled[prior_scaled>0] = 1.0

        # Normalize sum to one
        prior_scaled /= prior_scaled.sum()

        # Generate binary masks with prior probability of selecting 1
        grid = np.ones((self.num_masks, mask_size[0], mask_size[1]))
        for idx in range(self.num_masks):
            rand_idx = np.random.choice(np.arange(prior_scaled.size), self.num_mask_elements, replace=False, p=prior_scaled.ravel())
            grid[idx,...].ravel()[rand_idx] = 0.0

        # Resize binary masks to input_size
        # TODO: Parallelize
        masks = np.empty((self.num_masks, input_size[0], input_size[1]))
        if random_shift:
            # Resize binary masks with random shifts
            for i in range(self.num_masks):
                x = np.random.randint(0, self.mask_scale)
                y = np.random.randint(0, self.mask_scale)
                masks[i,...] = resize(grid[i], (input_size[0]+self.mask_scale, input_size[1]+self.mask_scale), order=order, mode='reflect', anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]
        else:
            masks = resize(grid, (self.num_masks, input_size[0], input_size[1]), order=order, mode='reflect', anti_aliasing=False)
        self.masks = masks
    
    def apply_masks_using_image(self, image):
        masked_images = np.zeros(((self.num_masks,)+image.shape))

        # Blend between probe and image according to masks
        for i, mask in enumerate(self.masks):
            masked_image = mask[...,np.newaxis] * self.probe + (1.0-mask[...,np.newaxis]) * image
            masked_images[i,...] = masked_image
        self.masked_probes = masked_images

    def mask_fill_gray(self):
        fill_image = 0.5 * np.ones(self.probe.shape)
        self.apply_masks_using_image(fill_image)

    def mask_fill_blur(self):
        blurred = skimage.filters.gaussian(
            self.probe,
            self.blur_fill_sigma_percent / 100.0 * max(self.probe.shape),
            multichannel=True,
            preserve_range=True
        )
        self.apply_masks_using_image(blurred)

        # import imageio
        # for i in range(10):
        #     imageio.imwrite('debug%d.png' % i, self.masked_probes[i])
        # import pdb
        # pdb.set_trace()

    def resnet_bb_fn(self, probes, gallery):
        if (not self.resnet_net):
            self.resnet_net = create_net(self.black_box, ebp_version=6)

        # Send gallery images forward through the network
        if isinstance(gallery[0], np.ndarray):
            # If third channel equals 3, assume images need preprocessing
            if gallery[0].shape[2] == 3:
                gallery = [convert_resnet101v4_image(im) for im in gallery]
        gallery_vecs = self.resnet_net.embeddings(gallery)

        # Send probe images forward through the network
        if isinstance(probes[0], np.ndarray):
            # If third channel equals 3, assume images need preprocessing
            if probes[0].shape[2] == 3:
                probes = [convert_resnet101v4_image(im) for im in probes]
        probe_vecs = self.resnet_net.embeddings(probes)

        # Compute score
        L2_similarity = lambda x, y: 1.0 - 0.5 * np.linalg.norm((x/np.linalg.norm(x,axis=1)[:,None])[:,None]-y/np.linalg.norm(y,axis=1)[:,None],axis=2)

        scores = L2_similarity(probe_vecs, gallery_vecs)
        return scores

    def contrastive_triplet_similarity(self):
        ref_scores = self.original_probe_ref_scores - self.masked_probe_ref_scores
        gallery_scores = self.original_probe_gallery_scores - self.masked_probe_gallery_scores
        scores = (ref_scores - gallery_scores).mean(axis=1)
        return scores

    def score_masks(self):
        # Compute original ref black box scores
        self.original_probe_ref_scores = self.black_box_fn([self.probe], self.refs) 

        # Compute original gallery black box scores
        if (not hasattr(self, 'original_probe_gallery_scores') or
                self.original_probe_gallery_scores is None):
            self.original_probe_gallery_scores = self.black_box_fn([self.probe], self.gallery)

        # Compute perturbed ref black box scores
        self.masked_probe_ref_scores = self.black_box_fn(
            self.masked_probes, self.refs)

        # Compute perturbed gallery black box scores
        self.masked_probe_gallery_scores = self.black_box_fn(
            self.masked_probes, self.gallery)

        # Compute mask scores using a triplet function
        self.mask_scores = self.triplet_scoring_fn()

    def combine_masks(self, indices):
        filtered_weights = self.mask_scores[indices]
        filtered_masks = self.masks[indices,...]
        weighted_masks = filtered_weights[...,np.newaxis,np.newaxis] * filtered_masks
        combination = weighted_masks.mean(axis=0)
        return combination

    def compute_saliency_map(self, positive_scores=True, percentile=0):
        # Sort mask scores
        sorted_idx = self.mask_scores.argsort()[::-1]
        pos_sorted_idx = sorted_idx[self.mask_scores[sorted_idx] > 0]
        neg_sorted_idx = sorted_idx[self.mask_scores[sorted_idx] < 0][::-1]

        # try: - most of the time it indicates wrong pair of images used
        # Select indices based on percentile
        if (positive_scores):
            threshold = np.percentile(self.mask_scores[pos_sorted_idx], percentile)
            selected_indices = self.mask_scores >= threshold
            saliency_map = 1.0-self.combine_masks(selected_indices)
        else:
            threshold = np.percentile(-self.mask_scores[neg_sorted_idx], percentile)
            selected_indices = -self.mask_scores >= threshold
            saliency_map = self.combine_masks(selected_indices)-1.0

        saliency_map -= saliency_map.min()
        saliency_map /= saliency_map.max()
        self.saliency_map = saliency_map
        # except IndexError as e:
        #     import pdb
        #     pdb.set_trace()
        #     self.saliency_map = np.zeros(self.masks.shape[1:])
        #     pass


    def evaluate(self):
        curr_step = 1
        num_steps = 5

        #if (self.gallery is None):
        #    num_steps += 1
        #    print_flush('{}/{} Building gallery...'.format(curr_step, num_steps), flush=True)
        #    self.build_gallery()
        #    curr_step += 1

        print_flush('{}/{} Computing prior...'.format(curr_step, num_steps), flush=True)
        self.priors[self.prior_type]()
        curr_step += 1
        
        print_flush('{}/{} Generating masks...'.format(curr_step, num_steps), flush=True)
        self.generate_masks()
        curr_step += 1
        
        print_flush('{}/{} Applying masks...'.format(curr_step, num_steps), flush=True)
        self.apply_masks()
        curr_step += 1
        
        print_flush('{}/{} Scoring masks...'.format(curr_step, num_steps), flush=True)
        self.score_masks()
        curr_step += 1
        
        print_flush('{}/{} Computing saliency map...'.format(curr_step, num_steps), flush=True)
        self.compute_saliency_map()
        
        print_flush('Finished!')

    def plot_gallery(self):
        ncols = 10
        nrows = int(math.ceil(1.0 * self.gallery_size / ncols))
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False, figsize=(ncols,nrows))
        if isinstance(self.gallery, pd.DataFrame):
            for i, id in enumerate(self.gallery.index):
                im = center_crop(self.gallery.at[id,'Filename'], convert_uint8=False)
                axes.flat[i].set_xticks([])
                axes.flat[i].set_yticks([])
                axes.flat[i].xaxis.label.set_visible(False)
                axes.flat[i].yaxis.label.set_visible(False)
                axes.flat[i].imshow(im)
        else:
            for i, im in enumerate(self.gallery):
                axes.flat[i].set_xticks([])
                axes.flat[i].set_yticks([])
                axes.flat[i].xaxis.label.set_visible(False)
                axes.flat[i].yaxis.label.set_visible(False)
                axes.flat[i].imshow(im)

        for ii in range(i+1, nrows*ncols):
            fig.delaxes(axes.flat[ii])

        fig.tight_layout(pad=0, w_pad=0, h_pad=0)
        fig.subplots_adjust(hspace=0, wspace=0)
        plt.show()

    def save_gallery(self, filename):
        ncols = 10
        nrows = int(math.ceil(1.0 * self.gallery_size / ncols))
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False, figsize=(ncols,nrows))
        if isinstance(self.gallery, pd.DataFrame):
            for i, id in enumerate(self.gallery.index):
                im = center_crop(self.gallery.at[id,'Filename'], convert_uint8=False)
                axes.flat[i].set_xticks([])
                axes.flat[i].set_yticks([])
                axes.flat[i].xaxis.label.set_visible(False)
                axes.flat[i].yaxis.label.set_visible(False)
                axes.flat[i].imshow(im)
        else:
            for i, im in enumerate(self.gallery):
                axes.flat[i].set_xticks([])
                axes.flat[i].set_yticks([])
                axes.flat[i].xaxis.label.set_visible(False)
                axes.flat[i].yaxis.label.set_visible(False)
                axes.flat[i].imshow(im)

        for ii in range(i+1, nrows*ncols):
            fig.delaxes(axes.flat[ii])

        fig.tight_layout(pad=0, w_pad=0, h_pad=0)
        fig.subplots_adjust(hspace=0, wspace=0)
        plt.savefig(filename, bbox_inches='tight') 
