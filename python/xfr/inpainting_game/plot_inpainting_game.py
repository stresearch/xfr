# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.



#!/usr/bin/env python3
import os

import glob
import dill as pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
import itertools
from xfr import inpainting_game as inpaintgame
import warnings
import imageio
import skimage
from collections import defaultdict
from xfr import show
from xfr import utils
from xfr.utils import create_net

import pandas as pd
from skimage.transform import resize
import argparse
from collections import OrderedDict
import re
mpl.rcParams.update({'font.size':22})
mpl.use('agg')

import xfr
from xfr import inpaintgame2_dir
from xfr import xfr_root
from xfr import inpaintgame_saliencymaps_dir

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # python 2 backport

regions = OrderedDict([
    ('jaw+cheek',     (['chin', 'jawline', 'cheek'], {
        'faceside': 'symmetric',
        'dilate_radius': 3,
    })),
    ('mouth',    (['lips'], {
        'faceside': 'symmetric',
        'dilate_radius': 9,
    })),
    ('nose',    (['nasal base', 'nasal tip', 'nasal body'], {
        'faceside': 'symmetric',
        'dilate_radius': 9,
    })),
    ('ear',     (['ear'], {
        'faceside': 'symmetric',
        'dilate_radius': 15,
    })),
    ('eye',     (['eye'], {
        'faceside': 'symmetric',
        'dilate_radius': 5,
    })),
    ('eyebrow', (['eyebrow'], {
        'faceside': 'symmetric',
        'dilate_radius': 5,
    })),

    # split by side of face
    ('left-face',  (['eye', 'eyebrow', 'cheek', 'jawline'], {
        'faceside': 'left',
        'dilate_radius': 9,
    })),
    ('right-face', (['eye', 'eyebrow', 'cheek', 'jawline'], {
        'faceside': 'right',
        'dilate_radius': 9,
    })),

    # split left and right face
    ('left-eye',     (['eye'], {
        'faceside': 'left',
        'dilate_radius': 5,
    })),
    ('right-eye',     (['eye'], {
        'faceside': 'right',
        'dilate_radius': 5,
    })),
])
regions_human_labels = {
    0: 'Jaw+Cheek',
    1: 'Mouth',
    2: 'Nose',
    3: 'Ears',  # excluded
    4: 'Eyes',  # replaced with L/R Eye
    5: 'Eyebrows',
    6: 'Left face',
    7: 'Right face',
    8: 'Left eye',
    9: 'Right eye',
    167: 'L/R Face',
    189: 'L/R Eye',
}

def overlap_mask(smap, img, gt_mask, pred_mask):
    rgb = img / max(0.0001, img.max()) * 0.4

    rgb[gt_mask] = np.array([0.6, 0.6, 0.6])
    rgb[pred_mask & gt_mask] = np.array([0,1,0])  # True Pos
    rgb[pred_mask & np.invert(gt_mask)] = np.array([1,0,0])  # False Pos
    return rgb

def make_inpaintinggame_plots(net_dict, params, human_net_labels):
    """ Runs inpainting analysis and generates plots.

        net_dict should be a dictionary of networks. If a network doesn't
        exist in net_dict, code will try to create it with create_net.
    """

    if params['threshold_type'] == 'mass-threshold':
        hgame_thresholds = np.append(np.arange(2e-3, 0, -5e-6), 0)
        hgame_percentile = None
    elif (params['threshold_type'] == 'percent' or
        params['threshold_type'] == 'percent-pixels'):
        params['threshold_type'] = 'percent-pixels'
        hgame_thresholds = None
        hgame_percentile = np.unique(np.sort(np.append(
            100*np.exp(-np.arange(0,15,0.1)),
            [0,100])))
    elif params['threshold_type'] == 'percent-density':  # <-- Standard
        hgame_thresholds = None
        hgame_percentile = np.unique(np.sort(np.append(
            np.arange(0,100,1),
            [0,100])))
    else:
        raise RuntimeError('Unknown threshold type %s '
                        '(try mass-threshold or percent)' %
                        params['threshold_type'])

    # -----------------  Step 1. run analysis
    nonmate_classification, inpainting_v2_data = (
        run_inpaintinggame_analysis(hgame_thresholds, hgame_percentile,
                                    params=params, net_dict=net_dict))

    nonmate_classification['ORIG_MASK_ID'] = nonmate_classification['MASK_ID']
    #  - - - - - - Combined asymetric masks
    for base_net, net_inp in inpainting_v2_data.groupby('NET'):
        counts = {}
        for mask_id, msk_grp in net_inp.groupby(['MASK_ID']):
            counts[mask_id] = len(msk_grp.loc[net_inp['TRIPLET_SET'] == 'PROBE'])

        net_data = nonmate_classification.loc[
            nonmate_classification['NET'] == base_net]
        for left, right in [
            (6, 7),
            (8, 9),
        ]:
            nonmate_classification.loc[
                (nonmate_classification['NET'] == base_net) & (
                    (nonmate_classification['MASK_ID']==left) |
                    (nonmate_classification['MASK_ID']==right)),
                'MASK_ID'
            ] = (100 + 10*left + right)


    # -----------------  Step 2. generate plots
    generate_plots(nonmate_classification, hgame_thresholds, hgame_percentile,
                   params, human_net_labels)

    for base_net, net_inp in inpainting_v2_data.groupby('NET'):
        print('\n%s has %d inpainted triplet examples from %d subjects.' % (
            base_net,
            len(net_inp.loc[net_inp['TRIPLET_SET'] == 'PROBE']),
            # len(net_inp['InpaintingFile'].unique()),
            len(net_inp['SUBJECT_ID'].unique()),
        ))
        for mask_id, msk_grp in net_inp.groupby(['MASK_ID']):
            print('\tmask %d contains %d images from %d subjects.' % (
                mask_id,
                len(msk_grp.loc[net_inp['TRIPLET_SET'] == 'PROBE']),
                # len(msk_grp['InpaintingFile'].unique()),
                len(msk_grp['SUBJECT_ID'].unique()),
            ))

        del msk_grp

    output_dir = params['output_dir']
    if params['output_subdir'] is not None:
        output_subdir = os.path.join(output_dir, params['output_subdir'])
        output_dir = output_subdir


    numTriplets = defaultdict(dict)
    for (base_net, method), method_data in nonmate_classification.groupby(['NET', 'METHOD']):
        print('\n%s + %s has %d inpainted triplet examples from %d subjects.' % (
            base_net,
            method,
            len(method_data),
            len(method_data['SUBJECT_ID'].unique()),
        ))
        for mask_id, msk_grp in method_data.groupby(['MASK_ID']):
            print('\tmask %d contains %d examples from %d subjects.' % (
                mask_id,
                len(msk_grp),
                len(msk_grp['SUBJECT_ID'].unique()),
            ))
            # assume all methods have the same number of triplets for a network
            numTriplets[base_net][mask_id] = len(msk_grp)

        del msk_grp

    for base_net, numTripletsMask in numTriplets.items():
        fig_ds, ax_ds = plt.subplots(1,1, figsize=(6,4), squeeze=True)
        x = np.array([0, 1, 2, 3, 5, 4])  # np.arange(len(numTripletsMask))
        ax_ds.bar(x, numTripletsMask.values())
        ax_ds.set_xticks(x)
        ax_ds.set_xticklabels([regions_human_labels[key] for key in tuple(numTripletsMask.keys())], rotation=50)
        fig_ds.subplots_adjust(top=1, bottom=0.5, left=0.2, right=0.98)
        show.savefig('datasets-stats-%s.png' % base_net, fig_ds, output_dir=output_dir)

    #  - - - - - - Generate maskoverlaps
    smap_root = (
       '%s{SUFFIX_AGGR}/' %
        params['smap_root']
    )
    smap_pattern = os.path.join(
        smap_root,
        '{NET}/subject_ID_{SUBJECT_ID}/{ORIGINAL_BASENAME}/inpainted/{ORIG_MASK_ID:05d}-{METHOD}-saliency.npz'
    )

    orig_pattern = os.path.join(
        inpaintgame2_dir,
        'aligned/{SUBJECT_ID}/{ORIGINAL_BASENAME}/inpainted/{ORIG_MASK_ID:05d}_truth.png'
    )
    mask_pattern = os.path.join(
        inpaintgame2_dir,
        'aligned/{SUBJECT_ID}/{ORIGINAL_BASENAME}/masks/{ORIG_MASK_ID:05d}.png'
    )

    for keys, grp in nonmate_classification.groupby(['NET', 'MASK_ID', 'METHOD']):
        for row_num, (idx, row) in enumerate(grp.iterrows()):
            if row['CLS_AS_TWIN'][-1] != 1:
                # must be correctly classified at the end
                # continue
                stable_correct = len(row['CLS_AS_TWIN']) - 1
                first_correct = len(row['CLS_AS_TWIN']) - 1
            else:
                stable_correct = np.max(np.where(row['CLS_AS_TWIN'] == 0)[0]) + 1
                first_correct = np.min(np.where(row['CLS_AS_TWIN'] == 1)[0])
            if row_num >= 40:
                break

            num_thresh_pixels_stable = (row['TRUE_POS'] +
                                        row['FALSE_POS'])[stable_correct]
            num_thresh_pixels_first = (row['TRUE_POS'] +
                                       row['FALSE_POS'])[first_correct]

            smap = np.load(smap_pattern.format(**row), allow_pickle=True)['saliency_map']
            img = imageio.imread(orig_pattern.format(**row))
            img = utils.center_crop(img, convert_uint8=False)
            gt_mask = imageio.imread(mask_pattern.format(**row))
            gt_mask = gt_mask.astype(bool)

            smap_sorted = np.sort(smap.flat)[::-1]
            threshold_first = smap_sorted[num_thresh_pixels_first]
            # threshold_stable = smap_sorted[num_thresh_pixels_stable]
            threshold_first = smap_sorted[num_thresh_pixels_first]
            # top_smap_stable = smap > threshold_stable

            top_smap_first = smap > threshold_first
            if np.any(np.isnan(smap)):
                import pdb
                pdb.set_trace()

            rgb = overlap_mask(smap, img, gt_mask, top_smap_first)

            fpath = os.path.join(
                output_dir,
                keys[0],
                'mask-%d' % row['MASK_ID'],
                row['METHOD'],
                '%s-%d-idflip.png' % (
                    row['ORIGINAL_BASENAME'].replace('/', '-'),
                    row['ORIG_MASK_ID']
                ))
            Path(os.path.dirname(fpath)).mkdir(exist_ok=True, parents=True)
            imageio.imwrite(fpath, (rgb*255).astype(np.uint8))

            # if params['threshold_type'] == 'percent-density':
            #     inpainted_probe = utils.center_crop(
            #         imageio.imread(row['InpaintingFile']),
            #         convert_uint8=False)
            #     percentiles = []
            #     cls = []
            #     fprs = [0, 0.01, 0.05, 0.10]
            #     for fpr in fprs:
            #         closest = np.argmin(np.abs(fpr * row['NEG'] - row['FALSE_POS']))
            #         percentiles.append(hgame_percentile[closest])
            #         cls.append(row['CLS_AS_TWIN'][closest])

            #     fpr_masks = inpaintgame.create_threshold_masks(
            #         smap,
            #         threshold_method=params['threshold_type'],
            #         percentiles=np.array(percentiles),
            #         thresholds=None,
            #         seed=params['seed'],
            #         include_zero_elements=params['include_zero_saliency'],
            #         blur_sigma=params['mask_blur_sigma'],
            #     )
            #     for fpr_msk, fpr in zip(fpr_masks, fprs):
            #         rgb = overlap_mask(smap, 0*img, gt_mask, fpr_msk)

            #         fpath = os.path.join(
            #             output_dir,
            #             keys[0],
            #             'mask-%d' % row['MASK_ID'],
            #             row['METHOD'],
            #             '%s-%d-at-%dfpr.png' % (
            #                 row['ORIGINAL_BASENAME'].replace('/', '-'),
            #                 row['ORIG_MASK_ID'],
            #                 int(np.round(fpr * 100)),
            #             ))
            #         imageio.imwrite(fpath, (rgb*255).astype(np.uint8))

            #         fpath = os.path.join(
            #             output_dir,
            #             keys[0],
            #             'mask-%d' % row['MASK_ID'],
            #             row['METHOD'],
            #             '%s-%d-at-%dfpr-mask.png' % (
            #                 row['ORIGINAL_BASENAME'].replace('/', '-'),
            #                 row['ORIG_MASK_ID'],
            #                 int(np.round(fpr * 100)),
            #             ))
            #         imageio.imwrite(fpath, (fpr_msk*255).astype(np.uint8))

            #         blended_probe = img.copy()
            #         blended_probe[fpr_msk] = inpainted_probe[fpr_msk]

            #         fpath = os.path.join(
            #             output_dir,
            #             keys[0],
            #             'mask-%d' % row['MASK_ID'],
            #             row['METHOD'],
            #             '%s-%d-at-%dfpr-blended.png' % (
            #                 row['ORIGINAL_BASENAME'].replace('/', '-'),
            #                 row['ORIG_MASK_ID'],
            #                 int(np.round(fpr * 100)),
            #             ))
            #         imageio.imwrite(fpath, (blended_probe).astype(np.uint8))


# output_dir = os.path.join(
#     subj_dir,
#     'sandbox/jwilliford/generated/Note_20191211_InpaintingGame2_Result_Plots')

def skip_combination(net, method, suffix_aggr):
    if net=='vgg' and (
        method == 'tlEBPreluLayer'
        or method == 'tlEBPposReflect'
        or method == 'tlEBPnegReflect'
        or method == 'meanEBP_VGG'  # already included
        # or method == 'ctscEBP'
    ):
        return True
    return False

human_labels_all = [
    ('diffOrigInpaint', 'Groundtruth'),
    ('inpaintingMask', 'Groundtruth - Inpainting Mask'),
    ('diffOrigInpaintEBP', 'Groundtruth via EBP'),
    ('diffOrigInpaintCEBP_median', 'cEBP Groundtruth (median)'),
    ('diffOrigInpaintCEBP_negW', 'cEBP Groundtruth (negW)'),
    ('meanEBP', 'Mean EBP'),
    ('tlEBP', 'Whitebox Triplet EBP'),
    ('tscEBP', 'Whitebox Triplet Similarity Contribution EBP'),
    ('ctscEBP', 'Whitebox Contrastive Triplet Similarity Contribution EBP'),
    ('ctscEBPv3', 'Whitebox Contrastive Triplet Similarity Contribution EBP v3'),
    ('ctscEBPv4',
     'Whitebox Triplet Contribution CEBP v4',
     'Whitebox Triplet Contribution CEBP',
    ),
    ('tsv2EBP', 'Whitebox Triplet Similarity (V2) EBP'),
    ('tsignEBP', 'Whitebox Triplet Sign EBP'),
    ('tsignCEBP', 'Whitebox Triplet Sign Contrastive EBP'),
    ('tsimCEBPv3',
     'Whitebox Triplet Contrastive EBP v3',
     'Whitebox Triplet CEBP',
    ),
    ('tsimPEBPv3',
     'Whitebox Triplet EBP v3',
     'Whitebox Triplet EBP',
    ),
    ('tsimCEBPv3unionSubtract',
     'Whitebox Triplet Contrastive EBP (v3 union-sub)',
    ),
    ('tsimCEBPv3cross',
     'Whitebox Triplet CEBP (v3 cross)',
     'Whitebox Triplet CEBP (cross)',
    ),
    ('tsimCEBPv3.1', 'Whitebox Triplet Similarity Contrastive EBP (v3.1)'),
    ('tlEBPreluLayer', 'Whitebox Triplet EBP (from ReLU)'),
    ('tlEBPnegReflect', 'Whitebox Triplet EBP (neg reflect)'),
    ('tlEBPposReflect', 'Whitebox Triplet EBP (pos reflect)'),
    ('final', 'Blackbox Contrastive Triplet Similarity (2 elem)'),
    ('bbox-rise', 'DISE'),
    ('wb-rise', 'Whitebox PartialConv RISE'),
    # ('pytorch-bb-rise', 'Blackbox RISE (PyTorch Implementation)'),
    ('pytorch-bb-bmay2rise', 'Blackbox Contrastive Triplet'),
    ('bb-bmay2rise', 'Blackbox RISE'),
    ('meanEBP_VGG', 'VGG Mean EBP'),
    ('meanEBP_ResNet', 'ResNet Mean EBP (Caffe)'),
    ('weighted_subtree_triplet_ebp', 'Subtree EBP'),
    ('contrastive_triplet_ebp', 'Contrastive EBP'),
    ('trunc_contrastive_triplet_ebp', 'Truncated cEBP'),
]
def get_base_methods(methods):
    base_methods = [meth.split('_scale_')[0] for meth in methods]
    base_methods = [meth.split('_trunc')[0] for meth in base_methods]
    base_methods = [meth.split('-1elem_')[0] for meth in base_methods]
    base_methods = [meth.split('-2elem_')[0] for meth in base_methods]
    base_methods = [meth.split('-4elem_')[0] for meth in base_methods]
    base_methods = [meth.split('_reluLayer')[0] for meth in base_methods]
    base_methods = [meth.split('_mode')[0] for meth in base_methods]
    base_methods = [meth.split('_v')[0] for meth in base_methods]
    return base_methods

def get_method_labels(methods, lookup):
    base_methods = get_base_methods(methods)
    labels = []
    for base_method in base_methods:
        try:
            labels.append(lookup[base_method])
        except KeyError:
            labels.append(base_method)

    return labels

def backupMethods(method, inpainted_region, orig_imT, inp_imT, error):
    """ If method could not be found, try to see if it is of known type.
        Otherwise, throw passed error.
    """
    if method == 'diffOrigInpaint':
        smap = np.sum(np.abs(orig_imT - inp_imT), axis=0)

        smap_blur = skimage.filters.gaussian(smap, 0.02 * max(smap.shape[:2]))
        smap_blur[smap==0] = 0
        smap = smap_blur
        # smap -= smap.min()
        smap /= smap.sum()
    elif method.split('+')[0] == 'inpaintingMask':
        smap0 = np.mean(np.abs(orig_imT - inp_imT), axis=0)
        smap = inpainted_region.astype(np.float)
        smap = np.maximum(smap, smap0).astype(bool).astype(np.float)

        smap = skimage.filters.gaussian(smap, 0.02 * max(smap.shape[:2]))

        if method == 'inpaintingMask+noise':
            noise = np.random.randn(*smap.shape) * 0.5
            # smap = np.maximum(smap + noise, 0)
            smap = np.abs(smap + noise)

        # smap -= smap.min()
        smap /= smap.sum()
    else:
        raise error
    return smap

human_net_labels_ = OrderedDict([
    ('vgg', 'VGG'),
    ('resnet', 'ResNet'),
    ('resnet_pytorch', 'ResNet (PyTorch)'),
    ('resnetv4_pytorch', 'ResNet v4'),
    ('resnetv6_pytorch', 'ResNet v6'),
    ('resnet+compat-orig', 'ResNet Fix Orig'),
    ('resnet+compat-scale1', 'ResNet Fix V2'),]
)

def tickformatter(x, pos):
    if float.is_integer(x):
        return '%d%%' % x
    else:
        return ''

# Classified As Nonmate
def config_axis(ax, leftmost=True):
    if leftmost:
        ax.set(
            ylabel='Probability Non-mate',
        )
    ax.set(
        xlabel='Top % of Salience Map - Replaced with Inpainted Twin',
        xscale='symlog',
        # yscale='symlog',
    )

    ax.grid(which='both', linestyle=':')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(tickformatter))
    # ax.yaxis.set_major_formatter(plt.FuncFormatter(tickformatter))

def config_axis_iou(ax, leftmost=True):
    if leftmost:
        ax.set(
            ylabel='IOU with Groundtruth',
        )
    ax.set(xlabel='Top % of Salience Map - Replaced with Inpainted Twin',
        xscale='symlog',
        # yscale='symlog',
    )
    ax.grid(which='both', linestyle=':')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(tickformatter))
    # ax.yaxis.set_major_formatter(plt.FuncFormatter(tickformatter))


def avg_class_prob(grp, classifyCol, balance_masks):
    if balance_masks:
        prob_nonmates_mask = dict()
        for mask_id, mask_grp in grp.groupby('MASK_ID'):
            prob_nonmates_mask[mask_id] = np.stack(
                mask_grp[classifyCol].values.tolist()).mean(axis=0)

        cls_as_nonmate = np.stack([*prob_nonmates_mask.values()]).mean(axis=0)
    else:
        cls_as_nonmate = np.stack(grp[classifyCol].values).mean(axis=0)

    # cls_as_nonmate = np.minimum(1, np.maximum(0, cls_as_nonmate))
    return cls_as_nonmate

def plot_roc_curve(ax, grp, hnet, label,
                   method_idx, balance_masks, leftmost=True, classifyCol='CLS_AS_TWIN'):
    cls_as_nonmate = avg_class_prob(grp, classifyCol, balance_masks)

    # cls_as_nonmate = np.minimum(1, np.maximum(0, cls_as_nonmate))

    fpos = np.stack(grp['FALSE_POS'].values).sum(axis=0)
    neg = np.stack(grp['NEG'].values).sum()
    fpr = fpos.astype(np.float64) / neg

    tpos = np.stack(grp['TRUE_POS'].values).sum(axis=0)
    pos = np.stack(grp['POS'].values).sum()
    tpr = tpos.astype(np.float64) / pos

    ax.plot(100*fpr,
            100*tpr,
            color='C%d' % (method_idx + 1),
            label=label,
    )
    if hnet is not None:
        ax.set_title(hnet)

    if leftmost:
        # ax.set(
        #     ylabel='Probability Non-mate',
        # )
        ax.set(
            ylabel='True Positive Rate\n(Sensitivity)',
        )
    ax.set(
        xlabel='False Positive Rate\n(1-Specificity)',
    )

    ax.grid(which='both', linestyle=':')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(tickformatter))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(tickformatter))

    ax.legend() # loc='upper center', bbox_to_anchor=(0.5, -0.1))

def plot_cls_vs_fpr(ax, grp, hnet, label,
                    method_idx,
                    balance_masks,
                    leftmost=True, classifyCol='CLS_AS_TWIN'):
    cls_as_nonmate = avg_class_prob(grp, classifyCol, balance_masks)

    fpos = np.stack(grp['FALSE_POS'].values.tolist()).sum(axis=0)
    neg = np.stack(grp['NEG'].values.tolist()).sum()
    fpr = fpos.astype(np.float64) / neg

    cls_at_fpr = dict()
    for target in [1e-2, 5e-2]:
        fpr_inds = np.argsort(np.abs(fpr - target))[:2]
        closest_fprs = fpr[fpr_inds]
        dists = np.abs(closest_fprs - target)
        # linearly interpolate
        w = 1/(dists+1e-9)
        w = w / np.sum(w)
        cls_at_fpr[target] = np.sum(w*cls_as_nonmate[fpr_inds])

    line, = ax.plot(100*fpr,
            100*cls_as_nonmate,
            color='C%d' % (method_idx + 1),
            # linestyle='--' if scale == 8 else '-',
            label=label,
            # linestyle='-' if ni==0 else '--',
            linewidth=2,
    )
    if hnet is not None:
        ax.set_title(hnet)

    # config_axis(ax, leftmost=leftmost)
    if leftmost:
        # ax.set(
        #     ylabel='Probability Non-mate',
        # )
        ax.set(
            ylabel='Classified as Inpainted Non-mate',
        )
    ax.set(
        xscale='symlog',
        xlabel='False Alarm Rate',
        xlim=(0,100),
    )

    ax.grid(which='both', linestyle=':')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(tickformatter))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(tickformatter))

    ax.legend() # loc='upper center', bbox_to_anchor=(0.5, -0.1))
    return line, cls_at_fpr

def method_label_and_idx(method, methods, human_net_labels, net=None):
    if net is not None:
        try:
            short_hnet = human_net_labels[net].split(' ')[0] + ' '
        except KeyError:
            short_hnet = net
            warnings.warn('Net %s does not have entry in human_net_labels.' %
                          net
                         )
    else:
        short_hnet = ''

    base_methods = get_base_methods(methods)

    human_labels = [(tup[0], tup[1], tup[1] if len(tup)==2 else tup[2])
                    for tup in human_labels_all if 
                    tup[0] in methods or
                    tup[0] in base_methods
                ]
    human_labels_simplified = [
        (key, slabel) for key, _, slabel in human_labels
    ]
    human_labels = [(key, label) for key, label, _ in human_labels]

    human_labels_lookup = OrderedDict(human_labels)
    human_labels_simp_lookup = OrderedDict(human_labels_simplified)

    try:
        method_idx = np.where([
            lbl == method for lbl in methods])[0][0]

        label = get_method_labels([method], human_labels_lookup)[0]
        slabel = get_method_labels([method], human_labels_simp_lookup)[0]

        paren_strs = []
        sparen_strs = []

        try:
            mat = re.search('pytorch-', method)
            if mat is not None:
                paren_strs.append('PyTorch/WIP')
                sparen_strs.append('PyTorch/WIP')
        except AttributeError:
            pass

        scale = None
        nelems = None
        try:
            scale = re.search('_scale_([0-9+]*[0-9])', method).group(1)
            if scale != '12':
                paren_strs.append('Scale ' + scale)
                sparen_strs.append('Scale ' + scale)
        except AttributeError:
            pass
        try:
            nelems = re.search('-([0-9]+)elem', method).group(1)
            if int(nelems) > 1:
                paren_strs.append(nelems + ' Elems')
        except AttributeError:
            pass

        fill = None
        blur_sigma = None
        try:
            mat = re.search('_(blur)=([0-9]+)', method)
            if mat is not None:
                fill = mat.group(1)
                blur_sigma = mat.group(2)
                paren_strs.append('Blur fill')
                # sparen_strs.append('Blur fill')
                if blur_sigma != '4':
                    paren_strs.append('Sigma ' + blur_sigma + '%')
                    # sparen_strs.append('Sigma ' + blur_sigma + '%')

            mat = re.search('_(gray)', method)
            if mat is not None:
                fill = mat.group(1)
                paren_strs.append('Gray fill')
                sparen_strs.append('Gray fill')

            mat = re.search('_(partialconv)_', method)
            if mat is not None:
                fill = mat.group(1)
                paren_strs.append('Partial conv')
                sparen_strs.append('Partial conv')
        except AttributeError:
            pass

        try:
            mat = re.search('_reluLayer', method)
            if mat is not None:
                paren_strs.append('ReLU')
        except AttributeError:
            pass

        try:
            topk = int(re.search('_top([0-9]+)', method).group(1))
            paren_strs.append('Top %d' % topk)
            # sparen_strs.append('Top %d' % topk)
        except AttributeError:
            pass

        try:
            ver = int(re.search('_v([0-9]+)', method).group(1))
            paren_strs.append('V%d' % ver)
            # sparen_strs.append('V%d' % ver)
        except AttributeError:
            pass

        try:
            pct_thresh = int(re.search('_pct([0-9]+)', method).group(1))
            paren_strs.append('Thresh %d%%' % pct_thresh)
            # sparen_strs.append('Threshold %d%%' % pct_thresh)
        except AttributeError:
            pass

        try:
            trunc = re.search('_trunc([0-9]+)', method).group(1)
            paren_strs.append('Trunc ' + trunc + '% Pos')
            sparen_strs.append('Truncated')
        except AttributeError:
            pass

        if len(paren_strs) > 0:
            label = '%s (%s)' % (label,  ', '.join(paren_strs))
        if len(sparen_strs) > 0:
            slabel = '%s (%s)' % (slabel,  ', '.join(sparen_strs))
    # except (IndexError, KeyError) as e:
    except KeyError as e:
        # This is actually handled in get_method_labels now.
        # try:
        #     # method did not exist ...
        #     meth, scale = method.split('_scale_')
        #     method_idx = np.where([
        #         lbl == meth for lbl in human_labels_lookup.keys()])[0][0]
        #     scale_suffix = ' (%s)' % scale
        #     label = human_labels_lookup[meth].format(NET=short_hnet) + scale_suffix
        #     slabel = human_labels_simp_lookup[meth].format(NET=short_hnet) + scale_suffix
        # except ValueError, KeyError:

        # fallback
        label = method
        slabel = method

    assert method_idx < 10  # not supported by cmap used
    return label, method_idx, slabel


def run_inpaintinggame_analysis(hgame_thresholds, hgame_percentile, params,
                                net_dict,
                               ):
    output_dir = params['output_dir']
    cache_dir = params['cache_dir']

    try:
        Path(cache_dir).mkdir(exist_ok=True)
    except PermissionError:
        raise PermissionError('[Errno 13]: permission denied: \'%s\'! Please specify '
            '\'--cache-dir\' parameter!')

    params['SUFFIX_AGGR'] = ['']
    reprocess = params['reprocess']
    seed = params['seed']
    if params['output_subdir'] is not None:
        output_subdir = os.path.join(output_dir, params['output_subdir'])
        output_dir = output_subdir

    # os.environ['PWEAVE_OUTPUT_DIR'] = output_dir
    # os.makedirs(output_dir, exist_ok=True)
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    # csv_output_dir = os.path.join(
    #     output_dir,
    #     'csv-data')
    # Path(csv_output_dir).mkdir(exist_ok=True)

    # include mates and non-mates
    # smap_root = (
    #     '%s{SUFFIX_AGGR}/' %
    #     inpaintgame_saliencymaps_dir
    # )
    smap_root = (
       '%s{SUFFIX_AGGR}/' %
        params['smap_root']
    )

    inpainting_v2_data = dict([(net, pd.read_csv(os.path.join(
        inpaintgame2_dir,
        # 'filtered_masks.csv'))
        'filtered_masks_threshold-{NET}.csv'.format(NET=net))))
        for net in params['NET']]) # 'vgg', 'resnet']])

    # '8/img/240/masks/000.png'
    subj_csv_pattern = os.path.join(
        inpaintgame2_dir,
        'subj-{SUBJECT_ID:d}.csv'
    )
    subj_csv_glob = os.path.join(
        inpaintgame2_dir,
        'subj-*.csv'
    )

    smap_pattern = os.path.join(
        smap_root,
        # '{NET}/subject_ID_{SUBJECT_ID}/{ORIGINAL_BASENAME}/inpainted/{MASK_ID}-{METHOD}-saliency.npz'
        '{NET}/subject_ID_{SUBJECT_ID}/{ORIGINAL_BASENAME}/inpainted/{MASK_ID:05d}-{METHOD}-saliency.npz'
    )

    orig_pattern = os.path.join(
        inpaintgame2_dir,
        'aligned/{SUBJECT_ID}/{ORIGINAL_BASENAME}/inpainted/{MASK_ID:05d}_truth.png'
    )
    mask_pattern = os.path.join(
        inpaintgame2_dir,
        'aligned/{SUBJECT_ID}/{ORIGINAL_BASENAME}/masks/{MASK_ID:05d}.png'
    )

    for net in inpainting_v2_data.keys():
        inpainting_v2_data[net]['OriginalFile'] = [
            orig_pattern.format(**row) for _, row in inpainting_v2_data[net].iterrows()
        ]
        inpainting_v2_data[net]['NET'] = net

    all_subj_data = []
    if params['SUBJECT_ID'] is None:
        for subj_csv in glob.glob(subj_csv_glob):
            all_subj_data.append(
                pd.read_csv(subj_csv))
    else:
        for subject_id in params['SUBJECT_ID']:
            all_subj_data.append(
                pd.read_csv(subj_csv_pattern.format(SUBJECT_ID=subject_id)))

    all_subj_data = pd.concat(all_subj_data)

    if params['SUBJECT_ID'] is None:
        params['SUBJECT_ID'] = all_subj_data['SUBJECT_ID'].unique().tolist() 

    all_subj_data['ORIGINAL_BASENAME'] = [
        os.path.splitext(fn)[0] for fn in all_subj_data['ORIGINAL_FILE'].values]

    def get_base_net(net):
        return net.split('+')[0]

    nonmate_cache_fns = set()
    for net in params['NET']:
        base_net = get_base_net(net)
        net_inp = inpainting_v2_data[base_net]
        print('\n%s has %d inpainted triplet examples from %d subjects.' % (
            base_net,
            len(net_inp.loc[net_inp['TRIPLET_SET'] == 'PROBE']),
            # len(net_inp['InpaintingFile'].unique()),
            len(net_inp['SUBJECT_ID'].unique()),
        ))
        for mask_id, msk_grp in net_inp.groupby(['MASK_ID']):
            print('\tmask %d contains %d images from %d subjects.' % (
                mask_id,
                len(msk_grp.loc[net_inp['TRIPLET_SET'] == 'PROBE']),
                # len(msk_grp['InpaintingFile'].unique()),
                len(msk_grp['SUBJECT_ID'].unique()),
            ))

        del msk_grp

    combined_inpaintings = pd.concat(
        inpainting_v2_data.values(),
        ignore_index=True,
    )
    net_inp = combined_inpaintings
    print('\nCombined nets have %d inpainted triplet examples from %d subjects.' % (
        len(net_inp.loc[net_inp['TRIPLET_SET'] == 'PROBE']),
        # len(net_inp['InpaintingFile'].unique()),
        len(net_inp['SUBJECT_ID'].unique()),
    ))
    for mask_id, msk_grp in net_inp.groupby(['MASK_ID']):
        print('\tmask %d contains %d images from %d subjects.' % (
            mask_id,
            len(msk_grp['InpaintingFile'].unique()),
            len(msk_grp['SUBJECT_ID'].unique()),
        ))

    print(combined_inpaintings.columns)
    inpainting_v2_data = combined_inpaintings

    snet = None
    classified_as_nonmate = []  # using operational threshold
    for net_name in params['NET']:
        base_net = get_base_net(net_name)
        subjs_net_inp = inpainting_v2_data.loc[
            (inpainting_v2_data['NET'] == base_net) &
            (inpainting_v2_data['SUBJECT_ID'].isin(params['SUBJECT_ID']))]

        if params['IMG_BASENAME'] is not None:
            subjs_net_inp = subjs_net_inp.loc[
                (subjs_net_inp['ORIGINAL_BASENAME'].isin(params['IMG_BASENAME'])) |
                (subjs_net_inp['TRIPLET_SET'] == 'REF')
            ]

        for (subject_id, mask_id), ip2grp in subjs_net_inp.groupby(
            ['SUBJECT_ID', 'MASK_ID']
        ):
            if mask_id not in params['MASK_ID']:
                continue

            ebp_version = None  # should need to be set, don't calculate EBP

            if snet is None or snet.net_name != net_name:
                snet = create_net(net_name, ebp_version=ebp_version,
                                  net_dict=net_dict)
                snet.net_name = net_name

            ip2ref = ip2grp.loc[ip2grp['TRIPLET_SET']=='REF']
            mate_embeds = snet.embeddings(
                [os.path.join(inpaintgame2_dir, fn) for fn in ip2ref['OriginalFile']]
            )
            mate_embeds = mate_embeds / np.linalg.norm(mate_embeds, axis=1, keepdims=True)
            original_gal_embed = mate_embeds.mean(axis=0, keepdims=True)
            original_gal_embed = original_gal_embed / np.linalg.norm(original_gal_embed, axis=1, keepdims=True)

            nonmate_embeds = snet.embeddings(
                [os.path.join(inpaintgame2_dir, fn) for fn in ip2ref['InpaintingFile']]
            )
            nonmate_embeds = nonmate_embeds / np.linalg.norm(nonmate_embeds, axis=1, keepdims=True)
            inpaint_gal_embed = nonmate_embeds.mean(axis=0, keepdims=True)
            inpaint_gal_embed = inpaint_gal_embed / np.linalg.norm(inpaint_gal_embed, axis=1, keepdims=True)

            # probes need to be combined with inpainted versions
            ip2probe = ip2grp.loc[ip2grp['TRIPLET_SET']=='PROBE']

            original_imITF = snet.preprocess_loader(
                [os.path.join(inpaintgame2_dir, fn) for fn in ip2probe['OriginalFile']]
            )
            inpaint_imITF = snet.preprocess_loader(
                [os.path.join(inpaintgame2_dir, fn) for fn in ip2probe['InpaintingFile']]
            )

            for (
                (idx, row),
                (orig_im, orig_imT,
                orig_fn),
                (inp_im, inp_imT,
                inp_fn)) in zip(
                ip2probe.iterrows(),
                original_imITF,
                inpaint_imITF,
            ):
                try:
                    orig_imT = orig_imT.cpu().numpy()
                    inp_imT = inp_imT.cpu().numpy()
                except AttributeError:
                    # for caffe
                    pass

                for method, suffix_aggr in itertools.product(
                    params['METHOD'],
                    params['SUFFIX_AGGR'],
                ):
                    # print('Net %s, Subj %d, Mask %d, Method %s' % (net, subject_id, mask_id, method))
                    if skip_combination(net=net, method=method, suffix_aggr=suffix_aggr):
                        continue

                    def calc_twin_cls():
                        d = row.to_dict()
                        d['METHOD'] = method
                        if method == 'meanEBP_VGG':
                            d['NET'] = 'vgg'
                            d['METHOD'] = method.split('_')[0]
                        elif method == 'meanEBP_ResNet':
                            d['NET'] = 'resnet+compat-scale1'
                            d['METHOD'] = method.split('_')[0]

                        d['SUFFIX_AGGR'] = suffix_aggr

                        # if (
                        #     method.startswith('wb-rise') or
                        #     method.startswith('pytorch-')
                        # ):
                        #     # wb-rise doesn't need the ebp fixes
                        #     d['NET'] = base_net
                        # else:
                        #     d['NET'] = net

                        smap_filename = smap_pattern.format(**d)
                        try:
                            if method.split('+')[0] == 'inpaintingMask':
                                raise IOError
                            smap = np.load(smap_filename)['saliency_map']
                        except IOError as e:
                            mask_filename = mask_pattern.format(**d)
                            inpainted_region = imageio.imread(mask_filename)
                            smap = backupMethods(method, inpainted_region,
                                                orig_imT, inp_imT, e)
                            np.savez_compressed(smap_filename, saliency_map=smap)

                        smap = resize(smap, orig_imT.shape[1:], order=0) # interp='nearest')
                        smap /= smap.sum()

                        print(smap.max())
                        nonmate_embed = snet.embeddings([inp_fn])
                        # cls, pg_dist, pr_dist = inpaintgame.inpainting_twin_game_percent_twin_classified(
                        cls, pg_dist, pr_dist = (
                            inpaintgame.classified_as_inpainted_twin(
                                snet,
                                orig_imT,
                                inp_imT,
                                original_gal_embed,
                                inpaint_gal_embed,
                                smap,
                                mask_threshold_method=params['threshold_type'],
                                thresholds=hgame_thresholds,
                                percentiles=hgame_percentile,
                                seed=seed,
                                include_zero_elements=params['include_zero_saliency'],
                                mask_blur_sigma=params['mask_blur_sigma'],
                            ))
                        return cls, pg_dist, pr_dist
                    if params['threshold_type'] == 'percent-density':
                        threshold_method_slug = 'pct-density%d' % (
                            len(hgame_percentile))
                    elif hgame_thresholds is not None:
                        threshold_method_slug='Thresh%d' % len(hgame_thresholds)
                    else:
                        threshold_method_slug='Percentile%d' % len(hgame_percentile),

                    cache_fn = (
                        # 'inpainted-id-hiding-game-twin-cls'
                        'inpainted-id-hiding-game-twin-cls-dists'
                        # operational-thresh{THRESH:0.4f}'
                        '-{SUBJECT_ID}-{MASK_ID}-{ORIGINAL_BASENAME}-0'
                        '-{NET}-{METHOD}{SUFFIX_AGGR}{SEED}-RetProb_'
                        'MskBlur{MASK_BLUR_SIGMA}-'
                        '{THRESHOLDS}{ZERO_SALIENCY_SUFFIX}'
                    ).format(
                            SUBJECT_ID=subject_id,
                            ORIGINAL_BASENAME=row['ORIGINAL_BASENAME'],
                            METHOD=method,
                            NET=net,
                            THRESH=snet.match_threshold,
                            SUFFIX_AGGR=suffix_aggr,
                            SEED='' if seed is None else '-Seed%d' % seed,
                            MASK_ID=mask_id,
                            THRESHOLDS=threshold_method_slug,
                            ZERO_SALIENCY_SUFFIX='ExcludeZeroSaliency' if
                            not params['include_zero_saliency'] else '',
                            MASK_BLUR_SIGMA=params['mask_blur_sigma'],
                        )
                    assert cache_fn not in nonmate_cache_fns, (
                        'Are you displaying the same method multiple times?'
                    )
                    nonmate_cache_fns.add(cache_fn)

                    def calc_saliency_intersect_over_union():
                        d = row.to_dict()
                        d['METHOD'] = method
                        d['SUFFIX_AGGR'] = suffix_aggr

                        # if (
                        #     method.startswith('wb-rise') or
                        #     method.startswith('pytorch-')
                        # ):
                        #     # wb-rise doesn't need the ebp fixes
                        #     d['NET'] = base_net
                        # else:
                        #     d['NET'] = net

                        if method == 'meanEBP_VGG':
                            d['NET'] = 'vgg'
                            d['METHOD'] = method.split('_')[0]
                        elif method == 'meanEBP_ResNet':
                            d['NET'] = 'resnet+compat-scale1'
                            d['METHOD'] = method.split('_')[0]

                        mask_filename = mask_pattern.format(**d)
                        inpainted_region = imageio.imread(mask_filename)
                        try:
                            if method == 'diffOrigInpaint':
                                raise IOError
                            smap_filename = smap_pattern.format(**d)
                            smap = np.load(smap_filename)['saliency_map']
                        except IOError as e:
                            smap = backupMethods(method, inpainted_region,
                                                orig_imT, inp_imT, e)

                        # smap = resize(smap, orig_imT.shape[1:], order=0)
                        # smap = resize(smap, (224, 224), order=0)
                        smap /= smap.sum()

                        # inpainted_region = resize(
                        #     inpainted_region, orig_imT.shape[1:], order=0)

                        nonmate_embed = snet.embeddings([inp_fn])
                        neg = np.sum(inpainted_region == 0)
                        pos = np.sum(inpainted_region != 0)
                        saliency_gt_overlap, fp, tp = inpaintgame.intersect_over_union_thresholded_saliency(
                            smap,
                            inpainted_region,
                            mask_threshold_method=params['threshold_type'],
                            thresholds=hgame_thresholds,
                            percentiles=hgame_percentile,
                            seed=seed,
                            include_zero_elements=params['include_zero_saliency'],
                            return_fpos=True,
                            return_tpos=True,
                        )
                        return saliency_gt_overlap, fp, neg, tp, pos

                    try:
                        cls_twin, pg_dist, pr_dist = utils.cache_npz(
                            cache_fn,
                            # calc_nonmate_cls,
                            calc_twin_cls,
                            reprocess_=reprocess,
                            cache_dir=cache_dir,
                            save_dict_={
                                'hgame_thresholds': hgame_thresholds,
                                'hgame_percentile': hgame_percentile,
                            }
                        )

                        saliency_gt_iou, false_pos, neg, true_pos, pos = utils.cache_npz(
                            ('inpainted-id-hiding-game-saliency-IoU-withcomp-py3'
                            '-{SUBJECT_ID}-{MASK_ID}-{ORIGINAL_BASENAME}-0'
                            '-{NET}-{METHOD}{SUFFIX_AGGR}_{THRESHOLDS}{ZERO_SALIENCY_SUFFIX}').format(
                                SUBJECT_ID=subject_id,
                                ORIGINAL_BASENAME=row['ORIGINAL_BASENAME'],
                                METHOD=method,
                                NET=net,
                                THRESH=snet.match_threshold,
                                SUFFIX_AGGR=suffix_aggr,
                                MASK_ID=mask_id,
                                THRESHOLDS=threshold_method_slug,
                                ZERO_SALIENCY_SUFFIX='ExcludeZeroSaliency' if
                                not params['include_zero_saliency'] else '',
                            ),
                            calc_saliency_intersect_over_union,
                            reprocess_=reprocess,
                            # reprocess_=True,
                            cache_dir=cache_dir,
                            save_dict_={
                                'hgame_thresholds': hgame_thresholds,
                                'hgame_percentile': hgame_percentile,
                            }
                        )
                        classified_as_nonmate.append((
                            net,
                            method,
                            row['ORIGINAL_BASENAME'],
                            inp_fn,
                            suffix_aggr,
                            subject_id,
                            mask_id,
                            np.nan, # cls_nonmate,
                            np.nan, # cls_nonmate[0],
                            np.nan, # cls_nonmate[-1],
                            cls_twin,
                            cls_twin[0],
                            cls_twin[-1],
                            saliency_gt_iou,
                            false_pos,
                            neg,
                            true_pos,
                            pos,
                        ))
                        if (params['include_zero_saliency'] and
                            false_pos[-1] != neg
                        ):
                            raise RuntimeError(
                                'False positive value for last threshold should be'
                                ' the number of negative elements (%d), but is %d.'
                                % (neg, false_pos[-1]))

                    except IOError as e:
                        if not params['ignore_missing_saliency_maps']:
                            raise e

    # for ret in classified_as_nonmate:
    #     ret.get(999999999)

    print('\nNumber of nonmate classification cache files: %d\n' % len(nonmate_cache_fns))

    nonmate_classification = pd.DataFrame(classified_as_nonmate, columns=[
        'NET',
        'METHOD',
        'ORIGINAL_BASENAME',
        'InpaintingFile',
        'SUFFIX_AGGR',
        'SUBJECT_ID',
        'MASK_ID',
        'CLS_AS_NONMATE',
        'Orig_Cls_Nonmate',
        'Twin_Cls_Nonmate',
        'CLS_AS_TWIN',
        'Orig_Cls_Twin',
        'Twin_Cls_Twin',
        'SALIENCY_GT_IOU',
        'FALSE_POS',
        'NEG',
        'TRUE_POS',
        'POS',
    ])
    # merge asymmetric regions
    #   nonmate_classification.loc[
    #       nonmate_classification['MASK_ID']==8, 'MASK_ID'] = 4
    #   nonmate_classification.loc[
    #       nonmate_classification['MASK_ID']==9, 'MASK_ID'] = 4
    #   nonmate_classification.loc[
    #       nonmate_classification['MASK_ID']==6, 'MASK_ID'] = 6
    #   nonmate_classification.loc[
    #       nonmate_classification['MASK_ID']==7, 'MASK_ID'] = 6

    assert (
        len(nonmate_classification['SUBJECT_ID'].unique()) <=
        len(params['SUBJECT_ID'])), (
            'Number of subjects not equal!'
        )

    with open(os.path.join(cache_dir, 'nonmate-cls.pkl'), 'wb') as f:
            pickle.dump(nonmate_classification, f)


    try:
        base_method = params['METHOD'][0]
        nonmate_classification['ComparedToBase'] = 0.0  # large value good
        for keys, grp in nonmate_classification.groupby(
            ['NET',
            'ORIGINAL_BASENAME',
            'MASK_ID',
            'InpaintingFile',
            ]
        ):
            base = grp.loc[grp['METHOD']==base_method].iloc[0]

            for idx, row in grp.iterrows():
                nonmate_classification.loc[idx, 'ComparedToBase'] = (
                    row['CLS_AS_TWIN'].mean() - base['CLS_AS_TWIN'].mean()
                )
        #        for far_value in np.arange(0.10, 0, -0.01):
        #            FAR = np.mean(np.stack(grp['FALSE_POS'] / grp['NEG']), axis=0)
        #            try:
        #                ind_lt = np.where(np.diff(FAR < far_value))[0][0]
        #                # ind_gt = ind_lt + 1
        #                cls_as_twin = np.stack(grp['CLS_AS_TWIN'])[:, ind_lt:ind_lt+2].mean()
        #            except IndexError:
        #                cls_as_twin = np.nan
        #    
        #            print('%s, Mask %d, Method %s%s:\t%0.2f' % (
        #                net, mask_id, method, suffix_aggr, cls_as_twin))
        #            cls_at_far.append(
        #                (net, mask_id, method, suffix_aggr, cls_as_twin, far_value))

        nonbase = nonmate_classification.loc[nonmate_classification['METHOD']!=base_method]
        bad_thresh, good_thresh = (
            np.percentile(nonbase['ComparedToBase'].values, (1, 99)))
        nonbase.sort_values('ComparedToBase', inplace=True)

        print('\nThe below images did particularly worse compared to base method:')
        for idx, row in ( nonbase.loc[nonbase['ComparedToBase']
                                    < bad_thresh].iterrows()):

            print(('    {NET}/subject_ID_{SUBJECT_ID}/{ORIGINAL_BASENAME}/inpainted/'
                '{MASK_ID:05d}-{METHOD}*'
                ' ({ComparedToBase:0.04f})'.format(**row))
            )

        nonbase.sort_values('ComparedToBase', ascending=False, inplace=True)

        print('\nThe below images did particularly better compared to base method:')
        for idx, row in ( nonbase.loc[nonbase['ComparedToBase']
                                    > good_thresh].iterrows()):

            print(('    {NET}/subject_ID_{SUBJECT_ID}/{ORIGINAL_BASENAME}/inpainted/'
                '{MASK_ID:05d}-{METHOD}*'
                ' ({ComparedToBase:0.04f})'.format(**row))
            )
        print('\n')
    except:
        pass

    return nonmate_classification, inpainting_v2_data


def generate_plots(nonmate_classification, hgame_thresholds, hgame_percentile,
                   params,
                   human_net_labels,
                  ):
    output_dir = params['output_dir']
    if params['output_subdir'] is not None:
        output_subdir = os.path.join(output_dir, params['output_subdir'])
        output_dir = output_subdir

    balance_masks = params['balance_masks']
    unequal_method_entries = False
    nonmate_classification_clone = nonmate_classification.copy(deep=True)
    for net, grp0 in nonmate_classification.groupby(['NET']):
        num_entries = None
        for method, grp1 in grp0.groupby(['METHOD']):
            print('%s %s has %d entries.' % (net, method, len(grp1)))
            if num_entries is None:
                num_entries = len(grp1)
            elif num_entries != len(grp1):
                unequal_method_entries = True

    net_indices = OrderedDict(
        [(net, ni) for ni, net in enumerate(params['NET'])]
    )
    cNets = len(net_indices)
    print('#nets=%d' % cNets)

    #   plt.close('all')

    plt_scale = 2

    figL, axesL = plt.subplots(
        1, 1, figsize=(5*plt_scale,2*plt_scale),
        sharex=True, sharey='row', squeeze=False)
    fig4, axes4 = plt.subplots(
        1, cNets, figsize=(6*cNets*plt_scale, 4*plt_scale),
        sharex=True, sharey='row', squeeze=False)
    fig4s, axes4s = plt.subplots(
        1, cNets, figsize=(6*cNets*plt_scale, 4*plt_scale),
        sharex=True, sharey='row', squeeze=False)
    # fig5s, axes5s = plt.subplots(
    #     1, cNets, figsize=(6*cNets*plt_scale,5*plt_scale),
    #     sharex=True, sharey='row', squeeze=False)

    cls_at_fpr_method = dict()
    lines = []
    for (method, suffix_aggr, net), grp in nonmate_classification.groupby(
        ['METHOD', 'SUFFIX_AGGR', 'NET'], sort=False
    ):
        hnet = human_net_labels[net]
        simplified_hnet = human_net_labels[net.split('+')[0]]
        # print('Plotting %s' % hnet)

        label, method_idx, slabel = method_label_and_idx(
            method,
            params['METHOD'],
            human_net_labels,
        )

        ni = net_indices[net]

        plot_cls_vs_fpr(axes4[0, ni], grp, hnet,
                        # method,
                        label,
                        method_idx=method_idx,
                        balance_masks=balance_masks,
                        leftmost=(ni==0))
        # axes4[0,ni].legend(loc='upper center', bbox_to_anchor=(0.5, -0.16))

        plot_cls_vs_fpr(axes4s[0, ni], grp, simplified_hnet,
                        slabel,
                        method_idx=method_idx,
                        balance_masks=balance_masks,
                        leftmost=(ni==0))
        if ni == 0:
            line, cls_at_fpr = plot_cls_vs_fpr(axesL[0, ni], grp, hnet,
                                slabel,
                                method_idx=method_idx,
                                balance_masks=balance_masks,
                                leftmost=(ni==0))
            cls_at_fpr_method[method] = cls_at_fpr
            line.set_linewidth(4)
            lines.append(line)
            axesL[0,ni].legend(loc='center')
            axesL[0,ni].axis('off')

    fig4s.subplots_adjust(top=0.95, bottom=0.1, left=0.15, right=0.96, hspace=0.9, wspace=0.05)
    show.savefig(
        'inpainted_twin_game_%s-net-split_simplified.png' %
        ('balanced-by-mask' if balance_masks else 'unbalanced'),
        fig4s,
        output_dir=output_dir)
    fig4.subplots_adjust(top=0.95, bottom=0.1, left=0.15, right=0.96, hspace=0.9, wspace=0.05)
    show.savefig(
        'inpainted_twin_game_%s-net-split.png' %
        ('balanced-by-mask' if balance_masks else 'unbalanced'),
        fig4,
        output_dir=output_dir)

    for line in lines:
        line.set_visible(False)
    axL = axesL[0,0]
    axL.set_title('')
    show.savefig('inpainted_twin_game_legend.png', figL, output_dir=output_dir,
                transparent=True)

    for ax in axes4s.flat:
            ax.get_legend().remove()
    for ax in axes4.flat:
            ax.get_legend().remove()

    fig4s.subplots_adjust(top=0.95, bottom=0.1, left=0.15, right=0.96, hspace=0.9, wspace=0.05)
    show.savefig(
        'inpainted_twin_game_%s-net-split_simplified-nolegend.png' %
        ('balanced-by-mask' if balance_masks else 'unbalanced'),
        fig4s,
        output_dir=output_dir)
    fig4.subplots_adjust(top=0.95, bottom=0.1, left=0.15, right=0.96, hspace=0.9, wspace=0.05)
    show.savefig(
        'inpainted_twin_game_%s-net-split-nolegend.png' %
        ('balanced-by-mask' if balance_masks else 'unbalanced'),
        fig4,
        output_dir=output_dir)


    # fig5s.subplots_adjust(top=0.95, bottom=0.1, left=0.10, right=0.98, hspace=0.9, wspace=0.05)
    # show.savefig(
    #     'inpainted_twin_game_cls_nonmate_vs_thresh_simplified_%s.png' %
    #     ('balanced-by-mask' if balance_masks else 'unbalanced'),
    #     fig5s,)
    #     output_dir=output_dir)

    plt.close('all')

    cls_at_fpr_method_msk = defaultdict(dict)
    for mask_id, grp0 in nonmate_classification.groupby(
        ['MASK_ID'], sort=False
    ):
        fig4s, axes4s = plt.subplots(
            1, 1, figsize=(8*cNets*plt_scale,1.8*plt_scale),
            sharex=True, sharey='row', squeeze=False)

        for (method, suffix_aggr), grp in grp0.groupby(
            ['METHOD', 'SUFFIX_AGGR'], sort=False
        ):
            label, method_idx, slabel = method_label_and_idx(
                method,
                params['METHOD'],
                human_net_labels,
            )

            ni = 0
            _, cls_at_fpr = plot_cls_vs_fpr(
                axes4s[0, ni], grp, None, slabel,
                method_idx=method_idx,
                balance_masks=balance_masks, leftmost=(ni==0))
            cls_at_fpr_method_msk[method][mask_id] = cls_at_fpr
            # axes4s[0, ni].set_xlim(0, 10)
            axes4s[0, ni].set(ylabel='Classified as\nInpainted\nNon-mate',
            )
            axes4s[0, ni].xaxis.set_major_formatter(
                plt.FuncFormatter(tickformatter))
            axes4s[0, ni].get_legend().remove()

        # fig4s.subplots_adjust(top=0.98, bottom=0.13, left=0.13, right=0.98, hspace=0.9, wspace=0.05)
        fig4s.subplots_adjust(top=0.98, bottom=0.22, left=0.16, right=0.96, hspace=0.9, wspace=0.05)

        # region = regions.keys()[mask_id] # <- doesn't work in python3
        # region = [*regions.keys()][mask_id] # <- doesn't work in python2
        try:
            region = list(regions.keys())[mask_id]
        except IndexError as e:
            if mask_id == 167:
                region = 'left-or-right-face'
            elif mask_id == 189:
                region = 'left-or-right-eye'
            else:
                raise e

        fn = 'inpainted_twin_game_simplified_%s_mask%d_%s.png' % (
            ('balanced-by-mask' if balance_masks else 'unbalanced'),
            mask_id,
            region,
        )

        show.savefig(fn, fig4s, output_dir=output_dir)

        plt.close('all')

    csv_rows = []
    for method, cls_at_fpr_maskid in  cls_at_fpr_method_msk.items():
        nrow = dict()
        print(method)
        print('\tOverall\t%0.9f\t%0.9f' % (
            cls_at_fpr_method[method][1e-2],
            cls_at_fpr_method[method][5e-2],
        ))
        nrow['method'] = method
        nrow['all,far=1e-2'] = cls_at_fpr_method[method][1e-2]
        nrow['all,far=5e-2'] = cls_at_fpr_method[method][5e-2]

        for mask_id in [2, 189, 5]:
        #for mask_id, cls_at_fpr in cls_at_fpr_maskid.items():
            cls_at_fpr = cls_at_fpr_maskid[mask_id]
            print('\t%d\t%0.9f\t%0.9f \t(%s)' % (
                mask_id,
                cls_at_fpr[1e-2],
                cls_at_fpr[5e-2],
                regions_human_labels[mask_id],
            ))
            nrow['%s,far=1e-2' % regions_human_labels[mask_id]] = cls_at_fpr[1e-2]
            nrow['%s,far=5e-2' % regions_human_labels[mask_id]] = cls_at_fpr[5e-2]
        csv_rows.append(nrow)

    csv_results = pd.DataFrame(csv_rows)
    csv_results.to_csv(os.path.join(output_dir, 'results.csv'))

    # for method, cls_at_fpr in cls_at_fpr_method.items():
    #     print('Overall\t%-80s\t%0.3f\t%0.3f' % (
    #         method,
    #         cls_at_fpr[1e-2],
    #         cls_at_fpr[5e-2],
    #     ))

    if unequal_method_entries:
        print('WARNING!!! Unequal method entries! Don\'t trust result!!!!')
