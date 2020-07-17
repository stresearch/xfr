#!/usr/bin/env python3
import os

import argparse
import numpy as np
import torch
from collections import OrderedDict

from xfr import xfr_root
from xfr import inpaintgame_saliencymaps_dir
from xfr.inpainting_game.plot_inpainting_game import make_inpaintinggame_plots

from create_wbnet import create_wbnet

human_net_labels_ = OrderedDict([
    ('vgg', 'VGG'),
    ('vggface2_resnet50', 'Resnet-50 (VGG Face2)'),
    ('resnet', 'ResNet'),
    ('resnet_pytorch', 'ResNet (PyTorch)'),
    ('resnetv4_pytorch', 'ResNet v4'),
    ('resnetv6_pytorch', 'ResNet v6'),
    ('resnet+compat-orig', 'ResNet Fix Orig'),
    ('resnet+compat-scale1', 'ResNet Fix V2'),
    ('lightcnn', 'Light CNN'),
])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Script for evaluating inpainting game and plotting results. '
        'Saliency maps must be already be generated. '
        'See eval/generate_inpaintinggame_*.py scripts for generating '
        'saliency maps.'
    )

    parser.add_argument(
        '--method',
        nargs='+', default=[
            'meanEBP_mode=awp_v08_cuda',
            # 'bbox-rise-2elem_blur=4_scale_12',
            'weighted_subtree_triplet_ebp_mode=awp,awp_v08_top32_cuda',
        ],
        dest='METHOD',
        help='saliency methods to compare, based on the slug used in saliency '
        'maps filenames. You will likely want to change this.'
    )
    parser.add_argument(
        '--subjects', nargs='+', dest='SUBJECT_ID',
        type=int,
        default=None,
        help='restrict processing to specific subjects'
    )
    parser.add_argument(
        '--img', dest='IMG_BASENAME',
        nargs='+',
        default=None,
        help='restrict processing to specific IJBC image numbers',
    )
    parser.add_argument(
        '--mask', nargs='+', dest='MASK_ID',
        type=int,
        default=[0, 1, 2, 4, 5, 6, 7, 8, 9],
        help='restrict processing to specific inpainting masks',
        #'{:05}'.format(mask_id)  for mask_id in range(8)
    )
    parser.add_argument(
        '--reprocess',
        action='store_true',
        help='forces reprocessing of all the inpainting game analysis',
    )
    parser.add_argument(
        '--seed', default=None, type=np.uint32,
        help='seed for randomizer',
    )
    parser.add_argument(
        '--output', dest='output_dir',
        default=os.path.join(xfr_root, 'output', 'inpainting_game'),
        help='change output directory, also see --output-subdir',
    )
    parser.add_argument(
        '--output-subdir', default=None,
        dest='output_subdir',
        help='add output subdirectory, useful if generating multiple plots',
    )
    parser.add_argument(
        '--mask-blur-sigma',
        dest='mask_blur_sigma',
        default=None,
        type=float,
        help='soften the edges of thresholded saliency when replacing with '
        'inpainted pixels',
    )
    parser.add_argument(
        '--ignore-missing',
        action='store_true',
        dest='ignore_missing_saliency_maps',
        help='force plotting even if there are missing saliency maps. '
        'Should not be needed.',
    )
    parser.add_argument(
        '--net',
        nargs='+', default=[
            'resnetv4_pytorch',
        ],
        dest='NET',
        help='network to analyze',
    )
    parser.add_argument(
        '--cache-dir',
        dest='cache_dir',
        # default='/disk1/jwilliford/cache',
        required=True,
        help='directory for caching results',
    )
    parser.add_argument(
        '--saliency-dir',
        dest='smap_root',
        default=inpaintgame_saliencymaps_dir,
        help='root directory of saliency maps to get results')

    args = parser.parse_args()
    params = vars(args)
    params['balance_masks'] = True
    params['include_zero_saliency'] = False
    params['threshold_type'] = 'percent-density'
    net_dict = dict()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on {}".format(device))
    else:
        device = torch.device("cpu")
        print("Running on CPU")



    for net_name in params['NET']:
        net_dict[net_name] = create_wbnet(net_name, device=device)

    make_inpaintinggame_plots(
        net_dict=net_dict,
        params=params,
        human_net_labels=human_net_labels_
    )
