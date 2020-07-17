#!/usr/bin/env python3
from __future__ import print_function

import os
import errno
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

import math
import glob
from collections import defaultdict
import argparse
import torch

from create_wbnet import create_wbnet

from xfr import inpaintgame2_dir
from xfr import inpaintgame_saliencymaps_dir

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # python 2 backport


parser = argparse.ArgumentParser(
    'Inpainting game requires determining which inpainted images have '
    'distinctive identities from their originals, for the network of '
    'interest. This script filters the inpainted images to those that have '
    'distinctive identities.'
)

parser.add_argument(
    'NET', nargs='+',
    help='name of networks',
)
# parser.add_argument(
#     '--inpainted-gallery', choices=['avg', 'sep'], default='avg',
#     help='leave as default'
# )
args = parser.parse_args()
params = vars(args)
params['average_nonmates'] = True  # (params['inpainted_gallery'] == 'avg')

margin_ratio = 0.01

plt.rcParams.update({'font.size': 22})

inpainting_pattern_rel = (
    'aligned/{SUBJECT_ID}/{ORIGINAL_BASENAME}/inpainted/'
    '{MASK_ID:05d}_out_0.png')
inpainting_pattern = os.path.join(
    inpaintgame2_dir,
    inpainting_pattern_rel)

original_pattern_rel = (
    # 'aligned/{SUBJECT_ID}/{ORIGINAL_BASENAME}/images/00000.png')
    'aligned/{SUBJECT_ID}/{ORIGINAL_BASENAME}/inpainted/'
    '00000_truth.png')
original_pattern = os.path.join(
    inpaintgame2_dir,
    original_pattern_rel)

all_subj_data = []
mask_separable = defaultdict(lambda: [])  # how separable the original vs mask-inpainted subjects
separability = []

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on {}".format(device))
else:
    device = torch.device("cpu")
    print("Running on CPU")



# for net_name in ['vgg', 'resnet']:
for net_name in params['NET']:
    snet = create_wbnet(net_name, device=device)
    assert snet is not None

    for subj_csv_fn in glob.glob(
        os.path.join(inpaintgame2_dir, 'subj-*.csv')
    ):
        subj_data = pd.read_csv(subj_csv_fn)
        if net_name==params['NET'][0]: # don't save it multiple times
            all_subj_data.append(subj_data)

        subj_data['ORIGINAL_BASENAME'] = [os.path.splitext(fn)[0] for fn in subj_data['ORIGINAL_FILE']]
        probe_fns = []
        mate_fns = []
        for idx, row in subj_data.iterrows():
            d = row.to_dict()
            if d['TRIPLET_SET'] == 'PROBE':
                probe_fns.append(
                    original_pattern.format(**d))
            elif d['TRIPLET_SET'] == 'REF':
                mate_fns.append(
                    original_pattern.format(**d))
        probe_embeds = snet.embeddings(probe_fns, norm=True)

        mate_embeds = snet.embeddings(mate_fns, norm=True)
        mate_embeds = mate_embeds.mean(axis=0, keepdims=True)
        mate_embeds = mate_embeds / np.linalg.norm(mate_embeds, axis=1, keepdims=True)

        probe_embeds = probe_embeds[:, np.newaxis, :]
        mate_embeds = mate_embeds[:, np.newaxis, :]

        pr_dist = np.linalg.norm(probe_embeds - mate_embeds, axis=2)

        # change dimensions to allow for multiple nonmate gallery
        # dimensions [probe, gallery, embedding index]

        # don't allow ear-mask to be included
        # for mask_id in [0,1,2,3,4,5,6,7,8,9]: #range(8):
        # leave out symmetric-eyes, left-face, and right-face
        for mask_id in [0,1,2,3,5,7,6,8,9]:
            nonmate_fns = []
            twin_probe_fns = []
            for idx, row in subj_data.iterrows():
                d = row.to_dict()
                d['MASK_ID'] = mask_id
                if d['TRIPLET_SET'] == 'PROBE':
                    twin_probe_fns.append(
                        inpainting_pattern.format(**d))
                else:
                    # if d['TRIPLET_SET'] == 'REF':
                    nonmate_fns.append(
                        inpainting_pattern.format(**d))

            twin_probe_embeds = snet.embeddings(twin_probe_fns, norm=True)
            twin_probe_embeds = twin_probe_embeds[:, np.newaxis, :]
            # twin_probe_embeds_ = twin_probe_embeds.copy()
            # twin_probe_embeds = twin_probe_embeds / np.linalg.norm(twin_probe_embeds, axis=1, keepdims=True)

            nonmate_embeds = snet.embeddings(nonmate_fns, norm=True)
            nonmate_embeds = nonmate_embeds[np.newaxis, :, :]
            if params['average_nonmates']:
                nonmate_embeds = nonmate_embeds.mean(axis=1, keepdims=True)
                nonmate_embeds = nonmate_embeds / np.linalg.norm(
                    nonmate_embeds, axis=2, keepdims=True)

            # axis [probe, gallery]
            pg_dist = np.linalg.norm(probe_embeds - nonmate_embeds, axis=2)
            min_gal = pg_dist.argmin(axis=1)
            pg_dist = pg_dist.min(axis=1, keepdims=True)

            # mate_correct = pr_dist < pg_dist
            # mate_correct = (
            #     (pr_dist < snet.match_threshold) &
            #     (pg_dist > snet.match_threshold))
            mate_correct = (
                ((pr_dist < pg_dist) &
                 (pr_dist < snet.match_threshold)))

            # diff = pr_dist - pg_dist # more negative == better
            mate_diff = pg_dist - pr_dist # more positive == better

            tpg_dist = np.linalg.norm(twin_probe_embeds - nonmate_embeds, axis=2)
            tpr_dist = np.linalg.norm(twin_probe_embeds - mate_embeds, axis=2)

            tmin_gal = pg_dist.argmin(axis=1)
            tpg_dist = tpg_dist.min(axis=1, keepdims=True)

            # margin = snet.match_threshold * margin_ratio

            # twin_correct = tpr_dist > tpg_dist
            # twin_correct = (
            #     (tpg_dist < snet.match_threshold) &
            #     (tpr_dist > snet.match_threshold))
            twin_correct = (
                (tpg_dist < tpr_dist) &
                (tpr_dist > snet.match_threshold)) # + margin))

            # twin_diff = tpg_dist - tpr_dist  # more negative == better
            twin_diff = tpr_dist - tpg_dist  # more positive == better

            mask_separable[mask_id].append((
                mate_correct, mate_diff,
                twin_correct, twin_diff,
            ))

            # if net_name=='vgg' and mask_id == 2:
            #     msk = subj_data['ORIGINAL_BASENAME'].values == 'img/1406'

            #     dbg_idx = np.where(
            #         ['img/1406' in pf for pf in probe_fns])[0][0]
            #     print(twin_correct[dbg_idx])
            # if net_name=='resnet' and mask_id==0:
            #     dbg_idx = np.where(
            #         ['img/1406' in pf for pf in probe_fns])[0][0]
            #     #np.where(subj_data['ORIGINAL_BASENAME'].values ==
            #     #                'img/1406')[0][0]
            #     # probe_embeds = snet.embeddings(probe_fns, norm=False)[dbg_idx]
            #     probe_embeds = snet.embeddings([probe_fns[dbg_idx]], norm=False)[0]
            #     twin_probe_embeds = snet.embeddings([twin_probe_fns[dbg_idx]], norm=False)[0]

            for i, (idx, row) in enumerate(
                subj_data.loc[subj_data['TRIPLET_SET']=='PROBE'].iterrows()
            ):
                d = row.to_dict()
                d['MASK_ID'] = mask_id
                min_gal_d = d.copy()
                min_gal_d['ORIGINAL_BASENAME'] = (
                    subj_data.iloc[min_gal[i]]['ORIGINAL_BASENAME']
                )
                separability.append((
                    net_name,
                    d['SUBJECT_ID'],
                    d['ORIGINAL_FILE'],
                    d['ORIGINAL_BASENAME'],
                    d['TRIPLET_SET'],
                    mask_id,
                    mate_correct[i], # CorrectlyCls
                    mate_diff[i],
                    twin_correct[i], # TwinCorrectlyCls
                    twin_diff[i],
                    original_pattern_rel.format(**d),
                    inpainting_pattern_rel.format(**d),
                    inpainting_pattern_rel.format(**min_gal_d) if
                        not params['average_nonmates'] else 'average',
                ))

                # if net_name=='vgg' and d['ORIGINAL_BASENAME'] == 'img/3916' and mask_id==2:
                #     import pdb
                #     pdb.set_trace()


all_subj_data = pd.concat(all_subj_data)

separability = pd.DataFrame(separability, columns=[
    'NET',
    'SUBJECT_ID',
    'ORIGINAL_FILE',
    'ORIGINAL_BASENAME',
    'TRIPLET_SET',
    'MASK_ID',
    'CorrectlyCls',
    'OrigTripletSim',
    'TwinCorrectlyCls',
    'TwinTripletSim',
    'OriginalFile',
    'InpaintingFile',
    'BestGalleryFile',
])

# print(separability.loc[(separability['OrigTripletSim'] > -0.01) &
#                        (separability['mask_id']==1)])
# 
# print(separability.loc[(separability['OrigTripletSim'] > -0.01) &
#                        (separability['mask_id']==3)])
# 
# print(separability.loc[(separability['OrigTripletSim'] > -0.01) &
#                        (separability['mask_id']==4)])

def include_masks_by_thresholds(data):
    included_masks = []
    for (subject_id, mask_id), grp in data.groupby(['SUBJECT_ID', 'MASK_ID']):
        some_probes_added = False
        # Add Probes
        for keys, grp2 in grp.groupby(['OriginalFile', 'InpaintingFile']):

            accept = np.all(
                grp2['CorrectlyCls'] &
                grp2['TwinCorrectlyCls'])
            if not accept:
                continue

            some_probes_added = True
            grp2_mean = grp2.iloc[[0]].copy()
            try:
                grp2_mean['OrigTripletSim_resnet'] = (
                    grp2.loc[grp2['NET']=='resnet']['OrigTripletSim'].values[0][0]
                )
                grp2_mean['TwinTripletSim_resnet'] = (
                    grp2.loc[grp2['NET']=='resnet']['TwinTripletSim'].values[0][0]
                )
            except IndexError:
                pass

            try:
                grp2_mean['OrigTripletSim_vgg'] = (
                    grp2.loc[grp2['NET']=='vgg']['OrigTripletSim'].values[0][0]
                )
                grp2_mean['TwinTripletSim_vgg'] = (
                    grp2.loc[grp2['NET']=='vgg']['TwinTripletSim'].values[0][0]
                )
            except IndexError:
                pass
            columns = [
                'SUBJECT_ID',
                'MASK_ID',
                'ORIGINAL_BASENAME',
                'OriginalFile',
                'InpaintingFile',
                'TRIPLET_SET',
                'OrigTripletSim_resnet',
                'TwinTripletSim_resnet',
                'OrigTripletSim_vgg',
                'TwinTripletSim_vgg',
            ]
            columns = [col for col in columns if col in grp2_mean.keys()]

            included_masks.append(grp2_mean[columns])

        if not some_probes_added:
            # import pdb
            # pdb.set_trace()
            continue

        # Add References
        ref_match = all_subj_data.loc[
            (all_subj_data['SUBJECT_ID']==subject_id) &
            (all_subj_data['TRIPLET_SET']=='REF')
        ]
        for (_, original_basename), grp2 in ref_match.groupby(['SUBJECT_ID', 'ORIGINAL_BASENAME']):
            df = grp2.copy()
            df['MASK_ID'] = mask_id
            df['OriginalFile'] = original_pattern_rel.format(
                MASK_ID=mask_id,
                SUBJECT_ID=subject_id,
                ORIGINAL_BASENAME=original_basename,
            )
            df['InpaintingFile'] = inpainting_pattern_rel.format(
                MASK_ID=mask_id,
                SUBJECT_ID=subject_id,
                ORIGINAL_BASENAME=original_basename,
            )
            df['OrigTripletSim_resnet'] = np.nan
            df['TwinTripletSim_resnet'] = np.nan
            df['OrigTripletSim_vgg'] = np.nan
            df['TwinTripletSim_vgg'] = np.nan
            included_masks.append(df.iloc[[0]][columns])

    included_masks = pd.concat(included_masks)
    return included_masks


for net_name, grp0 in separability.groupby(['NET']):
    included_masks = include_masks_by_thresholds(grp0)
    filtered_csv_path = os.path.join(
            inpaintgame2_dir,
            'filtered_masks_threshold-%s.csv' % net_name)
    included_masks.to_csv(
        filtered_csv_path,
        index=False,
    )

print('Percent correct classification (from all masks):')
for mskid, stats in mask_separable.items():
    correct_cls = [cc for cc, diff, tcc, tdiff in stats]
    boundary_diff = [diff for cc, diff, tcc, tdiff in stats]

    tcorrect_cls = [tcc for cc, diff, tcc, tdiff in stats]
    tboundary_diff = [tdiff for cc, diff, tcc, tdiff in stats]

    discriminability = np.mean(np.concatenate(correct_cls + tcorrect_cls, axis=0))
    boundary_diff = np.mean(np.concatenate(boundary_diff + tboundary_diff, axis=0))
    print('  * Mask %d: %.0f%% (%f)' % (mskid, 100 * discriminability, boundary_diff))

print('Written files:')

for net_name, grp0 in separability.groupby(['NET']):
    filtered_csv_path = os.path.join(
            inpaintgame2_dir,
            'filtered_masks_threshold-%s.csv' % net_name)
    print(' * %s' % filtered_csv_path)
