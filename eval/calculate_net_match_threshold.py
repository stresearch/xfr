#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
from sklearn.linear_model import LogisticRegression as LR

from xfr import inpaintgame2_dir
from xfr import xfr_root
from xfr import inpaintgame_saliencymaps_dir

if 'IJBC_PATH' in os.environ:
    ijbc_path = os.environ['IJBC_PATH']
else:
    ijbc_path = '/proj/janus3/data/Janus_CS4/IJB-C/'
    warnings.warn('IJBC_PATH environment variable is not set. Using "%s"' %
                  ijbc_path)

# ============== Parse arguments
parser = argparse.ArgumentParser(
    'Example script to calculate match threshold, used for STR ResNet v4 and '
    'v6 models.'
    'Needed to filter inpainting images for networks of the inpainting game, '
    'if match threshold is not otherwise set. '
    'Requires calculate_subject_dists_inpaintinggame.py to be ran beforehand.'
)

# parser.add_argument(
#     '--subj-csv', nargs='+', dest='SUBJ_CSV',
#     default=glob.glob(os.path.join(inpaintgame2_dir, 'subj-*.csv'))
# )
# parser.add_argument(
#     '--subjects', nargs='+', dest='SUBJECT_ID',
#     default=[8, 30, 350],
# )
# parser.add_argument(
#     '--mask', nargs='+', dest='MASK_ID',
#     default=[
#         '{:05}'.format(mask_id)  for mask_id in range(8)
#     ]
# )
parser.add_argument(
    'NET', nargs='+',
    default=['resnetv4_pytorch'],
)

args = parser.parse_args()
params = vars(args)

for net in params['NET']:
    in_dir = os.path.join(
        xfr_root,
        'output',
        'ROC_Curve_Analysis_Inpainting_Game/Net=%s' % net)
    npz_files = glob.glob(os.path.join(in_dir, '*.npz'))
    if len(npz_files) == 0:
        print('Skipping net %s. Could not find any files in %s.' % (
            net,
            in_dir,
        ))
        print('Did you run eval/calculate_subject_dists_inpaintinggame.py for '
              'this net?')
        continue
    mate_dists = []
    nonmate_dists = []
    for npzfile in npz_files:
        data = np.load(npzfile)
        mate_dists.append(data['mate_dists'])
        nonmate_dists.append(data['nonmate_dists'])

    mate_dists = np.concatenate(mate_dists)
    nonmate_dists = np.concatenate(nonmate_dists)

    thresholds = np.concatenate([mate_dists, nonmate_dists])
    thresholds.sort()
    thresholds = np.insert(thresholds, 0, 0)  # add 0 threshold
    thresholds = np.around(thresholds, 4)
    thresholds = np.unique(thresholds)

    fp = np.sum(nonmate_dists[:, np.newaxis] <= thresholds[np.newaxis, :], axis=0)
    fpr = fp.astype(np.float) / len(nonmate_dists)
    chosen_index = np.argmin(abs(fpr - 1e-4))
    thresh = thresholds[chosen_index]

    tp = np.sum(mate_dists[:, np.newaxis] <= thresholds[np.newaxis, :], axis=0)
    tpr = tp.astype(np.float) / len(mate_dists)

    lr = LR(fit_intercept=False)
    dists = np.concatenate([mate_dists, nonmate_dists]) - thresh

    # y = classification where 1 is nonmate
    y = np.ones(dists.shape, dtype=np.int)
    y[:len(mate_dists)] = 0
    lr.fit(dists[:, np.newaxis], y)

    # Prob = 1 / (1 + exp(- alpha * dist))
    alpha = lr.coef_[0,0]

    print("\nNet %s threshold=%f, \tplatt's scaling=%f" % (
        net, thresh,
        alpha, # lr.intercept_
    ))
    print("\nTo use, set the Whitebox object 'wb' parameters:\n")
    print("\twb.match_threshold = %f" % thresh)
    print("\twb.platts_scaling = %f\n" % alpha)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set(xlabel='FMR', ylabel='TMR')
    fig.savefig(os.path.join(in_dir, 'roc.png'))

    del thresholds
    del tpr
    del tp
    del fpr
    del fp
