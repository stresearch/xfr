# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.



from __future__ import print_function

import os
import errno
import sys

import numpy as np
import scipy
from pprint import pprint
import matplotlib
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import dill as pickle

import skimage
import skimage.morphology
from skimage import transform, filters
from skimage.transform import resize
from scipy.ndimage.filters import median_filter

import pytz
import imageio
import warnings
import random
import re
import pdb
import timeit
import xfr
from xfr import utils


# if 'IJBC_PATH' in os.environ:
#     ijbc_path = os.environ['IJBC_PATH']
# else:
#     ijbc_path = '/proj/janus6/data/Janus_CS4/IJB-C/'
#     warnings.warn('IJBC_PATH environment variable is not set. Using "%s"' %
#                   ijbc_path)
# 
# utils.set_default_print_env('RANDOM_SEED', '1')
# seed = int(os.environ['RANDOM_SEED']) * 1000
# 
# utils.set_default_print_env('NET', 'resnet101-l2')
# utils.set_default_print_env('NUM_SUBJECTS', '10')
# 
# output_dir = os.environ['PWEAVE_OUTPUT_DIR']
# num_subjects = int(os.environ['NUM_SUBJECTS'])
# net = utils.create_net(os.environ['NET'])

def calc_mate_nonmate_dists(net, num_subjects, seed, output_dir, ijbc_path):
    """ Calculate distances of pairs of mates and nonmates using IJBC.

        returns tuple with mate and nonmate distances.
    """
    ijbc_metadata = pd.read_csv(os.path.join(ijbc_path, 'protocols', 'ijbc_metadata.csv'))
    # remove entries with unknown identities
    ijbc_metadata = ijbc_metadata.loc[np.invert(np.isnan(ijbc_metadata['SUBJECT_ID']))]


    ijbc_metadata['Filename'] = [os.path.join(ijbc_path, fn) for fn in
                                ijbc_metadata['FILENAME']]

    ijbc_metadata.rename(inplace=True, columns={
        'SUBJECT_ID': 'SubjectID',
        'FACE_X': 'XMin',
        'FACE_Y': 'YMin',
        'FACE_WIDTH': 'Width',
        'FACE_HEIGHT': 'Height',
    })
    ijbc_metadata = ijbc_metadata.loc[
        np.invert(np.isnan(ijbc_metadata['XMin'].values)) &
        np.invert(np.isnan(ijbc_metadata['YMin'].values)) &
        np.invert(np.isnan(ijbc_metadata['Width'].values)) &
        np.invert(np.isnan(ijbc_metadata['Height'].values))]

    ijbc_metadata = ijbc_metadata.loc[
        ijbc_metadata['Width'] > 100]

    os.makedirs(output_dir, exist_ok=True)

    thresholds = np.arange(1, 0, -1e-3)

    sids = ijbc_metadata['SubjectID'].unique()

    mate_dists = []
    nonmate_dists = []
    # num_nonmates = 254
    num_nonmates = 64

    random.seed(seed)
    groups = ijbc_metadata.groupby(['SubjectID'])
    selected = random.sample(range(len(groups)),num_subjects)
    sampled_groups = [grp for i, grp in enumerate(groups) if i in selected]
    # sampled_groups = random.sample(ijbc_metadata.groupby(['SubjectID']),
    #                                num_subjects)
    total_duration = 0
    num_durations = 0
    seed += 1
    for group_num, (sid, subj_grp) in enumerate(sampled_groups):
        if len(subj_grp) < 2:
            continue
        # try:

        start_time = timeit.default_timer()
        print('Analyzing subject group %d ...' % group_num, end=' ')
        chosen_subjs = subj_grp.sample(2, random_state=seed)
        seed += 1

        chosen_others = ijbc_metadata.loc[ijbc_metadata['SubjectID'] !=
                                        sid].sample(num_nonmates, random_state=seed)

        chosen = pd.concat([chosen_subjs, chosen_others])
        embeddings = net.embeddings(chosen, norm=True)
        # import pdb
        # pdb.set_trace()
        # embeddings = embeddings.cpu().numpy()

        mate_embeds = embeddings[0:len(chosen_subjs)]
        mate_embeds = mate_embeds[:, np.newaxis, :]
        nonmate_embeds = embeddings[np.newaxis, 2:, :]

        mate_dists.append(np.linalg.norm(mate_embeds[0] - mate_embeds[1]))
        nonmate_dists.append(np.linalg.norm(mate_embeds - nonmate_embeds, axis=2))

        seed += 1
        duration = (timeit.default_timer() - start_time)
        total_duration += duration
        num_durations += 1

        # except Exception as err:
        #     warnings.warn('Skipping error %s' % err)
        #     print('Skipping error %s' % err)
        #     pass
        print('finished in %0.1f (avg %0.1f).' % (duration, total_duration / num_durations))

    mate_dists = np.stack(mate_dists)
    nonmate_dists = np.stack(nonmate_dists).reshape((-1,))

    return (mate_dists, nonmate_dists)
    # npfile = os.path.join(
    #     output_dir,
    #     'dists_net=%s_seed=%s.npz' % (net_name, seed))
    # np.savez_compressed(
    #     npfile,
    #     mate_dists=mate_dists,
    #     nonmate_dists=nonmate_dists,
    # )
