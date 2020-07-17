# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.



import os.path
import torch
import PIL
import numpy as np
import pdb
import uuid

import pandas as pd
import skimage
import skimage.morphology

import six
import itertools
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.use('agg')

import xfr
from xfr.utils import create_net
from xfr import utils
from xfr import inpaintgame2_dir
from xfr import inpaintgame_saliencymaps_dir
from xfr.models import blackbox as bb
from xfr.show import create_save_smap
import time

orig_image_pattern = os.path.join(
    inpaintgame2_dir,
    'aligned/{SUBJECT_ID}/{ORIGINAL_BASENAME}/inpainted/{MASK_ID:05d}_truth.png'
)
inpainted_image_pattern = os.path.join(
    inpaintgame2_dir,
    'aligned/{SUBJECT_ID}/{ORIGINAL_BASENAME}/inpainted/{MASK_ID:05d}_out_0.png'
)
mask_pattern = os.path.join(
    inpaintgame2_dir,
    'aligned/{SUBJECT_ID}/{ORIGINAL_BASENAME}/masks/{MASK_ID:05d}.png'
)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_bbox(blackbox_fn, probe_im, mates, nonmates, rise_scale,
                num_mask_elements, mask_fill_type,
                blur_sigma_percent,
                device,
               ):
    def bbox():
        strise = bb.STRise(
            probe=probe_im,
            refs=mates,
            gallery=nonmates,
            mask_scale=rise_scale,
            black_box_fn=blackbox_fn,
            num_mask_elements=num_mask_elements,
            mask_fill_type=mask_fill_type,
            blur_fill_sigma_percent=blur_sigma_percent,
            # invert_masks_before_combine=False,
            device=device,
            )

        # Evaluate black box
        strise.evaluate()

        # Save saliency map
        # plotOverlay(strise.probe, strise.saliency_map, fs_filename)
        return strise.saliency_map
    return bbox


def generate_bb_smaps(bb_score_fn, convert_from_numpy, net_name, img_base, subj_id,
                      mask_id, ebp_ver, overwrite,
                      device,
                      rise_scale=12,
                     ):

    subject_id = subj_id

    cropped_data_dir = os.path.join(
        inpaintgame2_dir,
        'aligned/{}'.format(subject_id)
    )
    multiprobe_data_dir = os.path.join(
        inpaintgame_saliencymaps_dir,
        '{}/subject_ID_{}'.format(
        net_name,
        subject_id))

    inpainting_v2_data = pd.read_csv(os.path.join(
        inpaintgame2_dir,
        'filtered_masks_threshold-{NET}.csv'.format(NET=net_name)))

    inpainting_v2_data = inpainting_v2_data.loc[
        (inpainting_v2_data['MASK_ID'] == int(mask_id)) &
        (inpainting_v2_data['SUBJECT_ID'] == int(subject_id))
    ]

    probe_data = []

    probes = [] # original probes
    mates = []  # original refs
    nonmates = []  # inpainted
    probe_masks = []
    probe_inpaints = []

    for idx, row in inpainting_v2_data.iterrows():
        d = row.to_dict()
        f = orig_image_pattern.format(**d)
        fm = mask_pattern.format(**d)
        finp = inpainted_image_pattern.format(**d)

        if os.path.exists(f):
            if d['TRIPLET_SET'] == 'REF':
                mates.append(f)
            elif d['ORIGINAL_BASENAME'] == img_base:
                probe_data.append(row)
                probes.append(f)
                probe_masks.append(fm)
                probe_inpaints.append(finp)
        else:
            print('Original file %s does not exist!' % f)

        if d['TRIPLET_SET'] == 'REF':
            assert os.path.exists(finp)
            nonmates.append(finp)

    assert len(probes)==1
    im_mates = [im for im in utils.image_loader(mates)]
    im_nonmates = [im for im in utils.image_loader(nonmates)]

    probe_data = pd.DataFrame(probe_data)

    for probe_idx, (
        (probe_im, probe_fn),
        probe_mask_fn,
        (_, probe_row)
    ) in enumerate(zip(
        [ret for ret in utils.image_loader(probes, returnFileName=True)],
        probe_masks,
        probe_data.iterrows()
    )):
        # Construct path if necessary
        extra_dirs = os.path.split(
            os.path.relpath(probe_fn, cropped_data_dir))[0]
        output_dir = os.path.join(multiprobe_data_dir, extra_dirs)

        print('\nOutput: %s\n' % output_dir)

        os.makedirs(output_dir, exist_ok=True)

        img_display = skimage.transform.resize(
            probe_im, (224, 224)
        )
        img_display = (img_display*255).astype(np.uint8)
        img_probe = convert_from_numpy(probe_im)

        mask_im = [im for im in utils.image_loader([probe_mask_fn])][0]

        mask_fill_type = 'blur'
        blur_sigma_percent = 4
        # for num_mask_elements in [1, 2, 4, 8, 16]:
        for num_mask_elements in [2]:
            t0 = time.time()
            fn = 'bbox-rise-%delem_%s=%d_scale_%s' % (
                num_mask_elements,
                mask_fill_type,
                blur_sigma_percent,
                rise_scale,
            )
            create_save_smap(
                fn,
                output_dir, overwrite,
                smap_fn=create_bbox(
                    blackbox_fn=bb_score_fn,
                    probe_im=probe_im,
                    mates=mates,
                    nonmates=nonmates,
                    rise_scale=rise_scale,
                    num_mask_elements=num_mask_elements,
                    mask_fill_type=mask_fill_type,
                    blur_sigma_percent=blur_sigma_percent,
                    device=device,
                ),
                probe_im=probe_im,
                mask_im=mask_im,
                mask_id=mask_id,
                probe_info=probe_row
            )
            t1 = time.time()
            total_n = t1 - t0
            total_min = int(total_n // 60)
            remain_s = total_n % 60
            print('Time: %dm %fs' % (total_min, remain_s))

        # mask_fill_type = 'gray'

        # for num_mask_elements in [2]:
        #     fn = 'bbox-rise-%delem_%s_scale_%s' % (
        #         num_mask_elements,
        #         mask_fill_type,
        #         rise_scale,
        #     )
        #     create_save_smap(
        #         fn,
        #         output_dir, overwrite,
        #         smap_fn=create_bbox(
        #             blackbox_fn=bb_score_fn,
        #             probe_im=probe_im,
        #             mates=mates,
        #             nonmates=nonmates,
        #             rise_scale=rise_scale,
        #             # net_name=net_name,
        #             num_mask_elements=num_mask_elements,
        #             mask_fill_type=mask_fill_type,
        #             blur_sigma_percent=blur_sigma_percent,
        #             device=device,
        #         ),
        #         probe_im=probe_im,
        #         mask_im=mask_im,
        #         mask_id=mask_id,
        #         probe_info=probe_row
        #     )
