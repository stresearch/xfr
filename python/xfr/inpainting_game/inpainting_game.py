# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
# from subjectness import rise
import skimage
import skimage.filters # needed by python3


def create_threshold_masks(
    saliency_map,
    threshold_method,
    percentiles=None,
    thresholds=None,
    seed=None,
    max_noise=1e-9,
    include_zero_elements=True,
    blur_sigma=None,
):
    """
    Args:
        include_zero_elements: extend mask beyond positive saliency elements.
    """
    np.random.seed(seed)
    if include_zero_elements:
        nonzero_saliency = 1
    else:
        # to prevent expansion of saliency masks past the
        # null elements
        nonzero_saliency = saliency_map != 0

    saliency_map_noise = (
        saliency_map +
        nonzero_saliency *
        np.random.rand(*saliency_map.shape) * max_noise)

    saliency_map_noise = (
        saliency_map_noise / saliency_map_noise.sum()
    )

    if threshold_method == 'percent-density':
        order = np.argsort(saliency_map_noise.flat)
        pdf = saliency_map_noise.flat[order]
        norm_cdf = np.cumsum(pdf)
        saliency_map_noise0 = saliency_map_noise.copy()
        saliency_map_noise.flat[order] = norm_cdf
        # correct for floating point errors
        saliency_map_noise = (
            saliency_map_noise / saliency_map_noise.max()
        )
        thresholds = 1.0 - percentiles.astype(saliency_map_noise.dtype) / 100

        if percentiles[-1] == 100:
            thresholds[-1] = 0
    elif thresholds is None:
        thresholds = np.percentile(saliency_map_noise, 100-percentiles)
        if percentiles[0] == 0:
            thresholds[0] = 1
        if percentiles[-1] == 100:
            thresholds[-1] = 0

    # everything greater than threshold is inpainted
    masks = (saliency_map_noise[np.newaxis, ...] > thresholds[:, np.newaxis, np.newaxis])
    # print(np.mean(masks, axis=(1,2)))

    if blur_sigma is not None and blur_sigma > 0:
        masks = masks.astype(saliency_map.dtype)
        for i in range(masks.shape[0]):
            if percentiles[i] == 100:
                continue

            masks[i] = skimage.filters.gaussian(
                masks[i], blur_sigma * np.min(saliency_map.shape) / 100.0)

    return masks


def classified_as_inpainted_twin(
    snet,
    original_imT,
    inpaint_imT,
    original_gal_embed,
    inpaint_gal_embed,
    saliency_map,
    mask_threshold_method,
    include_zero_elements=True,
    mask_blur_sigma=None,
    percentiles=None,
    thresholds=None,
    seed=None,
    # platts_scale=None,
    binary_classification=True,
    return_transitions=False,
):
    """ Uses saliency map to guide switching original_im to inpaint_im and
        calculates the the hybrid images are classified as the original subject
        or the inpainted subject.

    Args:
        saliency_map:
        percentiles: percent of the image to hide
    """
    masks = create_threshold_masks(
        saliency_map,
        threshold_method=mask_threshold_method,
        percentiles=percentiles,
        thresholds=thresholds, seed=seed,
        include_zero_elements=include_zero_elements,
        blur_sigma=mask_blur_sigma,
    )

    if original_imT.shape[0] == 1 or original_imT.shape[-1] != 3:
        # rgb_masks ... or gray mask if that's what network uses
        # eg. lightcnn
        rgb_masks = masks[:, np.newaxis, ...]
    elif original_imT.shape[0] == 3 or original_imT.shape[-1] != 3:
        rgb_masks = np.repeat(masks[:, np.newaxis, :, :], 3, axis=1)
    else:
        # if original_imT is actually (W,H,3) shaped image
        rgb_masks = np.repeat(masks[:, :, :, np.newaxis], 3, axis=-1)

    original_imT = original_imT.astype(np.float64)
    inpaint_imT = inpaint_imT.astype(np.float64)

    blends = (
        (1.0 - rgb_masks) * original_imT[np.newaxis, :, :, :] +
        rgb_masks * inpaint_imT[np.newaxis, :, :, :])

    # Blur masks slightly
    # hidden_mask = gaussian_filter(hidden_mask.astype(np.float32), sigma=2.0)

    blend_embeds = snet.embeddings(blends)
    blend_embeds = blend_embeds / np.linalg.norm(blend_embeds, axis=1, keepdims=True)

    pr_dist = np.linalg.norm(blend_embeds - original_gal_embed, axis=1)
    pg_dist = np.linalg.norm(blend_embeds - inpaint_gal_embed, axis=1)

    classified_as_twin = pg_dist < pr_dist
    assert not classified_as_twin[0]

    if return_transitions:
        return classified_as_twin, pg_dist, pr_dist, blends, masks
    else:
        return classified_as_twin, pg_dist, pr_dist


def intersect_over_union_thresholded_saliency(
    saliency_map,
    ground_truth,
    mask_threshold_method,
    percentiles=None,
    thresholds=None,
    seed=None,
    include_zero_elements=True,
    return_fpos=False,
    return_tpos=False,
):
    """ Uses saliency map to guide switching original_im to inpaint_im and
        calculates the the hybrid images are classified as the original subject
        or the inpainted subject.

    Args:
        saliency_map:
        percentiles: percent of the image to hide
    """
    ground_truth = ground_truth.astype(np.bool)

    masks = create_threshold_masks(
        saliency_map,
        threshold_method=mask_threshold_method,
        percentiles=percentiles,
        thresholds=thresholds, seed=seed,
        include_zero_elements=include_zero_elements,
    )

    intersection = (ground_truth[np.newaxis, ...]) & masks
    union = (ground_truth[np.newaxis, ...]) | masks
    iou = (
        intersection.sum(axis=(1,2)) /
        (union.sum(axis=(1,2)) + 1e-9)
    )
    ret = (iou,)
    if return_fpos:
        false_pos = np.invert(ground_truth[np.newaxis, ...]) & masks
        # true_neg = np.invert(ground_truth[np.newaxis, ...])
        ret += (np.sum(false_pos, axis=(1,2)),)

    if return_tpos:
        true_pos = (ground_truth[np.newaxis, ...]) & masks
        ret += (np.sum(true_pos, axis=(1,2)),)

    if len(ret)==1:
        return ret[0]  # not tested as of 10/7
    else:
        return ret


def ratio_mate_nonmate_saliency(saliency_mask, probe_mate_region, of_total=True):
    """ Calculate the ratio of the saliency mask in the mated region and ratio
        of the saliency mask in the non-mated region.

        of_total indicates whether to return the ratio from the total image or
        of the specific region.
    """
    smap_refpart = np.nansum(saliency_mask * probe_mate_region)
    smap_nmpart = np.nansum((saliency_mask * (1.0 - probe_mate_region)))
    if not of_total:
        smap_refpart /= np.nansum(probe_mate_region)
        smap_nmpart /= np.nansum(1.0 - probe_mate_region) # non-mate
    else:
        smap_refpart /= probe_mate_region.size
        smap_nmpart /= probe_mate_region.size
    return (smap_refpart, smap_nmpart)

def hidinggame_mated_nonmated_regions(
    smaps,
    probe_mate_region,
    percentiles=np.arange(0,101),
    add_noise=False,
    of_total=True,
):
    """ Calculate the ratios of the saliency that fall into the mated and
        non-mated regions.

    smaps =
    {'final_saliency_map': final_saliency_map, 'saliency_prior': saliency_prior}
    The keys used in smaps will be used in the returned results.
    probe_mate_region: should have 0's in the non-mated region
    """
    # percentile of 1% means that 
    #  only the top 1% of the saliency is evaluated
    percentiles = np.sort(percentiles)

    refparts = dict()
    nmparts = dict()

    for i, (type_, smap) in enumerate(smaps.items()):
        assert np.all(np.invert(np.isnan(smap)))
        if add_noise:
            smap = smap + np.random.rand(*smap.shape) * 1e-9
        thresholds = np.percentile(
            np.append(smap.flatten(), [0.0, 1.0]),
            100.0 - percentiles,
            interpolation='higher')
        refparts[type_] = []
        nmparts[type_] = []
        for thresh, percentile in zip(thresholds, percentiles):
            assert not np.isnan(thresh)
            if not np.isclose(np.mean(smap > thresh)*100, percentile,
                              atol=1e-2):
                raise RuntimeError(
                    'Failed to find accurate threshold for the top %0.1f%% of '
                    'saliency. This indicates that there is portion of '
                    'saliency map with exactly the same value. '
                    'Setting add_noise to True should prevent this.'
                    % percentile
                )
            refpart, nmpart = ratio_mate_nonmate_saliency(
                smap > thresh, probe_mate_region,
                of_total=of_total
            )
            refparts[type_].append(refpart)
            nmparts[type_].append(nmpart)

    ref = dict([(i, np.hstack(part)) for i, part in refparts.items()])
    nm = dict([(i, np.hstack(part)) for i, part in nmparts.items()])

    return ref, nm, percentiles

class HidingGame:
    def __init__(self, saliency_map, image, masking_fn, scoring_fn,
            hide_from_max=True,
            max_hidden_pct=100.0,
            delta_pct=1.0):

        self.saliency_map = saliency_map
        self.image = image
        self.masking_fn = masking_fn
        self.scoring_fn = scoring_fn
        
        self.hide_from_max = hide_from_max
        self.max_hidden_pct = max_hidden_pct 
        self.delta_pct = delta_pct
        self.masks = None
        self.scores = None

    def generate_masks(self):
        num_masks = int(self.max_hidden_pct / self.delta_pct + 1)

        start = 0 
        end = self.max_hidden_pct 
        self.sampled_pcts = np.linspace(start, end, self.num_masks)

        if self.hide_from_max:
            thresholds = np.percentile(self.saliency_map, self.sampled_pcts[::-1])
        else:
            thresholds = np.percentile(self.saliency_map, self.sampled_pcts)

        self.masks = self.saliency_map[...,np.newaxis] < thresholds
        self.masks = self.masks.transpose((2,0,1))
        self.masked_images = self.masking_fn(self.masks, self.image)  

    def evaluate(self):
        if self.masks is None:
            self.generate_masks()

        self.scores = self.scoring_fn(self.masked_images)
        return self.sampled_pcts, self.scores
