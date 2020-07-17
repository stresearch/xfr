# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.

import os
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import skimage.color
import skimage.filters
import skimage.transform


__all__ = [
    'blend_saliency_map',
    'create_save_smap',
    'plotMaskOverlap',
    'processSaliency',
    'ReturnComparison',
    'savefig',
]

def savefig(fn, fig=None, npdata=None, output_dir=None, transparent=False):
    if output_dir is None:
        output_dir = os.environ['PWEAVE_OUTPUT_DIR']
    fpath = os.path.join(output_dir, fn)

    try:
        os.remove(fpath)  # get around caching issue
    except OSError:
        pass

    if fig is None:
        plt.savefig(fpath, transparent=transparent)
    else:
        fig.savefig(fpath, transparent=transparent)

    if npdata is not None:
        npzfile = os.path.join(
            output_dir,
            os.path.splitext(fn)[0] + '.npz')
        np.savez(npzfile, **npdata)

def blend_saliency_map(image, smap,
                       blur=False,
                       blur_sigma=0.02,
                       scale_factor=1.0,
                       gamma=0.8,
                       ):
    """Blends a saliency map onto an image.

    Creates a visualization of the saliency map overlaid on the image by
    blending them together using the 'jet' colormap. The saliency map is
    automatically normalized to [0,scale_factor]. If necessary, will also resize
    the saliency map to match the extents of the image.

    Args:
        image: A three channel NumPy array of the image with float32 values
               scaled [0,1].
        smap: A single channel NumPy array of the saliency map with float32 values.
        blur: Boolean specifying whether or not to blur the saliency map.
        blur_sigma: Float that sets the standard deviation for Gaussian kernel
                    used to blur the saliency map.
        scale_factor: Float that sets maximum value of the normalized saliency map.
                      This affects how the saliency map gets colored using the
                      colormap. Values less than 1.0 will clip a larger region of
                      the saliency map to red.
        gamma: Float that adjusts the nonlinearity applied to the saliency map.
               This affects how the saliency map is blended with the image.
               Values less than 1.0 will exaggerate less salient features while
               values greater than 1.0 will supress them.

    Returns:
        A NumPy array of the saliency map overlaid on the image. Three channels
        with float32 values normalized [0,1].
    """

    overlay = ReturnComparison([image], [smap],
                  blur=blur,
                  blur_sigma=blur_sigma,
                  scale_factor=scale_factor,
                  gamma=gamma)[0]
    
    return overlay

def ReturnComparison(imgVec, attMaps,
                     suppressMap=None,
                     overlap=True,
                     blur=False,
                     blur_sigma=0.02,
                     scale_factor=1.0,
                     gamma= 0.8,
                     ):
    """ Assumes attMaps[i] corresponds to imgVec[i] 
    """
    if suppressMap is None:
        suppressMap = np.zeros(len(imgVec))
    outMaps = list()
    for i,img in enumerate(imgVec):

        attMap = attMaps[i].copy()
        attMap -= attMap.min() #Normalize
        # attMap = np.minimum(attMap,.0001)
        if attMap.max() > 0:
            attMap /= attMap.max()
            attMap=np.minimum(attMap, scale_factor)
            attMap /= scale_factor 
        else:
            suppressMap[i]=1
        attMap = skimage.transform.resize(
            attMap, (img.shape[:2]), order=3,
            mode='constant')
        if blur:
            attMap = skimage.filters.gaussian(attMap, blur_sigma * max(img.shape[:2])) #Blur
            attMap -= attMap.min()
            attMap /= attMap.max()

        cmap = plt.get_cmap('jet')
        attMapV = cmap(attMap)
        attMapV = np.delete(attMapV, 3, 2) #Colormap
        if overlap:
            attMap = 1 * (1 - attMap ** gamma).reshape(attMap.shape + (1,)) * img + (attMap ** gamma).reshape(attMap.shape + (1,)) * attMapV;
        if suppressMap[i]==0:
            outMaps.append(attMap)
        else:
            outMaps.append(img)
    return outMaps

def processSaliency(img, attMap):
    # attMap = rankNormalizeSaliency(attMap)
    attMap = attMap - attMap.min() #Normalize
    attMap = attMap / (attMap.max() + 1e-9)

    attMap = skimage.transform.resize(attMap, (img.shape[:2]), order=3, mode='constant')
    return attMap

def plotMaskOverlap(img, mask, smap, method, output_dir, mask_id, percent_threshold=None):
    if mask.ndim == 3:
        mask = mask[:,:,0]

    mask = mask.astype(bool)
    smap = smap + np.random.rand(*smap.shape) * 1e-9

    if percent_threshold is None:
        fname='{}/{}-{METHOD}-maskOverlap{SUFFIX}.png'.format(
            output_dir, mask_id,
            METHOD=method,
            SUFFIX='{SUFFIX}',
        )
        threshold = np.percentile(
                np.append(smap.flatten(), [0.0, 1.0]),
                100 - mask.mean()*100,
                interpolation='higher')
    else:
        fname='{}/{}-{METHOD}-maskOverlap-thresh={thresh}{SUFFIX}.png'.format(
            output_dir, mask_id,
            METHOD=method,
            thresh=percent_threshold,
            SUFFIX='{SUFFIX}',
        )
        threshold = np.percentile(
            np.append(smap.flatten(), [0.0, 1.0]),
            100 - percent_threshold,
            interpolation='higher')
        # if 'PWEAVE_GEN' not in os.environ:
        #     import pdb
        #     pdb.set_trace()

    top_smap = smap > threshold
    # rgb = np.zeros_like(img)
    img = img / 255.0
    rgb = img * 0.4
    rgb[top_smap & mask] = np.array([0,1,0])
    rgb[top_smap & np.invert(mask)] = np.array([1,0,0])
    rgb[np.invert(top_smap) & mask] = np.array([0.6, 0.6, 0.6])
    imageio.imwrite(fname.format(SUFFIX=''), (rgb*255).astype(np.uint8))

    #  orig_triplet_sim = probe_row['OrigTripletSim_%s' % base_net]
    #  twin_triplet_sim = probe_row['TwinTripletSim_%s' % base_net]

    #  #fname='{}/{}-{METHOD}-maskOverlap-SimScores.png'.format(output_dir, mask_id, METHOD=method)
    #  fig, ax = plt.subplots(figsize=(3,3))
    #  ax.imshow(rgb)
    #  ax.text(0.2, 0.1, '%0.2f' % orig_triplet_sim, horizontalalignment='center',
    #          verticalalignment='center', transform=ax.transAxes, color='white',
    #         fontsize=18)
    #  ax.text(0.8, 0.1, '%0.2f' % twin_triplet_sim, horizontalalignment='center',
    #          verticalalignment='center', transform=ax.transAxes, color='white',
    #         fontsize=18)
    #  ax.set_axis_off()
    #  fig.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    #  fig.savefig(fname.format(SUFFIX='-SimScores'))

def create_save_smap(method, output_dir, overwrite, smap_fn, mask_id,
                     probe_im, probe_info, mask_im,
                    ):
    overlay_filename = '{}/{}-{}-saliency-overlay.png'.format(
        output_dir,
        mask_id,
        method,
    )
    npz_filename = '{}/{}-{}-saliency.npz'.format(
        output_dir, mask_id,
        method,
    )
    if (overwrite or
        not (os.path.exists(overlay_filename) and os.path.exists(npz_filename))
    ):
        smap = smap_fn()
        smap = smap.astype(np.float32)

        smap -= smap.min()
        smap /= smap.sum()
        # Save saliency map
        smap = processSaliency(probe_im, smap)
        overlay = blend_saliency_map(probe_im, smap)
        imageio.imwrite(overlay_filename.format(SUFFIX=''), (overlay*255).astype(np.uint8))

        np.savez_compressed(npz_filename, saliency_map=smap)
        print('Created:\n %s\n' % overlay_filename)

