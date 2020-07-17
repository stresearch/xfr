# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.

import pandas as pd
import imageio
import numpy as np
import six
import os
from pathlib import Path
import torch
from torch.autograd import Variable
import random
import shutil
import skimage.transform
import xfr
from xfr.models import whitebox
from xfr.models import resnet
from xfr import xfr_root

__all__ = [
    'denormalize',
    'create_net',
    'init_weights',
    'init_random_seed',
    'init_model',
    'save_model',
    'copy_files',
    'cache_npz',
    'set_default_print_env',
    'image_loader',
    'crop_image',
    'crop_example_no_name',
    'center_crop',
    'convert_from_numpy',
    'iterate_param_sets',
    'prune_unneeded_exports',
]

def image_loader(images, returnImageIndex=False, returnFileName=False, repeats=1):
    """ Iterates over tuple: (displayable image, fn)

        This function is a modification of preprocess_loader from
        engine/experimental.py which has been modified to remove caffe
        dependencies.

        fn will be None if images is a numpy array

        Args:
            repeats: number of times each image is returned. If it is set to a
                number higher than 1, then the repeat number is added to return
                tuple. Used for channel-wise EBP.

        Raises:
            ValueError: might be raised when attempting to load image
    """
    if isinstance(images, pd.DataFrame):
        for i, (_, imginfo) in enumerate(images.iterrows()):
            img, sid, fn, _ = (
                crop_example_no_name(imginfo))
            assert img.max() <= 1.0 and img.min() >= 0.0

            # img = (img * 255).astype(np.uint8)
            # img = PIL.Image.fromarray(img).convert('RGB')

            ret = [img]

            if returnImageIndex:
                ret.append(i)
            if returnFileName:
                ret.append(fn)

            if repeats == 1:
                if len(ret) == 1:
                    yield ret[0]
                else:
                    yield tuple(ret)
            else:
                for repeat_num in range(repeats):
                    yield tuple(ret + [repeat_num])
    else:
        for i, img in enumerate(images):
            if isinstance(img, np.ndarray):
                assert img.ndim == 3 and img.shape[2] == 3
                fn = None
                cropped = img
            elif isinstance(img, six.string_types):
                fn = img
                img = imageio.imread(fn)
                img = img.astype(float) / 255
                cropped = center_crop(img, convert_uint8=False)
            else:
                raise NotImplementedError('Unhandled type %s' %
                                            type(img))

            ret = [cropped]

            if returnImageIndex:
                ret.append(i)
            if returnFileName:
                ret.append(fn)

            if repeats == 1:
                if len(ret) == 1:
                    yield ret[0]
                else:
                    yield tuple(ret)
            else:
                for repeat_num in range(repeats):
                    yield tuple(ret + [repeat_num])

def crop_image(img, crop_xywh=None, crop_tblr=None, roi_method='expand'):
    """ Copy of max_tracker function, without caffe and related dependencies.
    """
    if crop_xywh is not None:
        x = int(round(crop_xywh[0]))
        y = int(round(crop_xywh[1]))
        w = int(round(crop_xywh[2]))
        h = int(round(crop_xywh[3]))
    if crop_tblr is not None:
        y = int(round(crop_tblr[0]))
        y2 = int(round(crop_tblr[1]))
        x = int(round(crop_tblr[2]))
        x2 = int(round(crop_tblr[3]))
        w = y2 - y
        h = x2 - x

    center_x = x + w // 2
    center_y = y + h // 2

    if roi_method == 'constrict':
        cropDim = int(min(w, h))
    elif roi_method == 'constrict80':
        cropDim = int(min(w, h) * 0.8)
    elif roi_method == 'constrict50':
        cropDim = int(min(w, h) * 0.5)
    else:
        assert roi_method == 'expand'
        # the crop dimensions can't be larger than the image
        cropDim = min(
                max(w, h),
                min(img.shape[0], img.shape[1]))
    top = max(0, center_y - cropDim // 2)
    left = max(0, center_x - cropDim // 2)
    # make bottom and right relative to (potentially shifted) top and left
    bottom = min(img.shape[0], top + cropDim)
    right = min(img.shape[1], left + cropDim)
    # if hit bottom or right border, shift top or left
    top = max(0, min(top, bottom - cropDim))
    left = max(0, min(left, right - cropDim))

    cropped = img[top : bottom,
            left : right,
            :]
    # Image.fromarray((cropped*255).astype(np.uint8)).show()
    return (cropped, (top, bottom, left, right))

def crop_example_no_name(ex, data_root=''):
    '''
    Copy of function from engine/experimental.py without caffe dependencies.

    Raises:
        ValueError: imageio.imread might throw this exception if image invalid.
    '''
    img = imageio.imread(os.path.join(data_root, ex['Filename']))
    img = img.astype(float) / 255
    if img.ndim == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    try:
        cropped_img, roi_tblr = crop_image(
            img, crop_xywh=(ex['XMin'], ex['YMin'], ex['Width'], ex['Height']))
    except KeyError:
        cropped_img = img
    return (cropped_img, ex['SubjectID'], ex['Filename'], ex['SubjectID'])

def center_crop(img, convert_uint8=True):
    if isinstance(img, six.string_types):
        fn = img
        img = imageio.imread(fn)

    if convert_uint8 and img.dtype != np.uint8:
        if img.max() <= 1:  # make sure scale is correct ...
            # make copy to avoid side effects on img parameter
            img = img.copy() * 255
        img = img.astype(np.uint8)
        assert img.max() > 1

    #Pre-Process the Image
    imgScale = 224
    minDim = min(img.shape[:2])
    yx = (np.asarray(img.shape[:2]) - minDim) // 2
    # img = img[:minDim,:minDim,:] - we want to perform center cropping
    img = img[yx[0]:yx[0] + minDim,
                yx[1]:yx[1] + minDim]

    newSize = (int(img.shape[0]*imgScale/float(minDim)),
               int(img.shape[1]*imgScale/float(minDim)))

    imgS = skimage.transform.resize(img, (224, 224), preserve_range=True)
    imgS = imgS.astype(img.dtype)

    return imgS

def cache_npz(fn, fun, cache_dir, *args, **kwargs):
    """ Adds caching to a function call.

        Args:
            fn: filename for cache
            fun: the function to call if results are not cached
            reprocess_: reprocess, do not load
            cache_dir_: directory for the caching
            save_dict_: optional dictionary of other numpy arrays. If passed,
                the cached save_dict entries must be equal.
            *args, **kwargs: additional parameters passed to fun
    """
    Path(cache_dir).mkdir(exist_ok=True)  # remkdir
    fn = fn.replace('/','_')
    print("Cache:  %s" % fn)

    fpath = os.path.join(cache_dir, fn + '.npz')
    try:
        if 'reprocess_' in kwargs and kwargs['reprocess_']:
            raise IOError  # force reprocessing

        npdata = np.load(fpath, allow_pickle=True)

        if 'save_dict_' in kwargs:
            save_dict = kwargs['save_dict_']
            for key, val in save_dict.items():
                if not np.array_equal(npdata[key], val):
                    import pdb
                    pdb.set_trace()
                    raise IOError  # force reprocessing

        if False:  # debugging
            if 'reprocess_' in kwargs:
                del kwargs['reprocess_']
            if 'save_dict_' in kwargs:
                del kwargs['save_dict_']
            ret = fun(*args, **kwargs)
            assert np.allclose(npdata['arr_0'], ret)

        ret = npdata['arr_0']
        return ret
    except (IOError, KeyError):
        if 'reprocess_' in kwargs:
            del kwargs['reprocess_']

        save_dict = dict()
        if 'save_dict_' in kwargs:
            save_dict = kwargs['save_dict_']
            del kwargs['save_dict_']

        ret = fun(*args, **kwargs)

        save_dict['arr_0'] = ret
        # np.savez_compressed(fpath, ret)
        np.savez(fpath, **save_dict)

        # if True:  # debugging
        #     npdata = np.load(fpath)
        #     assert np.array_equal(npdata['arr_0'], ret)
        return ret

def set_default_print_env(var, default=None):
    if default is not None and var not in os.environ:
        os.environ[var] = default

    if var in os.environ:
        print('%s=%s' % (var, os.environ[var]))
        return os.environ[var]
    else:
        print('%s=<not set>' % (var))
        return None

def iterate_param_sets(params, params_export):
    """ Iterates over all the combination of params[params_export]
    """
    for k in params_export:
        # allow param to be tuple where first is a lambda function, where if it
        # returns true, the second part of tuple should be used as the
        # parameter.
        try:
            if k[0](params):
                k = k[1]
            else:
                continue
        except TypeError:
            pass

        if k not in params or params[k] is None:
            continue

        if len(params[k]) > 1:
            for val in params[k]:
                pams = params.copy()
                pams[k] = [val]
                # yield from iterate_param_sets(pams, params_export)
                for it in iterate_param_sets(pams, params_export):
                    yield it
            return
    yield params

def prune_unneeded_exports(params_export, params):
    pruned = []
    for k in params_export:
        # allow param to be tuple where first is a lambda function, where if it
        # returns true, the second part of tuple should be used as the
        # parameter.
        try:
            if k[0](params):
                k = k[1]
            else:
                continue
        except TypeError:
            pass

        if k not in params:
            continue

        pruned.append(k)

    return pruned

def freeze_batchnorm_stats(net):
    def set_bn_eval(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    net.apply(set_bn_eval)

def unfreeze_batchnorm_stats(net):
    def set_bn_eval(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train()
    net.apply(set_bn_eval)

def load_val_batches(val_data):
    val_batches = []
    val_labels = []
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=64)
    for batch, labels in val_loader:
        batch = Variable(batch.cuda(), volatile=True)
        val_batches.append(batch)
        val_labels.append(labels)
    all_labels = torch.cat(val_labels)
    all_labels = Variable(all_labels.cuda(), volatile=True)
    return val_batches, all_labels

def run_validation(net, val_batches, val_labels, val_loss_fn):
    all_scores = []
    for imgs in val_batches:
        _, scores = net(imgs)
        all_scores.append(scores)
        all_scores = torch.cat(all_scores)
        return val_loss_fn(all_scores, val_labels).data[0]

def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)

def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)

def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    net.apply(init_weights)

    # restore model weights
    if restore is not None:  # and os.path.exists(restore):
        # It's okay if loaded_state has additional variables, but
        # all of the variables in net should be loaded
        loaded_state = {
            k: val for k, val in torch.load(restore).items()
            if k in net.state_dict().keys()
        }
        net.load_state_dict(loaded_state)
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))
    # elif restore is not None:
    #     print("ERROR: Failed to restore model weights from %s" % restore)
    #     print("Using random weights!")
    #     raise FileNotFoundError()
    #     import pdb
    #     pdb.set_trace()
    else:
        print("New model initialized with random weights.")

    #   # check if cuda is available
    #   if torch.cuda.is_available():
    #       cudnn.benchmark = True
    #       net.cuda()

    return net

def create_net(net_name, ebp_version=6, device=None, net_dict=None):
    """ Helper function for included networks.

        Creates network if it doesn't already exist in net_dict.
    """
    # python3 pytorch code
    if device is None:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available()
            else "cpu")

    if net_dict is not None and net_name in net_dict:
        return net_dict[net_name]

    if ebp_version is not None and ebp_version < 4:
        raise DeprecationWarning('EBP version must be >= 4')

    if net_name == 'resnetv6_pytorch':
        resnet_path = os.path.join(xfr_root,
                                   'models/resnet101_l2_d512_twocrop.pth')
        model = resnet.resnet101v6(resnet_path, device)

        model.to(device)
        wbnet = whitebox.WhiteboxSTResnet(
            model,
        )
        net = whitebox.Whitebox(
            wbnet,
            ebp_version=ebp_version,
        ).to(device)

        net.match_threshold = 0.9636
        net.platts_scaling = 15.05
        return net

    elif net_name == 'resnetv4_pytorch':
        resnet_path = os.path.join(xfr_root,
                                   'models/resnet101v4_28NOV17_train.pth')
        model = resnet.resnet101v6(resnet_path, device)

        model.to(device)
        wbnet = whitebox.WhiteboxSTResnet(
            model,
        )
        wb = whitebox.Whitebox(
            wbnet,
            ebp_version=ebp_version,
        ).to(device)

        # From Caffe ResNetv4:
        # wb.match_threshold = 0.9252
        # wb.platts_scaling = 17.71
        wb.match_threshold = 0.9722
        wb.platts_scaling = 16.61
        return wb

    elif net_name == 'vggface2_resnet50':
        param_path = os.path.join(
            xfr_root, 'models/resnet101v4_28NOV17_train.pth')
        net = xfr.models.resnet50_128.resnet50_128(param_path)
        wb = xfr.models.whitebox.Whitebox(
            xfr.models.whitebox.Whitebox_resnet50_128(net))
        return wb

    else:
        if net_dict is None:
            raise NotImplemented(
                'create_net does not implemented network "%s"' %
                net_name
            )
        else:
            raise KeyError(
                'Could not find net "%s" in net_dict and '
                'create_net does not implemented that network.' %
                net_name
            )


def save_model(net, filename):
    """Save trained model. """
    if not os.path.exists(os.path.basename(filename)):
        os.makedirs(os.path.basename(filename))
    torch.save(net.state_dict(), filename)
    print("save pretrained model to: {}".format(filename))

def copy_files(paths, output_dir):
    """ Copy files to output / run directory with name that encodes the absolute
        path, in case it is needed to regenerate the results.
    """
    for path in paths:
        assert len(path) > 1, 'Make sure you pass a list of paths and not a single string!'
        path = os.path.abspath(path)
        new_fn = os.path.join(
            output_dir,
            path.replace('/','%')
        )

        shutil.copy2(path, new_fn)

def normalize_gpus(gpus, setEnviron = False):
    """ Given a set of gpus passed from the command line, convert that into
        something that's consistent with the setting of CUDA_VISIBLE_DEVICES.
        Throws an exception if the output length of the gpu list is less than
        the input length.
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        return gpus
    envString = os.environ['CUDA_VISIBLE_DEVICES']
    if not envString:
        return gpus
    # If there's no setting, just accept the list we were given.
    originalLen = len(gpus)
    visible = envString.split(',')
    if len(visible) < originalLen:
        raise ValueError("Command line specified more GPUs than are available via CUDA_VISIBLE_DEVICES")
    newGpus = []
    for x in gpus:
        if x < len(visible):
            newGpus.append(visible[x])
        else:
            raise ValueError("Command line GPU {} is outside visible range".format(x))
    if setEnviron:
        newEnvString = ','.join([str(x) for x in newGpus])
        os.environ['CUDA_VISIBLE_DEVICES'] = newEnvString
    return newGpus
    
