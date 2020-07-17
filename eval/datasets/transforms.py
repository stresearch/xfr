import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import io

def create_transforms(net_preproc_fn, transform, jitter, blur_radius=None):
    """ net_preproc_fn - perform mean subtraction as required by network, convert
            RGB byte image to a FloatTensor.
    """
    if transform == 'minimal':
        return preprocess_minimal(net_preproc_fn, jitter=jitter)
    elif transform == 'grayscale':
        return preprocess_grayscale(net_preproc_fn, jitter=jitter)
    elif transform == 'invert-grayscale':
        return preprocess_invert_grayscale(net_preproc_fn, jitter=jitter)
    elif transform == 'blur-grayscale':
        return preprocess_blur_grayscale(net_preproc_fn, blur_radius, jitter=jitter)
    else:
        raise RuntimeError('Unknown transform %s' % transform)


def generate_twocrop_ensemble():
    resize230 = transforms.Resize(230)
    resize256 = transforms.Resize(256)
    resize282 = transforms.Resize(282)
    crop = transforms.CenterCrop((224,224))
    flip = transforms.functional.hflip
    def twocrop_ensemble(img):
        crop1 = crop(resize230(img))
        crop2 = crop(resize256(img))
        crop3 = crop(resize282(img))
        return (crop1, flip(crop1), crop2, flip(crop2), crop3, flip(crop3))
    return twocrop_ensemble


def prepare_image_fn(jitter=False, blur_radius=None, blur_prob=1.0):
    """
    Returns a torchvision.Transform
    Input, Output to the returned transform is a PIL.Image.  Crops, jitters, etc.
    """
    transform_list = [transforms.Resize(256),]
    if jitter:
        transform_list.append(transforms.RandomCrop((224,224)))
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    else:
        transform_list.append(transforms.CenterCrop((224,224)))
    if blur_radius is not None and blur_prob > 0:
        transform_list.append(transforms.Lambda(generate_random_blur(blur_radius, blur_prob)))
    return transforms.Compose(transform_list)


def preprocess_minimal(net_preproc_fn, jitter):
    """ preprocessing function suitable for transform argument of
        datasets.ImageFolder
    """
    preproc_fn = prepare_image_fn(jitter=jitter)

    return transforms.Compose((preproc_fn,
                               transforms.Lambda(net_preproc_fn)))


def preprocess_grayscale(net_preproc_fn, jitter):
    """ preprocessing function suitable for transform argument of
        datasets.ImageFolder
    """
    return preprocess_blur_grayscale(net_preproc_fn, blur_radius=None, jitter=jitter)


def preprocess_invert_grayscale(net_preproc_fn):
    """ preprocessing function suitable for transform argument of
        datasets.ImageFolder
    """
    preproc_fn = prepare_image_fn(jitter=True)

    return transforms.Compose((preproc_fn,
                               transforms.Grayscale(3),
                               transforms.Lambda(net_preproc_fn)))


def preprocess_blur_grayscale(net_preproc_fn, blur_radius=None, blur_prob=1.0, jitter=True):
    """ preprocessing function suitable for transform argument of
        datasets.ImageFolder
    """
    preproc_fn = prepare_image_fn(jitter=jitter)

    transform_list = [preproc_fn]
    if blur_radius is not None and blur_prob > 0:
        transform_list.append(transforms.Lambda(
            generate_random_blur(blur_radius, blur_prob)))

    # transform_list.append(transforms.Lambda(to_grayscale))
    transform_list.append(transforms.Grayscale(3))
    transform_list.append(transforms.Lambda(net_preproc_fn))
    return transforms.Compose(transform_list)


def preprocess_with_artifacts(net_preproc_fn, jpeg_quality_range, scale_factor_range, jitter=True):
    """ preprocessing function suitable for transform argument of datasets.ImageFolder
        jitter only impacts the cropping, horizontal flip, and color
        jpeg_quality and scale factor can still be jittered if they contain a range
    """
    preproc_fn = prepare_image_fn(jitter=jitter)
    artifact_fn = generate_induce_artifacts(jpeg_quality_range, scale_factor_range)
    return transforms.Compose((preproc_fn, artifact_fn, transforms.Lambda(net_preproc_fn)))


# ============================================================================
#
#                     Distortion generation function
#
# ============================================================================

def generate_random_blur(blur_radius, blur_prob):
    def random_blur(img):
        if np.random.random() < blur_prob and blur_radius > 0:
            return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        else:
            return img
    return random_blur


def generate_induce_artifacts(jpeg_quality_range, scale_factor_range):
    assert len(jpeg_quality_range) == 2
    assert all([1 <= val <= 100 for val in jpeg_quality_range])
    assert jpeg_quality_range[0] <= jpeg_quality_range[1]
    assert len(scale_factor_range) == 2
    assert all([0 < val <= 1 for val in scale_factor_range])
    assert scale_factor_range[0] <= scale_factor_range[1]
    log_scale_min = np.log(scale_factor_range[0])
    log_scale_max = np.log(scale_factor_range[1])

    def induce_artifacts(img):
        log_scale = np.random.uniform(log_scale_min, log_scale_max)
        scale = np.exp(log_scale)
        quality = int(np.random.uniform(jpeg_quality_range[0], jpeg_quality_range[1]))
        # resize the image
        new_size = (int(img.size[0]*scale), int(img.size[1]*scale))
        img_small = img.resize(new_size)
        # JPEG compress/uncompress in memory
        f = io.BytesIO()
        img_small.save(f, format='JPEG', quality=quality)
        img_small = Image.open(f)
        # size back to original
        img = img_small.resize(img.size)
        return img
    return transforms.Lambda(induce_artifacts)


# ============================================================================
#
#                     ResNet Specific?
#
# ============================================================================

def resnet101v4_preprocess(jitter=False, blur_radius=None, blur_prob=1.0):
    """ preprocessing function suitable for transform argument of datasets.ImageFolder """
    # finally, convert PIL RGB image to FloatTensor
    prep_fn = prepare_image_fn(jitter, blur_radius, blur_prob)
    return transforms.Compose((prep_fn, transforms.Lambda(convert_resnet101v4_image)))


def resnet101v4_preprocess_with_artifacts(*args, **kwargs):
    return preprocess_with_artifacts(convert_resnet101v4_image, *args, **kwargs)

# def resnet101v4_preprocess_with_artifacts(jpeg_quality_range, scale_factor_range, jitter=True):
#     prep_fn = prepare_image_fn(jitter=jitter)
#     artifact_fn = generate_induce_artifacts(
#         jpeg_quality_range, scale_factor_range)
#     return transforms.Compose((prep_fn, artifact_fn, transforms.Lambda(convert_resnet101v4_image)))


def resnet101v4_preprocess_twocrop_ensemble():
    """
    Preprocessing function suitable for transform argument of datasets.ImageFolder
    Generates 6 images per input, 3 scales x horizontal flip
    """
    crop_fn = generate_twocrop_ensemble()
    def crop_and_convert(img):
        crops = [convert_resnet101v4_image(crop) for crop in crop_fn(img)]
        # stack all crops into a single tensor
        return torch.stack(crops, dim=0)
    return crop_and_convert


