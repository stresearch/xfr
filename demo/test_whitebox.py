
import sys
import os.path
import torch
import PIL
import numpy as np
import pdb
import uuid
import torch.nn.functional as F
import vipy.image
import vipy.visualize
import vipy.util
import tempfile

sys.path.append('../python')
from xfr.models.whitebox import WhiteboxSTResnet, Whitebox
from xfr.models.resnet import stresnet101
import xfr.models.whitebox
import xfr.models.lightcnn
import xfr.show

sys.path.append('../python/strface')
import strface.detection


def _detector(imgfile):
    """Faster RCNN face detector wrapper"""
    im = np.array(PIL.Image.open(imgfile))
    if torch.cuda.is_available():
        gpu_index = 0
    else:
        gpu_index = -1
    net = strface.detection.FasterRCNN(model_dir='../python/strface/models/detection', gpu_index=gpu_index, conf_threshold=None, rotate_flags=None, rotate_thresh=None, fusion_thresh=None, test_scales=800, max_size=1300)
    return net(im)

def _blend_saliency_map(img, smap, scale_factor=1.0, gamma=0.3, blur_sigma=0.05):
    """Input is PIL image, smap is real valued saliency map, output is PIL image"""
    img_blend = xfr.show.blend_saliency_map(np.array(img).astype(np.float32)/255.0, smap, blur_sigma=blur_sigma, gamma=gamma, scale_factor=scale_factor)
    return PIL.Image.fromarray(np.uint8(img_blend*255))  # [0,255] PIL image

def _encode_triplet_test_cases(wb, mask=None):
    """Run face detection and network encoding for standardized (mate, non-mate, probe) triplets for testing"""
    if mask is None:
        f_probe = '../data/n00000001_00000117.JPEG'
        f_nonmate = '../data/n00000002_00000100.JPEG'
        f_mate = '../data/n00000001_00000384.JPEG'
    elif mask == 'nose':
        f_probe = '../data/inpainting-game/IJBC/aligned/8/img/1017/inpainted/00002_truth.png'   # obama
        f_nonmate = '../data/inpainting-game/IJBC/aligned/8/img/1017/inpainted/00002_out_0.png'  # obama non-mate: inpainted probe
        f_mate = '../data/inpainting-game/IJBC/aligned/8/img/1406/inpainted/00002_truth.png'  # obama mate
    elif mask == 'mouth':
        f_probe = '../data/inpainting-game/IJBC/aligned/8/img/1017/inpainted/00001_truth.png'   # obama
        f_nonmate = '../data/inpainting-game/IJBC/aligned/8/img/1017/inpainted/00001_out_0.png'  # obama non-mate: inpainted  probe
        f_mate = '../data/inpainting-game/IJBC/aligned/8/img/1406/inpainted/00001_truth.png'  # obama mate
    elif mask == 'eyes':
        f_probe = '../data/inpainting-game/IJBC/aligned/8/img/1017/inpainted/00004_truth.png'   # obama
        f_nonmate = '../data/inpainting-game/IJBC/aligned/8/img/1017/inpainted/00004_out_0.png'  # obama non-mate: inpainted probe
        f_mate = '../data/inpainting-game/IJBC/aligned/8/img/1406/inpainted/00004_truth.png'  # obama mate
    else:
        raise ValueError('invalid mask region "%s"' % mask)

    bb_probe = _detector(f_probe)[0]
    bb_nonmate = _detector(f_nonmate)[0]
    bb_mate = _detector(f_mate)[0]

    im_probe = vipy.image.ImageDetection(filename=f_probe).boundingbox(xmin=bb_probe[0], ymin=bb_probe[1], width=bb_probe[2], height=bb_probe[3]).dilate(1.1).crop().mindim(256).centercrop(224, 224)
    im_nonmate = vipy.image.ImageDetection(filename=f_nonmate).boundingbox(xmin=bb_nonmate[0], ymin=bb_nonmate[1], width=bb_nonmate[2], height=bb_nonmate[3]).dilate(1.1).crop().mindim(256).centercrop(224, 224)
    im_mate = vipy.image.ImageDetection(filename=f_mate).boundingbox(xmin=bb_mate[0], ymin=bb_mate[1], width=bb_mate[2], height=bb_mate[3]).dilate(1.1).crop().mindim(256).centercrop(224, 224)
    img_probe_display = im_probe.clone().resize(112,112).rgb().numpy()  # numpy image

    x_mate = wb.net.encode(wb.net.preprocess(im_mate.pil()))
    x_nonmate = wb.net.encode(wb.net.preprocess(im_nonmate.pil()))
    img_probe = wb.net.preprocess(im_probe.pil())  # torch tensor

    return (x_mate, x_nonmate, img_probe, img_probe_display)

def ebp():
    """Excitation backprop in pytorch"""

    # Create whitebox object with STR-Janus resnet-101 convolutional network
    wb = Whitebox(WhiteboxSTResnet(stresnet101('../models/resnet101v4_28NOV17_train.pth')))
    x_probe = wb.net.preprocess(PIL.Image.open('../data/demo_face.jpg'))

    # Generate Excitation backprop (EBP) saliency map at first convolutional layer
    P = torch.zeros( (1, wb.net.num_classes()) );  P[0][0] = 1.0;  # one-hot prior probability
    img_saliency = wb.ebp(x_probe, P)

    # Overlay saliency map with display image
    img_display = PIL.Image.open('../data/demo_face.jpg').resize( (112,112) )
    outfile = 'test_whitebox_ebp.jpg';
    _blend_saliency_map(img_display, img_saliency).save(outfile)
    print('[test_whitebox.ebp]: saving saliency map blended overlay to "./%s"' % outfile)

def contrastive_ebp():
    """Contrastive excitation backprop"""
    wb = Whitebox(WhiteboxSTResnet(stresnet101('../models/resnet101v4_28NOV17_train.pth')))
    x_probe = wb.net.preprocess(PIL.Image.open('../data/demo_face.jpg'))
    img_saliency = wb.contrastive_ebp(x_probe, k_poschannel=0, k_negchannel=100)
    outfile = './test_whitebox_contrastive_ebp.jpg';
    _blend_saliency_map(PIL.Image.open('../data/demo_face.jpg').resize( img_saliency.shape ), img_saliency).save(outfile)
    print('[test_whitebox.contrastive_ebp]: saving saliency map blended overlay to "%s"' % outfile)

def truncated_contrastive_ebp():
    """Truncated contrastive excitation backprop"""
    wb = Whitebox(WhiteboxSTResnet(stresnet101('../models/resnet101v4_28NOV17_train.pth')))
    x_probe = wb.net.preprocess(PIL.Image.open('../data/demo_face.jpg'))
    img_saliency = wb.truncated_contrastive_ebp(x_probe, k_poschannel=0, k_negchannel=100, percentile=20)
    outfile = './test_whitebox_truncated_contrastive_ebp.jpg';
    _blend_saliency_map(PIL.Image.open('../data/demo_face.jpg').resize(img_saliency.shape), img_saliency).save(outfile)
    print('[test_whitebox.truncated_contrastive_ebp]: saving saliency map blended overlay overlay to "%s"' % outfile)

def triplet_ebp():
    """Triplet excitation backprop"""
    print('[test_whitebox.triplet_ebp]: Detection and encoding for (mate, probe)')
    wb = Whitebox(WhiteboxSTResnet(stresnet101('../models/resnet101v4_28NOV17_train.pth')))
    (x_mate, x_nonmate, img_probe, img_probe_display) = _encode_triplet_test_cases(wb, 'nose')
    wb.net.set_triplet_classifier((1.0/2500.0)*x_mate, (1.0/2500.0)*x_nonmate)  # rescale encodings to avoid softmax overflow
    P = torch.zeros( (1, wb.net.num_classes()) );  P[0][0] = 1.0;  # one-hot prior probability for mate
    img_saliency = wb.ebp(img_probe, P)
    outfile = './test_whitebox_triplet_ebp.jpg';
    _blend_saliency_map(img_probe_display, img_saliency).save(outfile)
    print('[test_whitebox.triplet_ebp]: saving saliency map blended overlay to "%s"' % outfile)

def contrastive_triplet_ebp():
    """Contrastive triplet excitation backprop"""
    print('[test_whitebox.contrastive_triplet_ebp]: Detection and encoding for (mate, non-mate, probe) triplet')
    wb = Whitebox(WhiteboxSTResnet(stresnet101('../models/resnet101v4_28NOV17_train.pth')))
    (x_mate, x_nonmate, img_probe, img_probe_display) = _encode_triplet_test_cases(wb, 'nose')
    wb.net.set_triplet_classifier((1.0/2500.0)*x_mate, (1.0/2500.0)*x_nonmate)  # rescale encodings to avoid softmax overflow
    img_saliency = wb.contrastive_ebp(img_probe, k_poschannel=0, k_negchannel=1)
    outfile = './test_whitebox_contrastive_triplet_ebp.jpg';
    _blend_saliency_map(img_probe_display, img_saliency).save(outfile)
    print('[test_whitebox.contrastive_triplet_ebp]: saving saliency map blended overlay to "%s"' % outfile)

def truncated_contrastive_triplet_ebp():
    """Truncated contrastive triplet excitation backprop"""
    print('[test_whitebox.truncated_contrastive_triplet_ebp]: Detection and encoding for (mate, non-mate, probe) triplet')
    wb = Whitebox(WhiteboxSTResnet(stresnet101('../models/resnet101v4_28NOV17_train.pth')))
    (x_mate, x_nonmate, img_probe, img_probe_display) = _encode_triplet_test_cases(wb, 'nose')
    wb.net.set_triplet_classifier((1.0/2500.0)*x_mate, (1.0/2500.0)*x_nonmate)  # rescale encodings to avoid softmax overflow
    img_saliency = wb.truncated_contrastive_ebp(img_probe, k_poschannel=0, k_negchannel=1, percentile=20)
    outfile = './test_whitebox_truncated_contrastive_triplet_ebp.jpg';
    _blend_saliency_map(img_probe_display, img_saliency).save(outfile)
    print('[test_whitebox.truncated_contrastive_triplet_ebp]: saving saliency map blended overlay to "%s"' % outfile)

def layerwise_ebp():
    """EBP alpha transparency montage starting from each interior layer and selected node specified by argmax excitation at this layer"""
    raise('Deprecated')
    
    wb = Whitebox(WhiteboxSTResnet(stresnet101('../models/resnet101v4_28NOV17_train.pth')), ebp_version=5, ebp_subtree_mode='all')
    im_probe = PIL.Image.open('../data/demo_face.jpg')
    img_display = np.array(im_probe.resize( (112,112) ))
    x_probe = wb.net.preprocess(im_probe)

    imlist = []
    layers = wb._layers()
    for k in range(0, len(layers)):
        print('[test_layerwise_ebp][%d/%d]: layerwise EBP "%s"' % (k, len(layers), layers[k]))
        img_saliency = wb.layerwise_ebp(x_probe, k_poschannel=0, k_layer=k, mode='argmax', mwp=False)
        outfile = os.path.join(tempfile.gettempdir(), '%s.jpg' % uuid.uuid1().hex)  # tempfile for montage
        img = PIL.Image.fromarray(np.concatenate( (img_display, np.expand_dims(np.minimum(255,10+img_saliency), 2)), axis=2)).convert('RGBA')
        bg = PIL.Image.new('RGBA', img.size, (255,255,255))
        alpha_composite = PIL.Image.alpha_composite(bg, img).convert('RGB').save(outfile, 'JPEG', quality=95)
        imlist.append(vipy.image.ImageDetection(filename=outfile, xmin=0, ymin=0, width=112, height=112).rgb())

    f_montage = './test_whitebox_layerwise_ebp.jpg'
    im_montage = vipy.visualize.montage(imlist, 112, 112, grayscale=False, skip=False, border=1)
    vipy.util.imwrite(im_montage.numpy(), f_montage)
    print('[test_whitebox.layerwise_ebp]: Saving montage (rowwise by layers, approaching the image layer in bottom right) to "%s"' % f_montage)



def weighted_subtree_triplet_ebp(topk=1, mask='nose'):
    """Weighted subtree excitation backprop, montage visualization of sorted layers"""

    print('[test_whitebox.weighted_subtree_triplet_ebp]: Detection and encoding for (mate, non-mate, probe) triplet')
    wbnet = WhiteboxSTResnet(stresnet101('../models/resnet101v4_28NOV17_train.pth'))
    wb = Whitebox(wbnet, ebp_version=5)
    (x_mate, x_nonmate, img_probe, img_probe_display) = _encode_triplet_test_cases(wb, mask=mask)
    wb.net.set_triplet_classifier((1.0/2500.0)*x_mate, (1.0/2500.0)*x_nonmate)  # rescale for softmax

    print('[test_whitebox.weighted_subtree_triplet_ebp]: topk=%d weighted subtree for ground truth mask region "%s"' % (topk, mask))
    (img_subtree, P_img, P_subtree, k_subtree) = wb.weighted_subtree_ebp(img_probe, k_poschannel=0, k_negchannel=1, topk=topk, do_max_subtree=False, subtree_mode='all', do_mated_similarity_gating=True)
    print('[test_whitebox.weighted_subtree_triplet_ebp]: weighted subtree EBP, selected layers=%s, P=%s' % (str(k_subtree), str(P_subtree)))

    imlist = []
    P_subtree.append(1.0)  # this contains the scale factors 
    P_img.append(img_subtree)
    for (k, (p, img_saliency)) in enumerate(zip(P_subtree, P_img)):
        outfile = os.path.join(tempfile.gettempdir(), '%s.jpg' % uuid.uuid1().hex)
        #alpha_composite = _blend_saliency_map(img_probe_display, np.float32(img_saliency)/255.0, scale_factor=1.0/(p+1E-12)).save(outfile, 'JPEG', quality=95)  # EBP scale factor for display 
        alpha_composite = _blend_saliency_map(img_probe_display, np.float32(img_saliency)/255.0, scale_factor=1.0).save(outfile, 'JPEG', quality=95)    # uniform scale factor for display
        imlist.append(vipy.image.ImageDetection(filename=outfile, xmin=0, ymin=0, width=112, height=112).rgb())

    f_montage = './test_whitebox_weighted_subtree_ebp_topk_%d_mask_%s.jpg' % (topk, mask)
    im_montage = vipy.visualize.montage(imlist, imgheight=112, imgwidth=112, skip=False, border=1)
    vipy.util.imwrite(im_montage.numpy(), f_montage)
    print('[test_whitebox.weighted_subtree_triplet_ebp]: Saving montage (rowwise subtree, sorted by increasing gradient weight) to "%s"' % f_montage)
    print('[test_whitebox.weighted_subtree_triplet_ebp]: Final image in montage (bottom right) is weighted subtree saliency map')


def ebp_lightcnn():
    """Light CNN: This model must be downloaded from: https://github.com/AlfredXiangWu/LightCNN and copied into 'models'"""
    if not os.path.exists('../models/LightCNN_29Layers_V2_checkpoint.pth.tar'):
        print('[test_whitebox.ebp_lightcnn]  Download the light-CNN model file "LightCNN_29Layers_V2_checkpoint.pth.tar" from https://github.com/AlfredXiangWu/LightCNN and copy into ../models/LightCNN_29Layers_V2_checkpoint.pth.tar')
        return None
    net = xfr.models.lightcnn.LightCNN_29Layers_v2(num_classes=80013)
    statedict = xfr.models.lightcnn.Load_Checkpoint('../models/LightCNN_29Layers_V2_checkpoint.pth.tar')
    net.load_state_dict(statedict)
    wb = xfr.models.whitebox.Whitebox(xfr.models.whitebox.WhiteboxLightCNN(net), ebp_subtree_mode='affineonly')
    x_probe = wb.net.preprocess(PIL.Image.open('../data/demo_face.jpg'))
    P = torch.zeros( (1, wb.net.num_classes()) );  P[0][0] = 1.0;  # one-hot prior probability
    img_saliency = wb.ebp(x_probe, P, mwp=False)

    img_display = PIL.Image.open('../data/demo_face.jpg').resize( (128,128) )
    outfile = './test_whitebox_ebp_lightcnn.jpg';
    _blend_saliency_map(img_display, img_saliency).save(outfile)
    print('[test_whitebox.ebp_lightcnn]: saving saliency map blended overlay to "%s"' % outfile)
    return img_saliency


def ebp_senet50_256():
    """VGGFace2 senet-50-256d:  https://github.com/ox-vgg/vgg_face2"""
    """This network will throw an exception due to currently unsupported Sigmoid() layers"""
    sys.path.append('../models/senet50_256_pytorch')
    import senet50_256
    net = senet50_256.senet50_256('../models/senet50_256_pytorch/senet50_256.pth')
    wb = xfr.models.whitebox.Whitebox(xfr.models.whitebox.Whitebox_senet50_256(net))

    x_probe = wb.net.preprocess(PIL.Image.open('../data/demo_face.jpg'))
    P = torch.zeros( (1, wb.net.num_classes()) );  P[0][0] = 1.0;  # one-hot prior probability
    img_saliency = wb.ebp(x_probe, P, mwp=False)
    img_display = PIL.Image.open('../data/demo_face.jpg').resize( (112,112) )
    outfile = './test_whitebox_ebp_senet50_256.jpg';
    _blend_saliency_map(img_display, img_saliency).save(outfile)
    print('[test_whitebox.ebp_senet50_256]: saving saliency map blended overlay to "%s"' % outfile)
    return img_saliency


def ebp_resnet50_128():
    """VGGFace2 resnet-50-128d:  https://github.com/ox-vgg/vgg_face2"""
    sys.path.append('../models/resnet50_128_pytorch')
    import resnet50_128
    net = resnet50_128.resnet50_128('../models/resnet50_128_pytorch/resnet50_128.pth')
    wb = xfr.models.whitebox.Whitebox(xfr.models.whitebox.Whitebox_resnet50_128(net))

    x_probe = wb.net.preprocess(PIL.Image.open('../data/demo_face.jpg'))
    P = torch.zeros( (1, wb.net.num_classes()) );  P[0][0] = 1.0;  # one-hot prior probability
    img_saliency = wb.ebp(x_probe, P, mwp=False)
    img_display = PIL.Image.open('../data/demo_face.jpg').resize( (112,112) )
    outfile = './test_whitebox_ebp_resnet50_128.jpg';
    _blend_saliency_map(img_display, img_saliency).save(outfile)
    print('[test_whitebox.ebp_resnet50_128]: saving saliency map blended overlay to "%s"' % outfile)
    return img_saliency


if __name__ == "__main__":
    """CPU-only demos (cached output in ./whitebox/*.jpg)"""
    # Baseline whitebox discriminative visualization methods
    if torch.cuda.is_available():
        dev = torch.device(0)
    else:
        dev = torch.device("cpu")
    # taa: Not what we meant at all.
    # torch.cuda.device(dev)
    ebp()
    contrastive_ebp()
    truncated_contrastive_ebp()
    triplet_ebp()
    contrastive_triplet_ebp()
    truncated_contrastive_triplet_ebp()

    # Weighted subtree whitebox discriminative visualization method
    weighted_subtree_triplet_ebp(topk=64, mask='nose')
    weighted_subtree_triplet_ebp(topk=64, mask='eyes')
    weighted_subtree_triplet_ebp(topk=64, mask='mouth')

    # Other whitebox face matchers
    ebp_resnet50_128()
    ebp_lightcnn()  # requires a model download (non-commercial use, see README)
