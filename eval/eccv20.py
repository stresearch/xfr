import sys
import os.path
import torch
import PIL
import numpy as np
import pdb
import uuid
import torch.nn.functional as F
import tempfile
from  scipy.spatial.distance import pdist, squareform
import copy
import random

np.random.seed(42)  # for repeatable take

import vipy.image
from vipy.image import ImageDetection
import vipy.visualize
import vipy.util
import vipy.linalg
from vipy.dataset.vggface2 import VGGFace2

sys.path.append('../python')  
from xfr.models.whitebox import WhiteboxSTResnet, Whitebox
from xfr.models.resnet import stresnet101
import xfr.models.whitebox
import xfr.show

sys.path.append('../demo')
from test_whitebox import _blend_saliency_map

sys.path.append('../python/strface')  
import strface.detection
face_detector = strface.detection.FasterRCNN(model_dir='../python/strface/models/detection', gpu_index=0 if torch.cuda.is_available() else -1, conf_threshold=None, rotate_flags=None, rotate_thresh=None, fusion_thresh=None, test_scales=800, max_size=1300)


def _detector(imgfile):
    """Faster RCNN face detector wrapper"""
    im = np.array(PIL.Image.open(imgfile))
    return face_detector(im)


def _vggface2_topk_frontal_nonmates(wb, topk):
    np.random.seed(42)  # for repeatable take
    n_minibatch = 2
    vggface2 = VGGFace2('/proj/janus6/vggface2')
    imlist = vipy.util.chunklistbysize([im for im in vggface2.frontalset(n_frontal=n_minibatch)], n_minibatch)
    imlist_preprocessed = [torch.cat([wb.net.preprocess(f_detection(im).pil()) for im in iml], dim=0) for iml in imlist]  # minibatch tensor
    X = [torch.squeeze(torch.sum(wb.net.encode(imchunk), dim=0)).detach().numpy() for imchunk in imlist_preprocessed]  # minibatch encode template
    X = vipy.linalg.row_normalized(np.array(X))
    X_subjectid = [imchunk[0].category() for imchunk in imlist]
    d_subjectid_to_topk_frontal_nonmates = {}
    for (k, d) in enumerate(squareform(pdist(X, metric='euclidean'))):
        j_sorted = np.argsort(d)[1:]  # increasing, do not include self distance=0 on diagonal
        d_subjectid_to_topk_frontal_nonmates[X_subjectid[k]] = [X_subjectid[j] for j in j_sorted[0:topk]]        
    vipy.util.save(d_subjectid_to_topk_frontal_nonmates, '_vggface2_topk_frontal_nonmates.pkl')  # cache
    return d_subjectid_to_topk_frontal_nonmates

def _vggface2_topk_nonmates(wb, topk):
    np.random.seed(42)  # for repeatable take
    n_minibatch = 2
    vggface2 = VGGFace2('/proj/janus6/vggface2')
    imlist = vipy.util.chunklistbysize([im for im in vggface2.take_per_subject(n_minibatch)], n_minibatch)
    imlist_preprocessed = [torch.cat([wb.net.preprocess(f_detection(im).pil()) for im in iml], dim=0) for iml in imlist]  # minibatch tensor
    X = [torch.squeeze(torch.sum(wb.net.encode(imchunk), dim=0)).detach().numpy() for imchunk in imlist_preprocessed]  # minibatch encode template
    X = vipy.linalg.row_normalized(np.array(X))
    X_subjectid = [imchunk[0].category() for imchunk in imlist]
    d_subjectid_to_topk_frontal_nonmates = {}
    for (k, d) in enumerate(squareform(pdist(X, metric='euclidean'))):
        j_sorted = np.argsort(d)[1:]  # increasing, do not include self distance=0 on diagonal
        d_subjectid_to_topk_frontal_nonmates[X_subjectid[k]] = [X_subjectid[j] for j in j_sorted[0:topk]]        
    vipy.util.save(d_subjectid_to_topk_frontal_nonmates, '_vggface2_topk_nonmates.pkl')  # cache
    return d_subjectid_to_topk_frontal_nonmates

def _vggface2_nonmates():
    np.random.seed(42)  # for repeatable take
    return VGGFace2('/proj/janus6/vggface2').take_per_subject(1)

def _triplet_mate_frontalpose_nonmate_top1_probe_mixedpose(n_subjects=32):
    np.random.seed(42)  # for repeatable take
    vggface2 = VGGFace2('/proj/janus6/vggface2')
    frontalset = [im for im in vggface2.frontalset(n_frontal=1)]
    matelist = frontalset[0:n_subjects]

    if n_subjects == 16:
        matelist[3] = frontalset[n_subjects+1]
        matelist[5] = frontalset[n_subjects+5]
        matelist[6] = frontalset[n_subjects+4]
        matelist[11] = frontalset[n_subjects+7]
        matelist[12] = frontalset[n_subjects+2]
        matelist[13] = frontalset[n_subjects+9]
        matelist[15] = frontalset[n_subjects+6]
        
    d_subjectid_to_topk_frontal_nonmates = vipy.util.load('_vggface2_topk_frontal_nonmates.pkl')  # cached
    nonmateidlist = []
    for m in matelist:
        for n in d_subjectid_to_topk_frontal_nonmates[m.category()]:
            if n not in nonmateidlist:
                nonmateidlist.append(n)
                break
    d_frontalset = {x.category():x for x in frontalset}  # for id lookup
    nonmatelist = [d_frontalset[k] for k in nonmateidlist]  # ordered
    probelist = [vggface2.take(n_subjects, im_mate.category()) for im_mate in matelist]

    assert(len(nonmatelist) == n_subjects)
    assert(len(probelist) == n_subjects)
    assert(len(probelist[0]) == n_subjects)
    assert(len(matelist) == n_subjects)

    return (matelist, nonmatelist, probelist)


def _k_mates_with_m_probes(n_subjects, n_probes):
    np.random.seed(42)  # for repeatable take
    vggface2 = VGGFace2('/proj/janus6/vggface2')
    subjects = np.random.choice(vggface2.subjects(), n_subjects)
    imsubjects = {s:list(vggface2.subjectset(s)) for s in subjects}
    matelist = [imsubjects[s][0] for s in subjects]
    probelist = [imsubjects[s][1:n_probes+1] for s in subjects]
    return (matelist, probelist)


def _n_subjects_k_mates_with_m_probes(n_subjects, k_mates, m_probes, mateset=None):
    np.random.seed(42)  # for repeatable take
    vggface2 = VGGFace2('/proj/janus6/vggface2')
    subjects = np.random.choice(vggface2.subjects(), n_subjects) if mateset is None else mateset
    imsubjects = {s:list(vggface2.subjectset(s)) for s in subjects}
    matelist = [imsubjects[s][0:k_mates] for s in subjects]
    probelist = [imsubjects[s][k_mates:m_probes+k_mates] for s in subjects]
    return (matelist, probelist)

def _all_nonmates(n=None, mateset=set()):
    np.random.seed(42)  # for repeatable take
    vggface2 = VGGFace2('/proj/janus6/vggface2')
    subjects = vggface2.subjects()
    nonmates = subjects if n is None else subjects[0:n]
    nonmatelist = [next(vggface2.subjectset(s)) for s in nonmates if s not in mateset]  
    return (nonmatelist)


def _triplet_mate_frontalpose_nonmate_top1_probe_frontalpose():
    n_subjects = 9
    np.random.seed(42)  # for repeatable take
    vggface2 = VGGFace2('/proj/janus6/vggface2')
    frontalset = [im for im in vggface2.frontalset(n_frontal=n_subjects+1)]
    subjectid = list(set([im.category() for im in frontalset]))  # unique
    matelist = [im for im in frontalset if im.category() in subjectid[0:n_subjects]]
    d_mate = vipy.util.groupbyasdict(matelist, lambda im: im.category())
    matelist = [v[0] for (k,v) in d_mate.items()]
    probelist = [v[1:] for (k,v) in d_mate.items()]
    d_subjectid_to_topk_frontal_nonmates = vipy.util.load('_vggface2_topk_frontal_nonmates.pkl')  # cached
    nonmateidlist = []
    for m in matelist:
        for n in d_subjectid_to_topk_frontal_nonmates[m.category()]:
            if n not in nonmateidlist:
                # select unique identity from top-k
                nonmateidlist.append(n)
                break
    nonmatelist = [x for x in frontalset if x.category() in nonmateidlist]  # ordered
    d_nonmate = vipy.util.groupbyasdict(nonmatelist, lambda im: im.category())
    nonmatelist = [d_nonmate[k][0] for k in nonmateidlist]  # ordered

    assert(len(nonmatelist) == n_subjects)
    assert(len(probelist) == n_subjects)
    assert(len(probelist[0]) == n_subjects)
    assert(len(matelist) == n_subjects)

    return (matelist, nonmatelist, probelist)

def _triplet_mate_frontalpose_nonmate_topk_probe_frontalpose():
    n_subjects = 9
    vggface2 = VGGFace2('/proj/janus6/vggface2', seed=42)
    frontalset = [im for im in vggface2.frontalset(n_frontal=n_subjects+1)]
    subjectid = sorted(list(set([im.category() for im in frontalset])))  # unique
    matelist = [im for im in frontalset if im.category() in subjectid[0:n_subjects]]
    d_mate = vipy.util.groupbyasdict(matelist, lambda im: im.category())
    matelist = [v[0] for (k,v) in d_mate.items()]
    probelist = [v[1:] for (k,v) in d_mate.items()]
    d_subjectid_to_topk_frontal_nonmates = vipy.util.load('_vggface2_topk_nonmates.pkl')  # cached
    nonmateidlist = d_subjectid_to_topk_frontal_nonmates[matelist[8].category()][0:n_subjects]
    nonmatelist = [vggface2.take(1, k)[0] for k in nonmateidlist]
    matelist = matelist[8]    
    probelist = [probelist[8]]
    return (matelist, nonmatelist, probelist)


def _triplet_montage(wb, matelist, nonmatelist, probelist, outfile, f_saliency=None):

    X_mate = [wb.net.encode(wb.net.preprocess(im.pil())) for im in matelist]
    X_nonmate = [wb.net.encode(wb.net.preprocess(im.pil())) for im in nonmatelist]

    # Create saliency for each matrix entry, overwrite probelist
    for (i, (x_mate, im_mate)) in enumerate(zip(X_mate, matelist)):
        for (j, (x_nonmate, im_nonmate)) in enumerate(zip(X_nonmate, nonmatelist)):
            wb.net.set_triplet_classifier(x_mate, x_nonmate)
            if f_saliency is not None:
                img_saliency = f_saliency(probelist[i][j])
                probelist[i][j].buffer(img_saliency)

    # Montage
    imlist = [ImageDetection(xmin=0, ymin=0, xmax=256, ymax=256).buffer(np.uint8(np.zeros( (256,256,3) )))]
    imlist = imlist + nonmatelist
    for (im_mate, im_matedprobes) in zip(matelist, probelist):
        imlist.append(im_mate)
        imlist = imlist + im_matedprobes
    img_montage = vipy.visualize.montage(imlist, 112, 112, rows=len(matelist)+1, cols=len(nonmatelist)+1, grayscale=False, skip=False, border=1, crop=False)
    return vipy.util.imwrite(img_montage, outfile)



def f_saliency_whitebox_ebp(wb, im):
    P = torch.zeros( (1, wb.net.num_classes()) );  P[0][0] = 1.0;  # one-hot prior probability  
    img_saliency = wb.ebp(wb.net.preprocess(im.pil()), P)
    if np.max(img_saliency) == 255:
        img_saliency = img_saliency.astype(np.float32)/255.0
    return np.array(_blend_saliency_map(np.array(im.pil().resize(img_saliency.shape)), img_saliency, gamma=0.5))

def f_saliency_whitebox_cebp(wb, im):
    img_saliency = wb.contrastive_ebp(wb.net.preprocess(im.pil()), k_poschannel=0, k_negchannel=1)
    if np.max(img_saliency) == 255:
        img_saliency = img_saliency.astype(np.float32)/255.0
    return np.array(_blend_saliency_map(np.array(im.pil().resize(img_saliency.shape)), img_saliency, gamma=0.5))

def f_saliency_whitebox_tcebp(wb, im):
    img_saliency = wb.truncated_contrastive_ebp(wb.net.preprocess(im.pil()), k_poschannel=0, k_negchannel=1, percentile=20)
    if np.max(img_saliency) == 255:
        img_saliency = img_saliency.astype(np.float32)/255.0
    return np.array(_blend_saliency_map(np.array(im.pil().resize(img_saliency.shape)), img_saliency, gamma=0.5))

def f_saliency_whitebox_weighted_subtree(wb, im):
    img_probe = wb.net.preprocess(im.pil())
    (img_saliency, P_img, P_subtree, k_subtree) = wb.weighted_subtree_ebp(img_probe, k_poschannel=0, k_negchannel=1, topk=64, do_max_subtree=False, subtree_mode='all', do_mated_similarity_gating=True, verbose=False)    
    img_saliency = np.float32(img_saliency)/255.0
    return np.array(_blend_saliency_map(np.array(im.pil().resize(img_saliency.shape)), img_saliency, gamma=0.5))

def f_saliency_whitebox_weighted_subtree_lightcnn(wb, im):
    img_probe = wb.net.preprocess(im.pil())
    (img_saliency, P_img, P_subtree, k_subtree) = wb.weighted_subtree_ebp(img_probe, k_poschannel=0, k_negchannel=1, topk=64, do_max_subtree=False, subtree_mode='affineonly_with_prior', do_mated_similarity_gating=True, verbose=False)    
    img_saliency = np.float32(img_saliency)/255.0
    return np.array(_blend_saliency_map(np.array(im.pil().resize(img_saliency.shape)), img_saliency, gamma=0.5))


def f_detection(im):
    bb = _detector(im.filename())
    if len(bb) > 0:
        bb = bb[0]
        im = im.boundingbox(xmin=bb[0], ymin=bb[1], width=bb[2], height=bb[3]).dilate(1.1).crop().mindim(256).centercrop(224, 224)
    else:
        im = im.mindim(256).centercrop(224, 224)
    print(im)
    return im

def f_detection_nocrop(im):
    bb = _detector(im.filename())
    if len(bb) > 0:
        bb = bb[0]
        im = im.boundingbox(xmin=bb[0], ymin=bb[1], width=bb[2], height=bb[3])
    return im


def figure1():
    """16x16 frontal mates, frontal non-mates, any probe, resnet-101 whitebox"""

    wb = Whitebox(WhiteboxSTResnet(stresnet101('../models/resnet101_l2_d512_twocrop.pth')))
    if not os.path.exists('_vggface2_topk_frontal_nonmates.pkl'):
        _vggface2_topk_frontal_nonmates(wb, topk=32)  # recompute once
    n_subjects = 16
    (matelist, nonmatelist, probelist) = _triplet_mate_frontalpose_nonmate_top1_probe_mixedpose(n_subjects)    

    # Detection and color correction
    matelist = [f_detection(im).rgb() for im in matelist]
    nonmatelist = [f_detection(im).rgb() for im in nonmatelist]
    probelist = [[f_detection(im).rgb() for im in iml] for iml in probelist]
    probelist_clean = copy.deepcopy(probelist)

    # Figure 1a
    probelist = copy.deepcopy(probelist_clean)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure1a_%d.jpg' % n_subjects, f_saliency=None)
    print('[eccv20.figure1]: Saving montage to "%s"' % f_montage)
    probelist_1a = copy.deepcopy(probelist)
   
    # Figure 1b
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_ebp(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure1b_%d.jpg' % n_subjects, f_saliency=f_saliency)
    print('[eccv20.figure1]: Saving montage to "%s"' % f_montage)
    probelist_1b = copy.deepcopy(probelist)

    # Figure 1c
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_cebp(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure1c_%d.jpg' % n_subjects, f_saliency=f_saliency)
    print('[eccv20.figure1]: Saving montage to "%s"' % f_montage)
    probelist_1c = copy.deepcopy(probelist)

    # Figure 1d
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_tcebp(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure1d_%d.jpg' % n_subjects, f_saliency=f_saliency)
    print('[eccv20.figure1]: Saving montage to "%s"' % f_montage)
    probelist_1d = copy.deepcopy(probelist)

    # Figure 1e
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_weighted_subtree(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure1e_%d.jpg' % n_subjects, f_saliency=f_saliency)
    print('[eccv20.figure1]: Saving montage to "%s"' % f_montage)
    probelist_1e = copy.deepcopy(probelist)

    # Figure 1f
    probelist = copy.deepcopy(probelist_clean)
    matelist = [matelist[0]]*n_subjects
    probelist = [probelist_1a[0]] + [probelist_1b[0]] + [probelist_1c[0]] + [probelist_1d[0]] + [probelist_1e[0]]
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure1f_%d.jpg' % n_subjects, f_saliency=None)
    print('[eccv20.figure1]: Saving montage to "%s"' % f_montage)


def figure2():
    """One mate, top-k nonmates, row-wise by approach"""
    n_subjects = 10
    wb = Whitebox(WhiteboxSTResnet(stresnet101('../models/resnet101_l2_d512_twocrop.pth')))
    if not os.path.exists('_vggface2_topk_nonmates.pkl'):
        _vggface2_topk_nonmates(wb, topk=32)  # recompute once
    (matelist, nonmatelist, probelist) = _triplet_mate_frontalpose_nonmate_topk_probe_frontalpose()

    # Detection and color correction
    matelist = [f_detection(im).rgb() for im in matelist]
    nonmatelist = [f_detection(im).rgb() for im in nonmatelist]
    probelist = [[f_detection(im).rgb() for im in iml] for iml in probelist]
    probelist_clean = copy.deepcopy(probelist)

    # Figure 2a
    probelist = copy.deepcopy(probelist_clean)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure2a_%d.jpg' % n_subjects, f_saliency=None)
    probelist_2a = copy.deepcopy(probelist)
    print('[eccv20.figure2a]: Saving montage to "%s"' % f_montage)

    # Figure 2b
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_ebp(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure2b_%d.jpg' % n_subjects, f_saliency=f_saliency)
    probelist_2b = copy.deepcopy(probelist)

    # Figure 2c
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_cebp(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure2c_%d.jpg' % n_subjects, f_saliency=f_saliency)
    probelist_2c = copy.deepcopy(probelist)

    # Figure 2d
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_tcebp(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure2d_%d.jpg' % n_subjects, f_saliency=f_saliency)
    probelist_2d = copy.deepcopy(probelist)

    # Figure 2e
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_weighted_subtree(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure2e_%d.jpg' % n_subjects, f_saliency=f_saliency)
    probelist_2e = copy.deepcopy(probelist)

    # Figure 2f
    probelist = copy.deepcopy(probelist_clean)
    matelist = [matelist[0]]*n_subjects
    probelist = [probelist_2a[0]] + [probelist_2b[0]] + [probelist_2c[0]] + [probelist_2d[0]] + [probelist_2e[0]]
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure2f_%d.jpg' % n_subjects, f_saliency=None)
    print('[eccv20.figure2]: Saving montage to "%s"' % f_montage)


def figure3():
    """same as figure1, but light-cnn"""
    n_subjects = 16
    net = xfr.models.lightcnn.LightCNN_29Layers_v2(num_classes=80013)
    statedict = xfr.models.lightcnn.Load_Checkpoint('../models/LightCNN_29Layers_V2_checkpoint.pth.tar')  
    net.load_state_dict(statedict)    
    wb = xfr.models.whitebox.Whitebox(xfr.models.whitebox.WhiteboxLightCNN(net), ebp_subtree_mode='affineonly_with_prior', eps=1E-16, ebp_version=5)  # FIXME: the version matters
    if not os.path.exists('_vggface2_topk_frontal_nonmates.pkl'):
        _vggface2_topk_frontal_nonmates(wb, topk=32)  # recompute once
    (matelist, nonmatelist, probelist) = _triplet_mate_frontalpose_nonmate_top1_probe_mixedpose(n_subjects)    

    # Detection and color correction
    matelist = [f_detection(im).rgb() for im in matelist]
    nonmatelist = [f_detection(im).rgb() for im in nonmatelist]
    probelist = [[f_detection(im).rgb() for im in iml] for iml in probelist]
    probelist_clean = copy.deepcopy(probelist)

    # Figure 3a
    probelist = copy.deepcopy(probelist_clean)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure3a_%d.jpg' % n_subjects, f_saliency=None)
    print('[eccv20.figure3]: Saving montage to "%s"' % f_montage)
    probelist_1a = copy.deepcopy(probelist)
   
    # Figure 3b
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_ebp(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure3b_%d.jpg' % n_subjects, f_saliency=f_saliency)
    print('[eccv20.figure3]: Saving montage to "%s"' % f_montage)
    probelist_1b = copy.deepcopy(probelist)

    # Figure 3c
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_cebp(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure3c_%d.jpg' % n_subjects, f_saliency=f_saliency)
    print('[eccv20.figure3]: Saving montage to "%s"' % f_montage)
    probelist_1c = copy.deepcopy(probelist)

    # Figure 3d
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_tcebp(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure3d_%d.jpg' % n_subjects, f_saliency=f_saliency)
    print('[eccv20.figure3]: Saving montage to "%s"' % f_montage)
    probelist_1d = copy.deepcopy(probelist)

    # Figure 3e
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_weighted_subtree_lightcnn(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure3e_%d.jpg' % n_subjects, f_saliency=f_saliency)
    print('[eccv20.figure3]: Saving montage to "%s"' % f_montage)
    probelist_1e = copy.deepcopy(probelist)

    # Figure 3f
    probelist = copy.deepcopy(probelist_clean)
    matelist = [matelist[0]]*n_subjects
    probelist = [probelist_1a[0]] + [probelist_1b[0]] + [probelist_1c[0]] + [probelist_1d[0]] + [probelist_1e[0]]
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure3f_%d.jpg' % n_subjects, f_saliency=None)
    print('[eccv20.figure3]: Saving montage to "%s"' % f_montage)



def figure4():
    """One mate, top-k nonmates, row-wise by approach"""
    n_subjects = 10
    net = xfr.models.lightcnn.LightCNN_29Layers_v2(num_classes=80013)
    statedict = xfr.models.lightcnn.Load_Checkpoint('../models/LightCNN_29Layers_V2_checkpoint.pth.tar')  
    net.load_state_dict(statedict)    
    wb = xfr.models.whitebox.Whitebox(xfr.models.whitebox.WhiteboxLightCNN(net), ebp_subtree_mode='affineonly_with_prior', eps=1E-16, ebp_version=5)  # FIXME: the version matters
    if not os.path.exists('_vggface2_topk_nonmates.pkl'):
        _vggface2_topk_nonmates(wb, topk=32)  # recompute once
    (matelist, nonmatelist, probelist) = _triplet_mate_frontalpose_nonmate_topk_probe_frontalpose()

    # Detection and color correction
    matelist = [f_detection(im).rgb() for im in matelist]
    nonmatelist = [f_detection(im).rgb() for im in nonmatelist]
    probelist = [[f_detection(im).rgb() for im in iml] for iml in probelist]
    probelist_clean = copy.deepcopy(probelist)

    # Figure 4a
    probelist = copy.deepcopy(probelist_clean)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure4a_%d.jpg' % n_subjects, f_saliency=None)
    probelist_4a = copy.deepcopy(probelist)
       
    # Figure 4b
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_ebp(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure4b_%d.jpg' % n_subjects, f_saliency=f_saliency)
    probelist_4b = copy.deepcopy(probelist)

    # Figure 4c
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_cebp(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure4c_%d.jpg' % n_subjects, f_saliency=f_saliency)
    probelist_4c = copy.deepcopy(probelist)

    # Figure 4d
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_tcebp(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure4d_%d.jpg' % n_subjects, f_saliency=f_saliency)
    probelist_4d = copy.deepcopy(probelist)

    # Figure 4e
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_weighted_subtree(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure4e_%d.jpg' % n_subjects, f_saliency=f_saliency)
    probelist_4e = copy.deepcopy(probelist)

    # Figure 4f
    probelist = copy.deepcopy(probelist_clean)
    matelist = [matelist[0]]*n_subjects
    probelist = [probelist_4a[0]] + [probelist_4b[0]] + [probelist_4c[0]] + [probelist_4d[0]] + [probelist_4e[0]]
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure4f_%d.jpg' % n_subjects, f_saliency=None)
    print('[eccv40.figure4]: Saving montage to "%s"' % f_montage)


def figure5():
    """Same as figure 3, but probe is now repeated"""
    n_subjects = 16
    net = xfr.models.lightcnn.LightCNN_29Layers_v2(num_classes=80013)
    statedict = xfr.models.lightcnn.Load_Checkpoint('../models/LightCNN_29Layers_V2_checkpoint.pth.tar')  
    net.load_state_dict(statedict)    
    wb = xfr.models.whitebox.Whitebox(xfr.models.whitebox.WhiteboxLightCNN(net), ebp_subtree_mode='affineonly_with_prior', eps=1E-16, ebp_version=5)  # FIXME: the version matters
    if not os.path.exists('_vggface2_topk_frontal_nonmates.pkl'):
        _vggface2_topk_frontal_nonmates(wb, topk=32)  # recompute once
    (matelist, nonmatelist, probelist) = _triplet_mate_frontalpose_nonmate_top1_probe_mixedpose(n_subjects)    

    probelist_repeated = []
    for (k,p) in enumerate(probelist):
        probelist_repeated.append([copy.deepcopy(probelist[k][0]) for j in range(0,len(probelist[k]))])
    probelist = probelist_repeated

    # Detection and color correction
    matelist = [f_detection(im).rgb() for im in matelist]
    nonmatelist = [f_detection(im).rgb() for im in nonmatelist]
    probelist = [[f_detection(im).rgb() for im in iml] for iml in probelist]
    probelist_clean = copy.deepcopy(probelist)

    # Figure 5a
    probelist = copy.deepcopy(probelist_clean)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure5a_%d.jpg' % n_subjects, f_saliency=None)
    print('[eccv20.figure5]: Saving montage to "%s"' % f_montage)
    probelist_1a = copy.deepcopy(probelist)
   
    # Figure 5b
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_ebp(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure5b_%d.jpg' % n_subjects, f_saliency=f_saliency)
    print('[eccv20.figure5]: Saving montage to "%s"' % f_montage)
    probelist_1b = copy.deepcopy(probelist)

    # Figure 5c
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_cebp(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure5c_%d.jpg' % n_subjects, f_saliency=f_saliency)
    print('[eccv20.figure5]: Saving montage to "%s"' % f_montage)
    probelist_1c = copy.deepcopy(probelist)

    # Figure 5d
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_tcebp(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure5d_%d.jpg' % n_subjects, f_saliency=f_saliency)
    print('[eccv20.figure5]: Saving montage to "%s"' % f_montage)
    probelist_1d = copy.deepcopy(probelist)

    # Figure 5e
    probelist = copy.deepcopy(probelist_clean)
    f_saliency = lambda im: f_saliency_whitebox_weighted_subtree_lightcnn(wb, im)
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure5e_%d.jpg' % n_subjects, f_saliency=f_saliency)
    print('[eccv20.figure5]: Saving montage to "%s"' % f_montage)
    probelist_1e = copy.deepcopy(probelist)

    # Figure 5f
    probelist = copy.deepcopy(probelist_clean)
    matelist = [matelist[0]]*n_subjects
    probelist = [probelist_1a[0]] + [probelist_1b[0]] + [probelist_1c[0]] + [probelist_1d[0]] + [probelist_1e[0]]
    f_montage = _triplet_montage(wb, matelist, nonmatelist, probelist, 'figure5f_%d.jpg' % n_subjects, f_saliency=None)
    print('[eccv20.figure5]: Saving montage to "%s"' % f_montage)



