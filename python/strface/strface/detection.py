# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.



import numpy as np
import os
import numpy as np
import imp
import torch 
import torchvision.ops
import torch.nn as nn
import torch.nn.functional

import time
import os
import sys
from math import ceil
import cv2  # FIXME: remove and replace with PIL
import torch
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'detection'))
from config import cfg

DIM_THRESH = 15
CONF_THRESH = 0.5
NMS_THRESH = 0.15
FUSION_THRESH = 0.60
VERBOSE = False

def log_info(s):
    if VERBOSE:
        print(s)

class FasterRCNN_Network(nn.Module):
    """PyTorch-1.3 model conversion of ResNet-101_faster_rcnn_ohem_iter_20000.caffemodel, leveraging MMDNN conversion tools"""
    def __init__(self, model_dir, device):
        super(FasterRCNN_Network, self).__init__()

        self.device = device
        torch.device(device)
        # Converted using convert_caffe_to_pytorch.convert_top()
        dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'detection')
        MainModel = imp.load_source('MainModel', os.path.join(dir, "top_layers.py"))
        self.top = torch.load(os.path.join(model_dir, "top_layers.pth"), map_location=device)
        self.top.eval()
        self.top = self.top.to(device)

        # Converted using convert_caffe_to_pytorch.convert_bottom()
        MainModel = imp.load_source('MainModel', os.path.join(dir, "bottom_layers.py"))
        self.bottom = torch.load(os.path.join(model_dir, "bottom_layers.pth"), map_location=device)
        self.bottom.eval()
        self.bottom = self.bottom.to(device)

        # Converted using convert_caffe_to_pytorch.convert_rpn()
        MainModel = imp.load_source('MainModel', os.path.join(dir, "rpn_layers.py"))
        self.rpn = torch.load(os.path.join(model_dir, "rpn_layers.pth"), map_location=device)
        self.rpn.eval()
        self.rpn = self.rpn.to(device)


        # Proposal layer:  manually imported from janus/src/rpn
        self._feat_stride = 16
        #self._anchors = rpn.generate_anchors.generate_anchors(scales=np.array( (8,16,32) ))    
        self._anchors = np.array([[ -84.,  -40.,   99.,   55.],
                                  [-176.,  -88.,  191.,  103.],
                                  [-360., -184.,  375.,  199.],
                                  [ -56.,  -56.,   71.,   71.],
                                  [-120., -120.,  135.,  135.],
                                  [-248., -248.,  263.,  263.],
                                  [ -36.,  -80.,   51.,   95.],
                                  [ -80., -168.,   95.,  183.],
                                  [-168., -344.,  183.,  359.]])
        self._num_anchors = self._anchors.shape[0]
        
    def __call__(self, im, im_info):
        # im is a tensor, N x 3 x H x W; im_info is another,
        # N x 3 (H, W, scale)
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            im = im.to(self.device)
            
            res4b22 = self.bottom(im)
            
            (rpn_cls_score, rpn_bbox_pred) = self.rpn(res4b22)
            
            (N,C,W,H) = rpn_cls_score.shape
            rpn_cls_score_reshape = torch.reshape(rpn_cls_score, (N, 2, -1, H))
            del rpn_cls_score
            rpn_cls_prob = torch.nn.functional.softmax(rpn_cls_score_reshape, dim=1)  # FIXME: is this dim right?
            rpn_cls_prob_reshape = torch.reshape(rpn_cls_prob, (N, 18, -1, H))
            del rpn_cls_prob
            
            # TODO: Make this handle multiple images, instead of horrible flaming death.
            rois = self._proposal_layer(rpn_cls_prob_reshape.cpu(), rpn_bbox_pred.cpu(), im_info)
            del rpn_bbox_pred
            # import pdb; pdb.set_trace()
            rois_gpu = rois.to(self.device)
            roi_pool5 = torchvision.ops.roi_pool(res4b22, rois_gpu, (14,14), 0.0625)
            del res4b22
            del rois_gpu
            (bbox_pred_1, cls_prob, cls_score) = self.top(roi_pool5)
            rois_cpu = rois.cpu()
            del rois
            bbox_pred_1_cpu = bbox_pred_1.cpu()
            del bbox_pred_1
            cls_prob_cpu = cls_prob.cpu()
            del cls_prob
            cls_score_cpu = cls_score.cpu()
            del cls_score
        return (rois_cpu, bbox_pred_1_cpu, cls_prob_cpu, cls_score_cpu)

    def _proposal_layer(self, rpn_cls_prob_reshape, rpn_bbox_pred, im_info):
        """rpn.proposal_layer"""
        # 'Only single item batches are supported'
        assert(rpn_cls_prob_reshape.shape[0] == 1) 

        # TODO: Defaults from caffe: make me configurable 
        #   {'PROPOSAL_METHOD': 'selective_search', 'SVM': False, 'NMS': 0.3, 'RPN_NMS_THRESH': 0.7, 'SCALES': [800], 
        #   'RPN_POST_NMS_TOP_N': 300, 'HAS_RPN': False, 'RPN_PRE_NMS_TOP_N': 6000, 'BBOX_REG': True, 'RPN_MIN_SIZE': 3, 'MAX_SIZE': 1333}
        cfg_key = 'TEST'      # either 'TRAIN' or 'TEST'
        pre_nms_topN = 6000   # RPN_PRE_NMS_TOP_N
        post_nms_topN = 300   # RPN_POST_NMS_TOP_N
        nms_thresh= 0.7       # RPN_NMS_THRESH
        min_size = 3          # RPN_MIN_SIZE                

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = rpn_cls_prob_reshape.detach().numpy()[:, self._num_anchors:, :, :]
        bbox_deltas = rpn_bbox_pred.detach().numpy()
        (im_height, im_width, im_scale) = im_info[0]  # H, W, scale

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = self._bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = self._clip_boxes(proposals, (im_height.item(), im_width.item()))

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = self._filter_boxes(proposals, min_size * im_scale.item())
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = self._nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        return torch.tensor(blob)


    def _bbox_transform_inv(self, boxes, deltas):
        """Cloned from janus-tne/src/python/fast_rcnn.bbox_transform.bbox_transform_inv"""
        if boxes.shape[0] == 0:
            return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

        boxes = boxes.astype(deltas.dtype, copy=False)

        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        
        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]
        
        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]

        pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
        
        return pred_boxes

    def _clip_boxes(self, boxes, im_shape):
        """Cloned from janus-tne/src/python/fast_rcnn.bbox_transform.clip_boxes"""
        # x1 >= 0
        boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes


    def _filter_boxes(self, boxes, min_size):
        """Cloned from janus-tne/src/python/rpn.proposal_layer._filter_boxes"""
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep


    def _nms(self, dets, thresh):
        """Cloned from janus-tne/src/python/nms/py_cpu_nms.py"""
        """FIXME: GPU acceleration needed?"""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        
        return keep


class FasterRCNN(object):
    "Wrapper for PyTorch RCNN detector"
    def __init__(self, model_dir, gpu_index, conf_threshold, rotate_flags,
                 rotate_thresh, fusion_thresh, test_scales, max_size):
        # This logs the contents of the detector_params dict, along with the other values that we passed.
        #log_info(f"Params=[{', '.join((chr(34) + k + chr(34) + ': ' + str(v)) for k, v in detector_params)}], threshold=[{conf_threshold}], "
        #         "rotate=[{rotate_flags}], rotate_thresh=[{rotate_thresh}], fusion_thresh=[{fusion_thresh}]")
        log_info(f"model=[{model_dir}], gpu=[{gpu_index}], threshold=[{conf_threshold}], "
                 "rotate=[{rotate_flags}], rotate_thresh=[{rotate_thresh}], fusion_thresh=[{fusion_thresh}]")
        # Now do any setup required by the parameters that the framework
        # itself won't handle.
        # import pdb; pdb.set_trace()
        if gpu_index >= 0:
            dev = torch.device(gpu_index)
            cfg.GPU_ID = gpu_index
        else:
            dev = torch.device("cpu")

        cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        cfg.TEST.SCALES = (test_scales,)
        cfg.TEST.MAX_SIZE = max_size
        self.net = FasterRCNN_Network(model_dir, dev)
        if conf_threshold is None:
            self.conf_threshold = CONF_THRESH
        else:
            self.conf_threshold = conf_threshold
        if rotate_flags is None:
            self.rotate_flags = 0
        else:
            self.rotate_flags = rotate_flags
        if rotate_thresh is None:
            self.rotate_thresh = conf_threshold
        else:
            self.rotate_thresh = rotate_thresh
        if fusion_thresh is None:
            self.fusion_thresh = FUSION_THRESH
        else:
            self.fusion_thresh = fusion_thresh
        log_info('Init success; threshold {}'.format(self.conf_threshold))

    def __call__(self, img, padding=0, min_face_size=DIM_THRESH):
        return self.detect(img, padding=padding, min_face_size=min_face_size)

    def detect(self, image, padding=0, min_face_size=DIM_THRESH):
        "Run detection on an image, with specified padding and min size"
        start_time = time.time()
        # import pdb; pdb.set_trace()
        width = image.shape[1]
        height = image.shape[0]
        # These values will get updated for resizing and padding, so we'll have good numbers
        # for un-rotating bounding boxes where needed
        detect_width = width
        detect_height = height
        color_space = 1 if image.ndim > 2 else 0

        log_info('w/h/cs: %d/%d/%d' %(width, height, color_space))

        img = np.array(image)

        if padding > 0:
            perc = padding / 100.
            padding = int(ceil(min(width, height) * perc))

            # mean bgr padding
            bgr_mean = np.mean(img, axis=(0, 1))
            detect_width = width + padding * 2
            detect_height = height + padding * 2
            pad_im = np.zeros((detect_height, detect_width, 3), dtype=np.uint8)
            pad_im[:, :, ...] = bgr_mean
            pad_im[padding:padding + height, padding:padding + width, ...] = img
            img = pad_im
            log_info('mean padded to w/h: %d/%d' % (img.shape[1], img.shape[0]))
            # cv2.imwrite('debug.png', im)

        if width <= 16 or height <= 16:
            img = cv2.resize(img, (32, 32))
            width = img.shape[1]
            height = img.shape[0]

        rotation_angles = []
        if (self.rotate_flags & 1) != 0:
            rotation_angles.append(90)
        if (self.rotate_flags & 2) != 0:
            rotation_angles.append(-90)
        if (self.rotate_flags & 4) != 0:
            rotation_angles.append(180)
        current_rotation = 0

        # parallel arrays: one is list of boxes, per rotation; other is list of scores
        det_lists = []
        box_proposals = None
        im_rotated = img
        while True:
            # global img_num
            # cv2.imwrite('debug_%08d.png'%img_num, im)
            # img_num += 1

            scores, boxes = im_detect(self.net, im_rotated, box_proposals)

            # Threshold on score and apply NMS
            cls_ind = 1
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]

            # Each row of dets is Left, Top, Right, Bottom, score
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            orig_dets = dets.shape
            #keep = nms(dets, NMS_THRESH, force_cpu=False)
            keep = self.net._nms(dets, NMS_THRESH)  # JEBYRNE
            dets = dets[keep, :]
            new_dets = dets.shape
            log_info('Before NMS: {}; after: {}'.format(orig_dets, new_dets))

            # If we just ran the detector on a rotated image, use the rotation threshold
            if current_rotation != 0:
                keep = np.where(dets[:, 4] > self.rotate_thresh)
            else:
                keep = np.where(dets[:, 4] > self.conf_threshold)
            # print 'After filter for rotation {}: keep = {}'.format(current_rotation, keep)
            dets = dets[keep]

            # This is converting the max coords to width and height. The coordinates haven't been
            # unrotated yet--save a bit of energy by thresholding and such first.
            dets[:, 2] = dets[:, 2] - dets[:, 0] + 1
            dets[:, 3] = dets[:, 3] - dets[:, 1] + 1
            if current_rotation != 0:
                # Now unrotate
                # Rotated coordinates are x_rot, y_rot, Wr, Hr
                # Unrotated, X, Y, W, H
                # for +90, width and height swap, top right becomes top left
                #   W = Hr, H = Wr, X = y_rot, Y = (rotated image width) - (x_rot + Wr)
                # for -90, width and height swap, bottom left becomes top left
                #   W = Hr, H = Wr, X = (rotated image height) - (y_rot + Hr), Y = x_rot
                # for 180, width and height same, bottom right becomes top left
                #   W = Wr, H = Hr, X = image width - (x_rot + Wr), Y = image height - (y_rot + Hr)
                if current_rotation == 90:
                    for det in dets:
                        x_rot = det[0]
                        y_rot = det[1]
                        det[0] = y_rot
                        # Image was rotated, so width and height swapped
                        det[1] = detect_height - (x_rot + det[2])
                        det[2], det[3] = det[3], det[2]
                elif current_rotation == -90:
                    for det in dets:
                        x_rot = det[0]
                        y_rot = det[1]
                        # Image was rotated, so width and height swapped
                        det[0] = detect_width - (y_rot + det[3])
                        det[1] = x_rot
                        det[2], det[3] = det[3], det[2]
                elif current_rotation == 180:
                    for det in dets:
                        x_rot = det[0]
                        y_rot = det[1]
                        det[0] = detect_width - (x_rot + det[2])
                        det[1] = detect_height - (y_rot + det[3])

            if padding > 0:
                # Adjust to original coordinates
                dets[:, 0] -= padding
                dets[:, 1] -= padding

                keep = np.where(np.bitwise_and(dets[:, 2] > min_face_size,
                                               dets[:, 3] > min_face_size))
                dets = dets[keep]
            else:
                keep = np.where(np.bitwise_and(dets[:, 2] > min_face_size,
                                               dets[:, 3] > min_face_size))
                dets = dets[keep]
            det_lists.append(dets)
            # Exit the list if we've done all the rotations we need
            if len(rotation_angles) == 0:
                break
            current_rotation = rotation_angles[0]
            rotation_angles = rotation_angles[1:]
            log_info('Rotating to %d' % current_rotation)
            if current_rotation == 90:
                im_rotated = cv2.transpose(img)
                im_rotated = cv2.flip(im_rotated, flipCode=1)
            elif current_rotation == -90:
                im_rotated = cv2.transpose(img)
                im_rotated = cv2.flip(im_rotated, flipCode=0)
            else:
                # Must be 180
                im_rotated = cv2.flip(img, flipCode=-1)

                # Now have 1, 3 (0, 90, -90), or 4 (0, 90, -90, 180) elements of det_lists.
        if len(det_lists) > 1:
            return self.select_from_rotated(det_lists, start_time)
        else:
            dets = det_lists[0]
            log_info('Found %d faces' % dets.shape[0])
            # print dets
            log_info('===elapsed %.6f===' % ((time.time() - start_time) * 1000))
            return dets

    def select_from_rotated(self, det_lists, start_time):
        "Given that we tried rotating the image, select the best rotation to use"
        dets = det_lists[0]
        original_dets = dets.shape[0]
        i = 0
        for rot_dets in det_lists[1:]:
            i = i + 1
            log_info('Processing rotated detections from slot %d' % (i))
            # Now iterate over the rows, 1/detection
            for rot_det in rot_dets:
                rot_xmin = rot_det[0]
                rot_ymin = rot_det[1]
                rot_xmax = rot_xmin + rot_det[2]
                rot_ymax = rot_ymin + rot_det[3]
                rot_area = rot_det[2] * rot_det[3]
                matched = False
                best_iou = 0.0
                for det in dets:
                    xmin = det[0]
                    ymin = det[1]
                    xmax = xmin + det[2]
                    ymax = ymin + det[3]
                    intersection_width = min(xmax, rot_xmax) - max(xmin, rot_xmin)
                    intersection_height = min(ymax, rot_ymax) - max(ymin, rot_ymin)
                    if intersection_width > 0 and intersection_height > 0:
                        intersection_area = intersection_width * intersection_height
                        union_area = rot_area + det[2] * det[3] - intersection_area
                        iou = intersection_area / union_area
                        if iou > best_iou:
                            best_iou = iou
                        if iou > self.fusion_thresh:
                            matched = True
                            if rot_det[4] > det[4]:
                                # Rotated detection was better
                                det[0] = rot_det[0]
                                det[1] = rot_det[1]
                                det[2] = rot_det[2]
                                det[3] = rot_det[3]
                                det[4] = rot_det[4]
                            break
                if not matched:
                    # Add this guy, since he had no matches
                    dets = np.vstack((dets, rot_det))
        log_info('Found %d face%s (orig %d)' %
                 (dets.shape[0], '' if dets.shape[0] == 0 else 's', original_dets))
        log_info('===elapsed %.6f===' % ((time.time() - start_time) * 1000))
        return dets

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a torch Tensor holding the image. Some transposition might have to occur, because
        we need N, 3, 800, 1205 (say), while the image itself is likely 800, 1205, 3. N is the number of images
        to process (if len(TEST.SCALES) > 1, then it won't be 1).
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        #log_info('Add %s from %s' % (im.shape, im_orig.shape))
        im_scale_factors.append(im_scale)
        # We need number of channels first, then height, then width
        im_transpose = im.transpose(2, 0, 1)
        processed_ims.append(im)

    # Create a tensor to hold the input images. Typically this will be
    # 1, 3, ..., 
    #blob = torch.Tensor(im_list_to_blob(processed_ims))
    blob = torch.Tensor(np.array(processed_ims).transpose([0,3,1,2]))  # JEBYRNE
    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (pytorch): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order, as (H, W, C)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    im_blob, im_scales = _get_image_blob(im)

    im_info = torch.Tensor(np.array([[im_blob.shape[2], im_blob.shape[3], im_scales[0]]], dtype=np.float32))

    # We think these are already the right shape?
    # # Now ready to supply inputs to network.
    # # reshape network inputs
    # net.blobs['data'].reshape(*(blobs['data'].shape))
    # if cfg.TEST.HAS_RPN:
    #     net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    # else:
    #     net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    # Returns are all on CPU
    (rois, bbox_pred, cls_prob, cls_score) = net(im_blob, im_info)
    del im_blob
    del im_info
    # gc.collect(2)
    # torch.cuda.empty_cache()

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = rois.detach().numpy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = cls_score.detach().numpy()
    else:
        # use softmax estimated probabilities
        scores = cls_prob.detach().numpy()

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.detach().numpy()
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes

def find_cuda_objs():
    objs = gc.get_objects()
    cuda_objs = []
    for obj in objs:
        try:
            dev = obj.device
            cuda_objs.append(obj)
        except:
            continue
    return cuda_objs



import numpy as np

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
