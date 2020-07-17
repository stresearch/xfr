# Copyright 2019 Systems & Technology Research, LLC
# Use of this software is governed by the license.txt file.



import torch
import numpy as np
import PIL
import strface.detection

def image():
    im = np.array(PIL.Image.open('1000a.jpg'))
    net = strface.detection.FasterRCNN(model_dir='../models/detection', gpu_index=0, conf_threshold=None, rotate_flags=None, rotate_thresh=None, fusion_thresh=None, test_scales=800, max_size=1300)
    print(net(im))
    
