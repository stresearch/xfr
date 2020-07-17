This directory contains models based on PyTorch [1] implementation (importing the Caffe models with the tool [3]).

MTCNN [2] is used for face detection. This bounding box is then extended by a factor 0.3 (except the extension outside image) to include the whole head, 
which is used as the input for networks (it's worth noting that this version is a bit tighter than the released loosely cropped version where the bounding box is extended by a factor 1.0).

Durining training, a region of 224x224 pixels is randomly cropped from each input, whose shorter size is resized to 256. 
The mean value of each channel is substracted for each pixel (mean vector [91.4953, 103.8827, 131.0912] in BGR order). More details can be found in the paper of VGGFace2.

Models:

senet50_256: A 256-D dimensionality-reduction layer stacking at the final global-average pooling layer on SE-ResNet-50 [4] model.

References for implementation:

[1] PyTorch: https://pytorch.org

[2] MTCNN: https://github.com/kpzhang93/MTCNN_face_detection_alignment

[3] MMdnn: https://github.com/Microsoft/MMdnn

[4] Squeeze-and-Excitation Networks: https://github.com/hujie-frank/SENet

Download:

This model was downloaded on Jan 1, 2020 from:
http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/pytorch/senet50_256_pytorch.tar.gz
