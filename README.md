# Explainable Face Recognition (XFR) Project

* Project website: [https://stresearch.github.io/xfr](https://stresearch.github.io/xfr)
* Code and dataset repository: [https://github.com/stresearch/xfr](https://github.com/stresearch/xfr)

Explainable face recognition is the problem of providing an interpretable reasoning for the outputs of a face recognition system.  This software distribution accompanies the arXiv paper:

J. Williford, B. May and J. Byrne, "Explainable Face Recognition", ECCV 2020 [(arXiv link TBD)](http://arxiv.org)

In this paper, we provide the first comprehensive benchmark for explainable face recognition (XFR), and make the following contributions:

* Discriminative saliency maps.  We introduce benchmark algorithms for discriminative visualization for both whitebox (convolutional network exposed) and blackbox (matching scores only) face recognition systems.  
* Inpainting game protocol.  We define a standardized evaluation protocol for fine grained discriminative visualization of faces.
* Inpainting game dataset for facial recognition.  We publicly release the inpainting game dataset, algorithms and baseline results to support reproducible research.


# Installation
Tested with Python 3.6, PyTorch 1.3.   We recommend installation in a python-3.6 virtual environment.

```python
pip install scipy numpy pillow torch torchvision scikit-image pandas dill vipy opencv-python jupyter easydict
```

Add the python directory to your PYTHONPATH

```export 
PYTHONPATH=$PYTHONPATH:/path/to/explainable_face_recognition/python/
```

Note: if you are using an older GPU (e.g., a K40), see https://github.com/NVIDIA/apex/issues/605.
There are apparently cases where PyTorch will generate code for the wrong GPU, which can lead
to "CUDA error: no kernel image is available for execution on the device" errors in both
the whitebox demo and some of the black box notebooks.

# Whitebox Explainable Face Recognition 

## Networks

The whitebox explainable face recognition requires a pytorch convolutional network trained for face matching.  In this distribution, we include support for three systems:

* Resnet-101+L2 normalization.  This is an legacy version of the STR-Janus system.
* Resnet-50-128d.  This is the the University of Oxford trained [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2) system.
* Light CNN.  This is a publicly available [Light-CNN](https://github.com/AlfredXiangWu/LightCNN) face matcher based on the max feature maps.

All of the networks are bundled with the distribution, with the exception of Light-CNN which must be downloaded separately for non-commercial use. 

[LightCNN-29 v2 download](https://drive.google.com/open?id=1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS)

To run the (optional) whitebox demos with this network, download and copy this file to:

```python
./models/LightCNN_29Layers_V2_checkpoint.pth.tar
```

## Demos

```python
cd demo    
python test_whitebox.py
```

This distribution comes with data for 95 subjects, packed as tarballs. To run most
of the whitebox cases, including test_whitebox.py, you will need to unpack those first.
See data/inpainting-game/README and unpack-aligned.sh for details.

This will generate the following subset of visualization outputs in CPU-only mode:

1. Excitation Backprop and Triplet EBP
2. Contrastive Excitation Backprop and Triplet Contrastive EBP
3. Truncated Contrastive Excitation Backprop and Triplet Truncated Contrastive EBP
4. Weighted subtree triplet EBP 
5. Excitation backprop for Light-CNN
6. Excitation backprop for VGGFace2 Resnet-50-128d

You can compare the outputs of running this demo with the cached result in demo/whitebox/*.jpg.

## Extending

Creating your own whitebox visualization on a new network requires implementing the following.  Your network must be a torch network, and you must create a custom class that subclasses WhiteboxNetwork() in python/xfr/models/whitebox.py.  You must provide an implementation of all of the methods of this class for your network.  Once you create a custom whitebox network for your model, then this can be passed to the constructor of Whitebox(), which exercises the API for the whitebox network to compute whitebox saliency maps for your network.  See the comments in WhiteboxNetwork() methods for description of the needed functionality.

Whitebox networks have the following limitations.  The whitebox network must satisfy the assumptions A1 and A2 in section 3.2 of "Top-down Neural Attention by Excitation Backprop" (ECCV'16):

<https://arxiv.org/pdf/1608.00507.pdf>

Furthermore, the current implementation does not support sigmoid, Tanh or ELU non-linear layers.  Furthermore, in order to implement the excitation backpropagation functionality, we leverage the use of torch hooks:

<https://pytorch.org/docs/stable/nn.html#torch.nn.Module.register_forward_hook>
<https://pytorch.org/docs/stable/nn.html#torch.nn.Module.register_forward_pre_hook>

Hooks are callbacks that are called every time after (or just before) forward() has computed an output for a module.  This functionality allows for custom modification of layer outputs to implement excitation backprop, and to register a tensor hook:

<https://pytorch.org/docs/stable/tensors.html#torch.Tensor.register_hook>

which are called every time a gradient with respect to the Tensor is computed.  The combination of these hooks allow for the creation of saliency maps rather than encodings.  This requirement means that if a layer includes a built in operation (e.g. A+B for tensors A and B within the forward() method for a layer), and it is desired to explore the output of these layers, then these operations must be exposed as a custom layer.  See Add() in python/xfr/models/resnet.py for an example.  Furthermore, if a layer is a container (e.g. nn.Sequential, nn.ModuleList, nn.ModuleDict), then this layer should be recursively enumerated into the layers within the container that expose forward operations.   See WhiteboxNetwork._layer_visitor in python/xfr/models/whitebox.py for an example of the recursive visiting order of layers in a whitebox.  This layer visitor can be customized to enumerate the order in which layers are visited for computing the whitebox visualization.  


# Blackbox Explainable Face Recognition

## Running the Jupyter Notebook Demos
### Starting the Jupyter Server
From the `demo/` directory, run
    
    jupyter notebook --no-browser --port XXXX
    
replacing XXXX with the port number you would like use to run the server.

### Accessing the Notebooks
Copy/paste the URL running the above command prints into your browser, replacing the local ip address with the external ip address of the server (`localhost` if running on a local machine)

Pick one of the "Subject" demos, i.e.:

* Open the notebook file `blackbox_demo_Subject_0.ipynb`
* Run the notebook to compute contrastive saliency maps
* Select `File: Close and Halt` before running another notebook

NOTE: Previously computed results are embedded in the notebooks if you want to avoid recomputing them.

## Advanced Black Box Usage
### Specifying a Custom Dataset
The most straightforward way to specify your own imagery for the probe, reference, and gallery is to pass in a filepath for the probe, and a list of filepaths for the reference(s) and gallery to `STRise()`.

Alternatively, the probe can be a numpy array and the reference(s) and gallery can be a list of numpy arrays.

### Specifying a Custom Scoring Function
To use a custom scoring function to replace ResNet as the black box, pass in the custom function as an argument to `STRise()` by setting `black_box_fn=custom_fn` and removing `black_box='resnetv6_pytorch'`. The interface for this `custom_fn` is as follows:

    scores = custom_fn(images_a, images_b)

Where `images_a` is:
* a list of numpy arrays of images

and `images_b` corresponds to the type of the reference(s) and gallery specified as in the custom dataset:

* a list of filepaths to the images
* a list of numpy arrays of images

This function must then be able to compute pairwise similarity scores between the images from `images_a` and `images_b`, where `scores` is a numpy array with dimensions (`len(images_a)`, `len(images_b)`).

See the notebook file `blackbox_demo_pittpatt.ipynb` for an example of defining and using a custom scoring function for the PittPatt black box system. For this demo to work, the `pittpatt_apps_root` and `pittpatt_sdk_root` variables in the `pittpatt_bb_fn()` should be modified to point to the local paths for the `PittPattApps` and `PittPattSDK` directories respectively.


# Inpainting Game

Included with this software deliverable is a dataset and protocol for evaluating performance for an explainable face recognition algorithm for the STR ResNet (v4 and v6) networks. Custom networks will be supported in future releases. There are two stages of the running the inpainting game:

1. Generate the saliency maps for all of the combinations of probes and mates and inpainting references for the network being tested.
2. Run the inpainting game analysis and plot the results.

The first stage is expected to take about 36 hours for the white box methods and 18 hours for the RISE-based blackbox method using a single Titan X. The second stage is much faster, but can take a couple of hours. Both methods save their results and do not recompute unless the '--overwrite' parameter is passed.

## Generating the saliency maps for the Inpainting Game

The saliency maps for the whitebox resnetv4_pytorch on a single gpu or on cpu is as follows:

```python
python eval/generate_inpaintinggame_wb_saliency_maps_multigpu.py --net resnetv4_pytorch
```

For command line help, refer to:

```python
python eval/generate_inpaintinggame_wb_saliency_maps_multigpu.py --help
```

The saliency maps for the blackbox resnet4_pytorch on a single gpu or on cpu is as follows:
```python
python eval/generate_inpaintinggame_bb_saliency_maps_multigpu.py --net resnetv4_pytorch
```

For command line help, refer to:

```python
python eval/generate_inpaintinggame_bb_saliency_maps_multigpu.py --help
```

## Evaluating and plotting the Inpainting Game

The inpainting game can be evaluated and plotted as follows:

```python
ipython --pdb -- eval/run_inpainting_game_eval.py --cache-dir /temp
```

For command line help, refer to:
```python
ipython --pdb -- eval/run_inpainting_game_eval.py --help
```

## Evaluating new networks

The easiest way to add new methods is to add them to the create_wbnet function in ./eval/create_wbnet.py. See create_wbnet for examples. The parameters match_threshold and platts_scaling should be defined for the inpainting game. These values can be calculated by calling `.eval/calculate_subject_dists_inpaintinggame.py`, for example:

```python
python eval/calculate_subject_dists_inpaintinggame.py --ijbc-path /path/to/IJB-C/ --net vggface2_resnet50
```

and then `calculate_net_match_threshold`:

```python
python eval/calculate_net_match_threshold.py --net vggface2_resnet50
```

You will see a result like the following:

  Net vggface2_resnet50 threshold=0.896200, platt's scaling=15.921608

Set these values in the Whitebox object, for example, via `create_wbnet` function in `eval/create_wbnet.py` for the corresponding network:

```python
  wb.match_threshold = 0.8962
  wb.platts_scaling = 15.92
```

The inpainting game dataset needs to be filtered to include only the inpaintings that sufficiently change the network. An example of how to do this is in `eval/filter_inpaintinggame_for_net.py`. This file can be used if the network has been added to `create_wbnet.py`.

Help can be found with the `--help` argument for both commands.  After these steps, the inpainting game instructions above can be used.


# License

The following notice covers the Systems & Technology Research, LLC.
Explainable Face Recognition software delivery.  

> This software is provided to the United States Government (USG) under 
> contract number 2019-19022600003 with Unlimited Rights as defined at 
> Federal Acquisition Regulation 52.227-14, “Rights in Data-General” 
> (May 2014)("Unlimited rights" means the rights of the Government to 
> use, disclose, reproduce, prepare derivative works, distribute copies 
> to the public, and perform publicly and display publicly, in any manner 
> and for any purpose, and to have or permit others to do so”).
 
> Copyright (C) 2020 Systems & Technology Research, LLC.
> http://www.systemstechnologyresearch.com


The following notice covers the Light-CNN software:  ./python/xfr/models/lightcnn.py
 

> MIT License
>
> Copyright (c) 2017 Alfred Xiang Wu

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:

> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.

> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

The images in the inpainting-game dataset are a subset of the IJB-C dataset and
are covered by the IJB-C and [Creative Common licenses](https://creativecommons.org/licenses/):

> This product contains or makes use of the following data made available by the Intelligence Advanced Research Projects Activity (IARPA): IARPA Janus Benchmark C (IJB-C) data detailed at [Face Challenges homepage](https://www.nist.gov/programs-projects/face-challenges).

The following notice covers the face detector bundled with this distribution.

The face detector was originally trained by University of Massachusetts, Amherst and converted to PyTorch by the authors.  If you use this software, please consider citing:

H. Jiang and E. Learned-Miller, "Face Detection with the Faster R-CNN" in FG (IEEE Conference on Automatic Face and Gesture Recognition) 2017.

# Acknowledgement

This research is based upon work supported by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA) under contract number 2019-19022600003. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of ODNI, IARPA, or the U.S. Government.  The U.S. Government is authorized to reproduce and distribute reprints for Governmental purpose notwithstanding any copyright annotation thereon.
