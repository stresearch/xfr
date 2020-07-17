import os
import torch
import sys
import xfr
from xfr import xfr_root
from xfr.models import whitebox
from xfr.models import resnet
import warnings

def create_wbnet(net_name, device, ebp_version=None, ebp_subtree_mode=None):
    """ Helper function for included networks.

        The best ebp_subtree_mode can depend on the network. If set to None,
        it should be set to the best value.
    """
    # python3 pytorch code
    if device is None:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available()
            else "cpu")

    if ebp_version is not None and ebp_version < 4:
        raise DeprecationWarning('EBP version must be >= 4')

    if net_name == 'resnetv6_pytorch':

        if ebp_subtree_mode is None:
            ebp_subtree_mode = 'norelu'

        resnet_path = os.path.join(xfr_root,
                                   'models/resnet101_l2_d512_twocrop.pth')
        model = resnet.resnet101v6(resnet_path, device)

        model.to(device)
        wbnet = whitebox.WhiteboxSTResnet(
            model,
        )
        net = whitebox.Whitebox(
            wbnet,
            ebp_subtree_mode=ebp_subtree_mode,
            ebp_version=ebp_version,
        ).to(device)

        net.match_threshold = 0.9636
        net.platts_scaling = 15.05

        return net

    elif net_name == 'resnetv4_pytorch':

        if ebp_subtree_mode is None:
            ebp_subtree_mode = 'norelu'

        resnet_path = os.path.join(xfr_root,
                                   'models/resnet101v4_28NOV17_train.pth')
        model = resnet.resnet101v6(resnet_path, device)

        model.to(device)
        wbnet = whitebox.WhiteboxSTResnet(
            model,
        )
        wb = whitebox.Whitebox(
            wbnet,
            ebp_subtree_mode=ebp_subtree_mode,
            ebp_version=ebp_version,
        ).to(device)

        # From Caffe ResNetv4:
        # wb.match_threshold = 0.9252
        # wb.platts_scaling = 17.71
        wb.match_threshold = 0.9722
        wb.platts_scaling = 16.61
        return wb

    elif net_name == 'vggface2_resnet50':
        if ebp_subtree_mode is None:
            ebp_subtree_mode = 'norelu'
        if ebp_version is not None:
            warnings.warn('ebp_version %s is ignored for %s' % (
                ebp_version,
                net_name,
            ))
        sys.path.append(os.path.join(
            xfr_root, 'models/resnet50_128_pytorch'))
        import resnet50_128
        param_path = os.path.join(
            xfr_root, 'models/resnet50_128_pytorch/resnet50_128.pth')

        net = resnet50_128.resnet50_128(param_path)
        net.to(device)
        wb = xfr.models.whitebox.Whitebox(
            xfr.models.whitebox.Whitebox_resnet50_128(net),
            ebp_subtree_mode=ebp_subtree_mode,
            ebp_version=ebp_version,
        ).to(device)

        wb.match_threshold = 0.896200
        wb.platts_scaling = 15.921608

        return wb

    elif net_name == 'lightcnn':
        if ebp_subtree_mode is None:
            ebp_subtree_mode = 'affineonly_with_prior'
        # if ebp_version is not None:
        #     warnings.warn('ebp_version %s is ignored for %s' % (
        #         ebp_version,
        #         net_name,
        #     ))
        if not os.path.exists(os.path.join(
            xfr_root, 'models/LightCNN_29Layers_V2_checkpoint.pth.tar')
        ):
            raise RuntimeError(
                'Missing light-CNN model file. Download file '
                '"LightCNN_29Layers_V2_checkpoint.pth.tar" from '
                'https://github.com/AlfredXiangWu/LightCNN and copy into '
                'explainable_face_recognition/models/LightCNN_29Layers_V2_checkpoint.pth.tar')
            return None
        net = xfr.models.lightcnn.LightCNN_29Layers_v2(num_classes=80013)
        statedict = xfr.models.lightcnn.Load_Checkpoint(os.path.join(
            xfr_root, 'models/LightCNN_29Layers_V2_checkpoint.pth.tar'))
        net.load_state_dict(statedict)
        net.to(device)
        wb = xfr.models.whitebox.Whitebox(
            xfr.models.whitebox.WhiteboxLightCNN(net),
            ebp_subtree_mode=ebp_subtree_mode,
            ebp_version=ebp_version,
        ).to(device)
        wb.match_threshold = 0.829200
        wb.platts_scaling = 10.877741
        return wb

    else:
        raise NotImplementedError(
            'create_wbnet does not implemented network "%s"' %
            net_name
        )
