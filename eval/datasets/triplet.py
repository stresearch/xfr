import os
import pandas as pd

import torch
import torch.utils.data as data
import torchvision.datasets.folder

class TripletDataLoader(data.Dataset):
    """ Load images from a CSV file containing:
        1. a filepath
        2. a class label
        3. roi min x
        4. roi min y
        5. roi width
        6. roi height
    """

    def __init__(self,
                 data_file_p,
                 loader=torchvision.datasets.folder.default_loader,
                 transform=None,
                 data_root=None,
                 return_file_info=False,
                 **kwargs
                 ):
        # super().__init__(**kwargs)
        assert data_root is not None
        self.data_root = data_root
        self.data_file_p = data_file_p
        self.transform = transform
        assert not isinstance(self.transform, str)
        self.loader = loader

        with open(self.data_file_p, 'r') as ff:
            ds = pd.read_csv(
                ff,
            )
        assert ds.shape[0] > 0, '%s was empty!' % self.data_file_p

        self.probe_ds = ds[ds['TRIPLET_SET']=='PROBE']
        self.ref_ds = ds[ds['TRIPLET_SET']=='REF']
        self.ref_ds = self.ref_ds.set_index(keys=['SUBJECT_ID', 'MASK_ID'])
        self.return_file_info = return_file_info

    def shuffle(self):
        self.probe_ds = self.probe_ds.sample(frac=1)

    def load_image(self, column_path, data):
        """
            data - pandas Series (DataFrame row).
        """
        path = data[column_path]
        if os.path.isabs(path):
            img = self.loader(path)
        elif not isinstance(self.data_root, str):
            # data_root is a list
            img = None
            for root in self.data_root:
                try:
                    img = self.loader(os.path.join(root, path))
                except FileNotFoundError:
                    pass
                if img is not None:
                    break
            assert img is not None
        elif self.data_root is not None:
            img = self.loader(os.path.join(
                self.data_root, path))
        else:
            img = self.loader(os.path.join(
                os.path.dirname(self.data_file_p), path))

        # img = img.crop((
        #     data['FACE_X'],  # left
        #     data['FACE_Y'],  # top
        #     data['FACE_X'] + self.ds.loc[idx, 'FACE_WIDTH'],  # right
        #     data['FACE_Y'] + self.ds.loc[idx, 'FACE_HEIGHT'],  # bottom
        # ))
        if self.transform is not None:
            img = self.transform(img)
        return img[None, ...]

    def load_images(self, column_path, data):
        """
            data - pandas DataFrame
        """
        ret = []
        for _, row in data.iterrows():
            ret.append(self.load_image(column_path, row))
        ret = torch.cat(ret)
        return ret

    def __getitem__(self, idx):
        """ Returns probe image, set of mated references, and set of non-mated
            references.

            Probe image should have dimension of 1x3x224x224.
        """
        probe_data = self.probe_ds.iloc[idx]
        probe_im = self.load_image('OriginalFile', probe_data)

        ref_data = self.ref_ds.loc[probe_data['SUBJECT_ID'], probe_data['MASK_ID']]
        ref_mate_ims = self.load_images('OriginalFile', ref_data)
        ref_nonmate_ims = self.load_images('InpaintingFile', ref_data)
        if self.return_file_info:
            return probe_im, ref_mate_ims, ref_nonmate_ims, probe_data
        else:
            return probe_im, ref_mate_ims, ref_nonmate_ims

    def __len__(self):
        import pdb
        pdb.set_trace()
        return self.ds.shape[0]
