#### Libraries

from os import makedirs
from os.path import join, exists, dirname, isfile
from multiprocessing import Pool
from datetime import datetime
import random
import pickle

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import pyvips

from ..aux import vips2numpy, create_dir

#### Functions and Classes

class SVSPredictDataset(Dataset):
    
    def __init__(
        self,
        predict_data_dir,
        predict_metadata_path,
        prepare,
        metadata_write_dir,
        coords_write_dir,
        coords_read_dir,
        logging_name,
        patch_size,
        num_workers,
        transformations,
    ):
        """
        This dataset is used to predict the output of .svs files in the `predict_data_dir`.
            All .svs files should be in the root directory of `predict_data_dir`.
        """
        super(SVSPredictDataset, self).__init__()
        self.predict_data_dir = predict_data_dir
        self.predict_metadata_path = predict_metadata_path
        self.prepare = prepare
        self.metadata_write_dir = metadata_write_dir
        self.coords_write_dir = coords_write_dir
        self.coords_read_dir = coords_read_dir
        self.logging_name = logging_name

        self.patch_size = patch_size
        self.num_workers  = num_workers

        self.transformations = transformations


        #####################################
        # In case of DGC style datasets
        predict_meta = pd.read_csv(self.predict_metadata_path)
        predict_image_filenames = list(predict_meta.apply(lambda x: join(self.predict_data_dir, x.id, x.filename), axis=1))

        # predict_image_filenames = os.listdir(self.predict_data_dir)

        if self.prepare:
            if self.coords_read_dir:
                pass
            else:
                self._calculate_coords(predict_image_filenames)
        else:
            if self.coords_read_dir:
                self.patch_coords = self._read_coords(self.coords_read_dir, self.dataset_type)
            else:
                if self.coords_write_dir is None:
                    self.coords_write_dir = join("./coords/", self.logging_name)
                    
                self.patch_coords = self._read_coords(self.coords_write_dir, self.dataset_type)
            
            print(f"total number of patches loaded = {len(self.patch_coords)}")



    def _calculate_coords(self, filenames):
        pool = Pool(processes=self.num_workers)
        print("pool")
        pool_out = pool.map(self._fetch_coords, filenames)
        patch_coords = [elem for sublist in pool_out for elem in sublist]

        num_patches_created = len(patch_coords)
        print(f"number of pathces created from prediction set = {num_patches_created}")

        with open(join(self.coords_write_dir, "predict.data"),'wb') as filehandle:
            pickle.dump(patch_coords, filehandle)
            filehandle.close()
        
        return num_patches_created


    def _fetch_coords(self, fname):
        print(fname, flush=True)
        img = self._load_file(fname)
        patches = self._patching(img)
        dirs = [fname] * len(patches)
        return list(zip(dirs, patches))
    

    def _patching(self, img):       
        coords = []
        
        vertical_patches = np.floor(img.height / self.patch_size)
        horizontal_patches = np.floor(img.width / self.patch_size) * 128

        for i in range(vertical_patches):
            for j in range(horizontal_patches):
                coords.append((j * self.patch_size, i * self.patch_size))

        return coords


    def _load_file(self, file):
        image = pyvips.Image.new_from_file(str(file))
        return image


    def _img_to_tensor(self, img, x, y):
        t = img.crop(x, y, self.patch_size, self.patch_size)
        t_np = vips2numpy(t)
        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        out_t = trans(t_np)
        out_t = out_t[:3, :, :]
        return out_t


    # -> Dunder Methods


    def __getitem__(self, index):
        fname, coord_tuple = self.patch_coords[index]
        img = self._load_file(fname)
        out = self._img_to_tensor(img, coord_tuple[0], coord_tuple[1])
        if self.transformations is not None:
            out = self.transformations(out)
        return out, out.size()


    def __len__(self):
        return len(self.patch_coords)