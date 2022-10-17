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

from .aux import vips2numpy

#### Functions and Classes

class GDCSVSDataset(Dataset):
    
    def __init__(
        self,
        gdc_data_dir,
        gdc_metadata_path,
        cancer_type,
        ratio_per_type,
        dataset_type,
        split_ratio,
        test_random_seed,
        train_val_random_seed,
        prepare,
        metadata_write_dir,
        coords_write_dir,
        coords_read_dir,
        logging_name,
        patch_size,
        num_patches_per_image,
        pathcing_seed,
        whitespace_threshold,
        num_workers,
        transformations,
    ):
        """
        `ratio_per_type`: The ratio of images sampled from each cancer type when cancer_type='all'. Real number in [0,1].

        Sample:
        """
        super(GDCSVSDataset, self).__init__()
        self.gdc_data_dir = gdc_data_dir
        self.gdc_metadata_path = gdc_metadata_path
        self.cancer_type = cancer_type
        self.ratio_per_type = ratio_per_type
        self.dataset_type = dataset_type
        self.split_ratio = split_ratio
        self.test_random_seed = test_random_seed
        self.train_val_random_seed = train_val_random_seed
        self.prepare = prepare
        self.metadata_write_dir = metadata_write_dir
        self.coords_write_dir = coords_write_dir
        self.coords_read_dir = coords_read_dir
        self.logging_name = logging_name

        self.patch_size = patch_size
        self.num_patches_per_image = num_patches_per_image
        self.pathcing_seed = pathcing_seed
        self.whitespace_threshold = whitespace_threshold
        self.num_workers  = num_workers

        self.transformations = transformations

        if self.prepare:
            if self.coords_read_dir:
                pass
            else:
                self.Prepare()
        else:
            if self.coords_read_dir:
                self.patch_coords = self._read_coords(self.coords_read_dir, self.dataset_type)
            else:
                if self.coords_write_dir is None:
                    self.coords_write_dir = join("./coords/", self.logging_name)
                    
                self.patch_coords = self._read_coords(self.coords_write_dir, self.dataset_type)
            
            print(f"total number of patches loaded = {len(self.patch_coords)}")


    # -> Aux Methods


    def Prepare(self):
        """
        This method is only run on the first GPU in strategy=ddp. It is called 
            under prepare() method in pl DataModule.
        """

        # Prompting the start of preparation.
        print("Data Preparation Started ...")


        # Making sure the directories exist.
        if self.metadata_write_dir is None:
            self.metadata_write_dir = join("./metadata/", self.logging_name)

        self._create_dir(self.metadata_write_dir)

        if self.coords_write_dir is None:
            self.coords_write_dir = join("./coords/", self.logging_name)
        
        self._create_dir(self.coords_write_dir)


        # Creating proper metadata.
        #   -> Reading in the gdc metadata.
        gdc_meta = pd.read_csv(self.gdc_metadata_path)


        #   ->  Choosing which images to include in the dataset.
        if self.cancer_type == 'all':
            # Samples num_images_by_type_dict[type] from each cancer type.
            # Logic can be set by defining `num_images_by_type_dict`.
            num_images_by_type_dict = gdc_meta.groupby('primary_site').count()['id'].to_dict()
            for key, value in num_images_by_type_dict.items():
                num_images_by_type_dict[key] = int(np.ceil(self.ratio_per_type * value))
            metadata = gdc_meta.groupby('primary_site', group_keys=True).apply(
                lambda x: x.sample(n=num_images_by_type_dict[x.iloc[0].primary_site])
            ).reset_index(drop=True).copy(deep=True)
        else:
            gdc_meta = gdc_meta.sample(frac=1).reset_index(drop=True).copy(deep=True)
            gdc_meta = gdc_meta[gdc_meta['primary_site']==self.cancer_type].copy(deep=True)
            n = int(np.ceil(self.ratio_per_type * len(gdc_meta)))
            metadata = gdc_meta.iloc[0:n].reset_index(drop=True).copy(deep=True)

        #   -> Saving the dataset metadata to file for future reference and debugging.
        metadata.to_csv(join(self.metadata_write_dir, 'metadata.csv'), index=False)

        print(f"total number of images added to the dataset = {len(metadata)}")
        print(f"total number of patches that will be created by the dataset = {len(metadata) * self.num_patches_per_image}")


        #   -> Splitting images into train/test/val or predict
        if self.dataset_type == "predict":
            predict_meta = metadata.copy(deep=True)

            # Saving splitted metadata files for future reference and debugging.
            predict_meta.to_csv(join(self.metadata_write_dir, 'predict.csv'), index=False)

            # Reporting the count of images in each split
            print(f"total number of images added to the predict dataset = {len(predict_meta)}")
        else:
            train_meta, val_meta, test_meta = self._split_metadata(metadata, self.split_ratio)

            # Saving splitted metadata files for future reference and debugging.
            train_meta.to_csv(join(self.metadata_write_dir, 'train.csv'), index=False)
            val_meta.to_csv(join(self.metadata_write_dir, 'val.csv'), index=False)
            test_meta.to_csv(join(self.metadata_write_dir, 'test.csv'), index=False)

            # Reporting the count of images in each split
            print(f"total number of images added to the train dataset = {len(train_meta)}")
            print(f"total number of images added to the val dataset = {len(val_meta)}")
            print(f"total number of images added to the test dataset = {len(test_meta)}")
        
 
        #   -> Creating a list of filenames for each splitted dataset
        if self.dataset_type == "predict":
            predict_image_filenames = list(predict_meta.apply(lambda x: join(self.gdc_data_dir, x.id, x.filename), axis=1))
        else:
            train_image_filenames = list(train_meta.apply(lambda x: join(self.gdc_data_dir, x.id, x.filename), axis=1))
            val_image_filenames = list(val_meta.apply(lambda x: join(self.gdc_data_dir, x.id, x.filename), axis=1))
            test_image_filenames = list(test_meta.apply(lambda x: join(self.gdc_data_dir, x.id, x.filename), axis=1))
        

        #   -> calculating the coordinates and storing them for each split
        if self.dataset_type == "predict":
            #TODO->Paul
            # Some type of coords calculation for predict dataset.
            pass
        else:
            npc_train = self._calculate_coords(train_image_filenames, 'train')
            npc_val = self._calculate_coords(val_image_filenames, 'val')
            npc_test = self._calculate_coords(test_image_filenames, 'test')

            print(f"total number of patches created = {npc_train + npc_val + npc_test}")
        
        print("Data Preparation Done!")
        
    
    def _split_metadata(self, metadata, split_ratio):
        # Calculating the train/val/test set sizes
        n = len(metadata)
        train_size = int(split_ratio[0] * n)
        val_size = int(split_ratio[1] * n)
        test_size = n - train_size - val_size
        
        # Shuffling the metadata and creating train/val/test metadata dataframes
        test_rg = None
        train_val_rg = None

        if self.test_random_seed is not None:
            test_rg = np.random.default_rng(self.test_random_seed)

        if self.train_val_random_seed is not None:
            train_val_rg = np.random.default_rng(self.train_val_random_seed)

        #   -> First we split the metadata to test/not-test using `test_rg`.
        #       This has precedence over pl.seed_everything().
        meta = metadata.sample(frac=1, replace=False, random_state=test_rg).reset_index(drop=True).copy(deep=True)

        test_meta = meta.iloc[0 : test_size].reset_index(drop=True).copy(deep=True)

        #   -> Then we split the non-test portion of data to train/val using `train_val_rg`.`
        #       This has precedence over pl.seed_everything().
        meta = meta.iloc[test_size : ].sample(frac=1, replace=False, random_state=train_val_rg).reset_index(drop=True).copy(deep=True)

        train_meta = meta.iloc[0 : train_size].reset_index(drop=True).copy(deep=True)
        val_meta = meta.iloc[train_size : ].reset_index(drop=True).copy(deep=True)

        return train_meta, val_meta, test_meta 
    

    def _calculate_coords(self, filenames, split_type):
        """
        `split_type` can be 'train', 'val', 'test', or 'prediction'
        """
        pool = Pool(processes=self.num_workers)
        print("pool")
        pool_out = pool.map(self._fetch_coords, filenames)
        patch_coords = [elem for sublist in pool_out for elem in sublist]
        # Randomly shuffling the patches before feeding to dataloader
        random.shuffle(patch_coords)

        num_patches_created = len(patch_coords)
        print(f"number of pathces created from {split_type} = {num_patches_created}")

        with open(join(self.coords_write_dir, split_type + ".data"),'wb') as filehandle:
            pickle.dump(patch_coords, filehandle)
            filehandle.close()
        
        return num_patches_created

    
    def _read_coords(self, coords_read_dir, split_type):
        with open(join(coords_read_dir, split_type + ".data"), 'rb') as filehandle:
                patch_coords = pickle.load(filehandle)
                filehandle.close()
        return patch_coords


    def _fetch_coords(self, fname):
        print(fname, flush=True)
        img = self._load_file(fname)
        patches = self._patching(img, seed=self.pathcing_seed)
        dirs = [fname] * len(patches)
        return list(zip(dirs, patches))


    def _load_file(self, file):
        image = pyvips.Image.new_from_file(str(file))
        return image
    

    def _patching(self, img, seed):
        if seed is not None:
            random.seed(seed)
        
        # Making sure not to spend more than 5 mins per image to find the required number of patches
        start_time = datetime.now()
        spent_time = datetime.now() - start_time

        count = 0
        coords = []
        while count < self.num_patches_per_image and spent_time.total_seconds() < 300:
            # [4, x, y] -> many [4, 512, 512]
            rand_i = random.randint(0, img.width - self.patch_size)
            rand_j = random.randint(0, img.height - self.patch_size)
            temp = self._img_to_tensor(img, rand_i, rand_j)
            if self._filter_whitespace(temp, threshold=self.whitespace_threshold):
                if self._get_intersections(rand_j, rand_i, coords):
                    coords.append((rand_i, rand_j))
                    count += 1
            spent_time = datetime.now() - start_time
        return coords


    def _get_intersection(self, a_x, a_y, b_x, b_y):
        # Tensors are row major
        if abs(a_x - b_x) < self.patch_size and abs(a_y - b_y) < self.patch_size:
            return True
        else:
            return False


    def _get_intersections(self, x, y, coords):
        if len(coords) == 0:
            return True
        else:
            ml = set(map(lambda b: self._get_intersection(b[0], b[1], x, y), coords))
            if True in ml:
                return False
            else: 
                return True


    def _filter_whitespace(self, tensor_3d, threshold):
        r = np.mean(np.array(tensor_3d[0]))
        g = np.mean(np.array(tensor_3d[1]))
        b = np.mean(np.array(tensor_3d[2]))
        channel_avg = np.mean(np.array([r, g, b]))
        if channel_avg < threshold:
            return True
        else:
            return False


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

    
    def _create_dir(self, directory):
        if not exists(directory):
            makedirs(directory)
            print(f"directory created: {directory}")


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