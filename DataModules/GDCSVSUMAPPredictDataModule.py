#### Libraries

import argparse
from os import makedirs
from os.path import exists, join
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn
import numpy as np

from Datasets.GDCSVSUMAPDataset import GDCSVSDataset
from Utils.Stats import DataLoaderStats
from Utils.aux import create_dir, save_transformation, load_transformation

#### Functions and Classes

class GDCSVSDataModule(pl.LightningDataModule):

    @staticmethod
    def add_dataset_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # -> Datasset Args

        parser.add_argument(
            "--gdc_data_dir",
            type = str,
            default = "/home/data/gdc/",
            help = "Address of the dataset directory. [default: /home/data/gdc/]"
        )

        parser.add_argument(
            "--gdc_metadata_path",
            type = str,
            default = "/home/data/gdc/metadata.csv",
            help = "Path pointing to the main GDC dataset metadata .csv file. [default: /home/data/gdc/metadata.csv]"
        )

        parser.add_argument(
            "--cancer_type",
            type = str,
            default = "Bronchus and lung",
            help = "Type of cancer to train on. If chosen to be `all` the dataset would choose ceil `ratio_per_type` of each cancertype. [default: Bronchus and lung]"
        )

        parser.add_argument(
            "--ratio_per_type",
            type = float,
            default = 0.1,
            help = "The ratio of images of a each cancer type we want to add to dataset per total number of images available in the dataset for that cancer type. Can be used with `cancer_type`='all'. [default: 0.1]"
        )

        parser.add_argument(
            "--dataset_type",
            type = str,
            default = 'train',
            help = "The type of dataset we are trying to create. Can be 'train', 'val', 'test', or 'predict'. If `dataset_type`='predict' and `prepare`=True, only predict.data would be generated. [default: 'train']"
        )

        parser.add_argument(
            "--split_ratio",
            nargs = 3,
            type = float,
            default = [0.9, 0.05, 0.05],
            help = "The ratios we want to split the data into train/val/test. Sum of the ratios should be equal tp 1. The ratios should be seperated only with white space. [default: 0.9 0.05 0.05]"
        )

        parser.add_argument(
            "--test_random_seed",
            type = int,
            default = None,
            help = "Seed used to split dataset into test and not test. If None, pl.seed_everything() will seed this. But if provided, this has predence. [default: None]"
        )

        parser.add_argument(
            "--train_val_random_seed",
            type = int,
            default = None,
            help = "Seed used to split train_val dataset into train and val. If None, pl.seed_everything() will seed this. But if provided, this has predence. [default: None]"
        )

        parser.add_argument(
            "--metadata_write_dir",
            type = str,
            default = None,
            help = "Directory defining where to save the generated dataset metadata .csv files. If not provided, all generated metadata .csv files will be stored in './logs/metadata/logging_name/'. [default: None]"
        )

        parser.add_argument(
            "--coords_write_dir",
            type = str,
            default = None,
            help = "Directory defining where to save the generated coordinates .data files. If not provided, all generated coordinate files will be stored in './logs/coords/logging_name/'. [default: None]"
        )

        parser.add_argument(
            "--transformations_write_dir",
            type = str,
            default = None,
            help = "Directory defining where to save the generated transformations and inverse transformations .obj files. If not provided, all generated coordinate files will be stored in './logs/tb_logs/logging_name/'. [default: None]"
        )

        parser.add_argument(
            "--transformations_read_dir",
            type = str,
            default = None,
            help = "Directory defining where to read previously generated transformations and inverse transformations .obj files. This directory should include trans.obj and inv_trans.obj. If not provided, no transformations is applied. [default: None]"
        )

        parser.add_argument(
            "--coords_read_dir",
            type = str,
            default = None,
            help = "Directory defining where to read previously generated coordinates .data files. This directory should include train.data, val.data, test.data, or predict.data depending on dataset_type. If not provided, the coordinates would be generated during in the prepare() method of pl DataModule and will be stored in './logs/coords/logging_name/'. [default: None]"
        )

        parser.add_argument(
            "--patch_size",
            type = int,
            default = 64,
            help = "Size of the square patches sampled from each image. [default: 64]"
        )

        parser.add_argument(
            "--num_patches_per_image",
            type = int,
            default = 10,
            help = "Number of patches that will be sampled from each image. [default: 10]"
        )

        parser.add_argument(
            "--pathcing_seed",
            type = int,
            default = None,
            help = "Seed used to generate random patches. pl.seed_everything() will not set the seed for pathcing. It should be passed manually. [default: None]"
        )

        parser.add_argument(
            "--whitespace_threshold",
            type = float,
            default = 0.82,
            help = "The threshold used for classifying a patch as mostly white space. The mean of pixel values over all channels of a patch after applying transformations is compared to this threshold. [default: 0.82]"
        )

        parser.add_argument(
            "--num_dataset_workers",
            type = int,
            default = 8,
            help = "Number of processor workers used for patching images. [default: 8]"
        )

        # -> Data Module Args

        parser.add_argument(
            "--batch_size",
            type = int,
            default = 128,
            help = "The batch size used with all dataloaders. [default: 128]"
        )

        parser.add_argument(
            "--num_dataloader_workers",
            type = int,
            default = 8,
            help = "Number of processor workers used for dataloaders. [default: 8]"
        )

        parser.add_argument(
            "--per_image_normalize",
            action = argparse.BooleanOptionalAction,
            help = "Whether to normalize each patch with respect to itself."
        )

        parser.add_argument(
            "--normalize_transform",
            action = argparse.BooleanOptionalAction,
            help = "If passed, DataModule will calculate or load the whole training dataset mean and std per channel and passes it to transforms."
        )

        parser.add_argument(
            "--resize_transform_size",
            type = int,
            default = None,
            help = "If provided, the every patch would be resized from patch_size to resize_transform_size. [default: None]"
        )

        return parser

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
        metadata_write_dir,
        coords_write_dir,
        transformations_write_dir,
        transformations_read_dir,
        coords_read_dir,
        logging_name,
        patch_size,
        num_patches_per_image,
        pathcing_seed,
        whitespace_threshold,
        num_dataset_workers,
        batch_size,
        num_dataloader_workers,
        per_image_normalize,
        normalize_transform,
        resize_transform_size,
        *args,
        **kwargs,
    ):
        """split_ration is a list of three numbers summing up to 1, e.g. [0.7, 0.2, 0.1].
                The first number is the ratio of training set.
                The second number is the ratio of validation set.
                The third number is the ratio of test set.
        """

        super().__init__()
        
        self.gdc_data_dir = gdc_data_dir
        self.gdc_metadata_path = gdc_metadata_path
        self.cancer_type = cancer_type
        self.ratio_per_type = ratio_per_type
        self.dataset_type = dataset_type
        self.split_ratio = split_ratio
        self.test_random_seed = test_random_seed
        self.train_val_random_seed = train_val_random_seed
        self.metadata_write_dir = metadata_write_dir
        self.coords_write_dir = coords_write_dir
        self.transformations_write_dir = transformations_write_dir
        self.transformations_read_dir = transformations_read_dir
        self.coords_read_dir = coords_read_dir
        self.logging_name = logging_name
        self.patch_size = patch_size
        self.num_patches_per_image = num_patches_per_image
        self.pathcing_seed = pathcing_seed
        self.whitespace_threshold = whitespace_threshold
        self.num_dataset_workers = num_dataset_workers

        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.per_image_normalize = per_image_normalize
        self.normalize_transform = normalize_transform
        self.resize_transform_size = resize_transform_size

        # saving hyperparameters to checkpoint
        self.save_hyperparameters()

        self.dataset_kwargs = {
            "gdc_data_dir": self.gdc_data_dir,
            "gdc_metadata_path": self.gdc_metadata_path,
            "cancer_type": self.cancer_type,
            "ratio_per_type": self.ratio_per_type,
            "split_ratio": self.split_ratio,
            "test_random_seed": self.test_random_seed,
            "train_val_random_seed": self.train_val_random_seed,
            "metadata_write_dir": self.metadata_write_dir,
            "coords_write_dir": self.coords_write_dir,
            "coords_read_dir": self.coords_read_dir,
            "logging_name": self.logging_name,
            "patch_size": self.patch_size,
            "num_patches_per_image": self.num_patches_per_image,
            "pathcing_seed": self.pathcing_seed,
            "whitespace_threshold": self.whitespace_threshold,
            "per_image_normalize": per_image_normalize,
            "num_workers": self.num_dataset_workers
        }

        #   Making sure the directories exist.
        if self.transformations_write_dir is None:
            self.transformations_write_dir = join("./logs/tb_logs/", self.logging_name)

        create_dir(self.transformations_write_dir)


    def prepare_data(self):
        if self.trainer.state.fn in (TrainerFn.FITTING, TrainerFn.TUNING):
            # Calculating the coordinats
            dataset = GDCSVSDataset(dataset_type=self.dataset_type, prepare=True, transformations=None, **self.dataset_kwargs)
            
            # We can use the condition if self.normalize_transform here, but I decided to calculate 
            #   train dataset stats the first time a new dataset is created.

            # Finding normalization parameters
            if dataset.coords_read_dir is not None:
                stats_dir = join(dataset.coords_read_dir, "stats")
            elif dataset.coords_write_dir is not None:
                stats_dir = join(dataset.coords_write_dir, "stats")
            else:
                create_dir(join("./logs/coords/", self.logging_name))
                stats_dir = join("./logs/coords/", self.logging_name, "stats")

            if not exists(stats_dir):
                train_dataset = GDCSVSDataset(dataset_type="train", prepare=False, transformations=None, **self.dataset_kwargs)
                
                # All stats should be calculated at highest stable batch_size to reduce approximation errors for mean and std
                loader = DataLoader(train_dataset, batch_size=256, num_workers=self.num_dataloader_workers)
                loader_stats = DataLoaderStats(loader, stats_dir)

        if self.trainer.state.fn == TrainerFn.PREDICTING:
            # Calculating the coordinats
            dataset = GDCSVSDataset(dataset_type='predict', prepare=True, transformations=None, **self.dataset_kwargs)

        
    def setup(self, stage=None):

        # Determining transformations to apply.

        transforms_list = []
        inverse_transforms_list = []
        final_size = self.patch_size

        if self.normalize_transform:
            if self.coords_read_dir is not None:
                stats_dir = join(self.coords_read_dir, "stats")
            elif self.coords_write_dir is not None:
                stats_dir = join(self.coords_write_dir, "stats")
            else:
                create_dir(join("./logs/coords/", self.logging_name))
                stats_dir = join("./logs/coords/", self.logging_name, "stats")
                
            std = np.loadtxt(join(stats_dir, "std.gz"))
            mean = np.loadtxt(join(stats_dir, "mean.gz"))

            print("**************************************")
            print(f"mean of training set used for normalization: {mean}")
            print(f"std of training set used for normalization: {std}")

            transforms_list.append(
                transforms.Normalize(mean=mean, std=std)
            )

            inverse_transforms_list.insert(0, transforms.Normalize(mean=-mean, std=np.array([1, 1, 1])))
            inverse_transforms_list.insert(0, transforms.Normalize(mean=np.array([0, 0, 0]), std=1/std))

        if self.resize_transform_size is not None:
            transforms_list.append(
                transforms.Resize(size=self.resize_transform_size, interpolation=InterpolationMode.BILINEAR)
            )

            inverse_transforms_list.insert(0, transforms.Resize(size=self.patch_size, interpolation=InterpolationMode.BILINEAR))

            final_size = self.resize_transform_size

        transforms_list.append(
            transforms.CenterCrop(final_size)
        )

        transformations = transforms.Compose(transforms_list)
        inverse_transformations = transforms.Compose(inverse_transforms_list)

        # Saving transformations to file
        save_transformation(transformations, join(self.transformations_write_dir, "trans.obj"))
        save_transformation(inverse_transformations, join(self.transformations_write_dir, "inv_trans.obj"))


        # Creating corresponding datasets

        if stage in (None, "fit"):
            self.train_dataset = GDCSVSDataset(dataset_type="train", prepare=False, transformations=transformations, **self.dataset_kwargs)
            self.val_dataset = GDCSVSDataset(dataset_type="val", prepare=False, transformations=transformations, **self.dataset_kwargs)
        elif stage in (None, "validate"):
            self.val_dataset = GDCSVSDataset(dataset_type="val", prepare=False, transformations=transformations, **self.dataset_kwargs)
        elif stage in (None, "test"):
            self.test_dataset = GDCSVSDataset(dataset_type="test", prepare=False, transformations=transformations, **self.dataset_kwargs)
        elif stage in (None, "predict"):
            transformations = load_transformation(join(self.transformations_read_dir, "trans.obj"))
            self.predict_dataset = GDCSVSDataset(dataset_type="predict", prepare=False, transformations=transformations, **self.dataset_kwargs)

        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_dataloader_workers)

    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_dataloader_workers)

    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_dataloader_workers)


    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_dataloader_workers)