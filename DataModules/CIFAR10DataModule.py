#### Libraries

from os import makedirs
from os.path import exists, join
import argparse
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn
import numpy as np
from sklearn.model_selection import train_test_split

from Datasets.CIFAR10Dataset import CIFAR10Dataset
from Utils.Stats import DataLoaderStats
from Utils.aux import create_dir, save_transformation

#### Functions and Classes

class CIFAR10DataModule(pl.LightningDataModule):

    @staticmethod
    def add_dataset_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # -> Datasset Args

        parser.add_argument(
            "--data_dir",
            type = str,
            default = "/home/axh5735/cifar10_entropy/7_7.5",
            help = "Address of dataset directory"
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
            "--per_image_normalize",
            action = argparse.BooleanOptionalAction,
            help = "Whether to normalize each patch with respect to itself."
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

        parser.add_argument(
            "--stats_write_dir",
            type = str,
            default = None,
            help = "Directory defining where to save the calculated mean and std of training set as .gz files. If not provided, all generated coordinate files will be stored in './logs/tb_logs/logging_name/stats'. [default: None]"
        )

        parser.add_argument(
            "--stats_read_dir",
            type = str,
            default = None,
            help = "Directory defining where to load the calculated mean and std of training set as .gz files. [default: None]"
        )

        parser.add_argument(
            "--transformations_write_dir",
            type = str,
            default = None,
            help = "Directory defining where to save the generated transformations and inverse transformations .obj files. If not provided, all generated coordinate files will be stored in './logs/tb_logs/logging_name/'. [default: None]"
        )

        return parser

    def __init__(
        self,
        data_dir,
        split_ratio,
        test_random_seed,
        train_val_random_seed,
        batch_size,
        per_image_normalize,
        num_dataloader_workers,
        normalize_transform,
        resize_transform_size,
        stats_write_dir,
        stats_read_dir,
        transformations_write_dir,
        logging_name,
        *args,
        **kwargs,
    ):
        """split_ration is a list of three numbers summing up to 1, e.g. [0.7, 0.2, 0.1].
                The first number is the ratio of training set.
                The second number is the ratio of validation set.
                The third number is the ratio of test set.
        """

        super().__init__()
        
        self.data_dir = data_dir
        self.split_ratio = split_ratio
        self.test_random_seed = test_random_seed
        self.per_image_normalize = per_image_normalize
        self.train_val_random_seed = train_val_random_seed

        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.normalize_transform = normalize_transform
        self.resize_transform_size = resize_transform_size
        self.stats_write_dir = stats_write_dir
        self.stats_read_dir = stats_read_dir
        self.transformations_write_dir = transformations_write_dir
        self.logging_name = logging_name

        # All cifar images are 32 x 32
        self.patch_size = 32

        self.train_ratio = self.split_ratio[0]
        self.test_ratio = self.split_ratio[2]
        self.val_ratio = (1 - self.train_ratio - self.test_ratio) / (1 - self.test_ratio)

        # saving hyperparameters to checkpoint
        self.save_hyperparameters()

        self.dataset_kwargs = {
            "data_dir": self.data_dir,
            "test_random_seed": self.test_random_seed,
            "per_image_normalize": self.per_image_normalize,
        }

        #   Making sure the directories exist.
        if self.stats_write_dir is None:
            self.stats_write_dir = join("./logs/tb_logs/", self.logging_name, "stats")
        
        create_dir(self.stats_write_dir)

        if self.transformations_write_dir is None:
            self.transformations_write_dir = join("./logs/tb_logs/", self.logging_name)

        create_dir(self.transformations_write_dir)


    def prepare_data(self):
        if self.trainer.state.fn in (TrainerFn.FITTING, TrainerFn.TUNING):
            if self.stats_read_dir is None:
                train_dataset = CIFAR10Dataset(train=True, test_ratio=self.test_ratio, transform=None, **self.dataset_kwargs)
                
                # All stats should be calculated at highest stable batch_size to reduce approximation errors for mean and std
                loader = DataLoader(train_dataset, batch_size=256, num_workers=self.num_dataloader_workers)
                loader_stats = DataLoaderStats(loader, self.stats_write_dir)

        
    def setup(self, stage=None):
        # Determining transformations to apply.

        transforms_list = []
        inverse_transforms_list = []
        final_size = self.patch_size

        if self.normalize_transform:
            if self.stats_read_dir is not None:
                stats_dir = self.stats_read_dir
            else:
                stats_dir = self.stats_write_dir
                
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
        if stage in (None, "fit", "validate"):
            train_val_dataset = CIFAR10Dataset(train=True, test_ratio=self.test_ratio, transform=transformations, **self.dataset_kwargs)

            if self.train_val_random_seed is None:
                self.train_dataset, self.val_dataset = train_test_split(train_val_dataset, test_size=self.val_ratio)
            else:
                self.train_dataset, self.val_dataset = train_test_split(train_val_dataset, test_size=self.val_ratio, random_state=self.train_val_random_seed)
            
            print(f"Number of images in train dataset: {len(self.train_dataset)}")
            print(f"Number of images in validation dataset: {len(self.val_dataset)}")

        elif stage in (None, "test"):
            self.test_dataset = CIFAR10Dataset(train=False, test_ratio=self.test_ratio, transform=transformations, **self.dataset_kwargs)
            print(f"Number of images in test dataset: {len(self.test_dataset)}")

        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_dataloader_workers)

    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_dataloader_workers)

    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_dataloader_workers)