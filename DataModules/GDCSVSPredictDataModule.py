#### Libraries

from os.path import exists, join
import argparse
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn
import numpy as np

from Datasets.GDCSVSDataset import GDCSVSDataset
from Utils.Stats import DataLoaderStats
from Utils.aux import create_dir, load_transformation

#### Functions and Classes

class GDCSVSPredictDataModule(pl.LightningDataModule):

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
            "--coords_read_dir",
            type = str,
            default = None,
            help = "Directory defining where to read previously generated coordinates .data files. This directory should include train.data, val.data, test.data, or predict.data depending on dataset_type. If not provided, the coordinates would be generated during in the prepare() method of pl DataModule and will be stored in './logs/coords/logging_name/'. [default: None]"
        )

        parser.add_argument(
            "--transformations_read_dir",
            type = str,
            default = None,
            help = "Directory defining where to read previously generated transformations and inverse transformations .obj files. This directory should include trans.obj and inv_trans.obj. If not provided, no transformations is applied. [default: None]"
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
            "--per_image_normalize",
            action = argparse.BooleanOptionalAction,
            help = "Whether to normalize each patch with respect to itself."
        )

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

        return parser

    def __init__(
        self,
        gdc_data_dir,
        gdc_metadata_path,
        cancer_type,
        ratio_per_type,
        metadata_write_dir,
        coords_write_dir,
        coords_read_dir,
        transformations_read_dir,
        logging_name,
        patch_size,
        num_patches_per_image,
        pathcing_seed,
        whitespace_threshold,
        per_image_normalize,
        num_dataset_workers,
        batch_size,
        num_dataloader_workers,
        *args,
        **kwargs,
    ):
        """
        docstring goes here.
        """

        super().__init__()
        
        self.gdc_data_dir = gdc_data_dir
        self.gdc_metadata_path = gdc_metadata_path
        self.cancer_type = cancer_type
        self.ratio_per_type = ratio_per_type
        self.metadata_write_dir = metadata_write_dir
        self.coords_write_dir = coords_write_dir
        self.coords_read_dir = coords_read_dir
        self.transformations_read_dir = transformations_read_dir
        self.logging_name = logging_name
        self.patch_size = patch_size
        self.num_dataset_workers = num_dataset_workers
        self.num_patches_per_image = num_patches_per_image
        self.pathcing_seed = pathcing_seed
        self.whitespace_threshold = whitespace_threshold
        self.per_image_normalize = per_image_normalize

        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers

        # Because of conflicts on,oading checkpoints, we will not save hparams using self.save_hyperparameters()
        #   If needed, should be manually added with a different naming convention than the one used for hparams
        #   during train/val/test.

        # The last 6 variables are not important at all for prediction dataset.
        self.dataset_kwargs = {
            "gdc_data_dir": self.gdc_data_dir,
            "gdc_metadata_path": self.gdc_metadata_path,
            "cancer_type": self.cancer_type,
            "ratio_per_type": self.ratio_per_type,
            "metadata_write_dir": self.metadata_write_dir,
            "coords_write_dir": self.coords_write_dir,
            "coords_read_dir": self.coords_read_dir,
            "logging_name": self.logging_name,
            "patch_size": self.patch_size,
            "num_workers": self.num_dataset_workers,
            "whitespace_threshold": self.whitespace_threshold,
            "per_image_normalize": self.per_image_normalize,
            "pathcing_seed": self.pathcing_seed,
            "num_patches_per_image": self.num_patches_per_image,

            "split_ratio": [1, 0, 0],
            "test_random_seed": None,
            "train_val_random_seed": None,
        }


    def prepare_data(self):
        if self.trainer.state.fn == TrainerFn.PREDICTING:
            # Calculating the coordinats
            dataset = GDCSVSDataset(dataset_type='predict', prepare=True, transformations=None, **self.dataset_kwargs)

        
    def setup(self, stage=None):
        # Loading transformations from file
        transformations = None
        
        if self.transformations_read_dir is not None:
            transformations = load_transformation(join(self.transformations_read_dir, "trans.obj"))

        # Creating corresponding datasets
        if stage in (None, "predict"):
            self.predict_dataset = GDCSVSDataset(dataset_type="predict", prepare=False, transformations=transformations, **self.dataset_kwargs)

    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_dataloader_workers)