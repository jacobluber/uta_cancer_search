from __future__ import print_function
from fcntl import F_SEAL_SEAL
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split, distributed
from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger
from utils import *
from vae5 import *
from dataloader import *
import argparse
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

if __name__ == '__main__':
    svs_dir =
    cancer_type =
    patch_size = 
    patches =
    workers =
    write_coords = False
    coords_file_name = 
    read_coords = 
    custom_coords_file = 
    train_size =
    test_size =
    batch_size = 

    kwargs = {'batch_size':batch_size,'pin_memory':True,'num_workers':workers}

    transformations = transforms.Compose([transforms.ToPILImage(),transforms.CenterCrop(64),transforms.ToTensor(),transforms.Normalize(mean=[0.5937,0.5937,0.5937,0.5937], std=[0.0810,0.0810,0.0810,0.0810])])
    input_data = SvsDatasetFromFolder(svs_dir,cancer_type,patch_size,patches,workers,write_coords,coords_file_name,read_coords,custom_coords_file,transforms=transformations)
    data_train, data_other = random_split(input_data, [int(train_size), int(test_size)])
    data_test,data_val = random_split(data_other, [int(int(test_size)/2),int(int(test_size)/2)])
    train_loader = torch.utils.data.DataLoader(data_train,  **kwargs)
    test_loader = torch.utils.data.DataLoader(data_test,  **kwargs)
    val_loader = torch.utils.data.DataLoader(data_val, **kwargs)