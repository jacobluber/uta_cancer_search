from __future__ import print_function
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


parser = argparse.ArgumentParser(description='H&E Autoencoder')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--nodes',type=int,default=1,metavar='N',help='number of nodes to utilize for training')
parser.add_argument('--gpus',type=int,default=4,metavar='N',help='number of GPUs to utilize per node (default: 4)')
parser.add_argument('--workers',type=int,default=16,metavar='N',help='number of CPUs to use in the pytorch dataloader (default: 16)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--patches',type=int,default=200,metavar='N',help='number of patches to sample per H&E image (default: 200)')
parser.add_argument('--patch-size',type=int,default=512,metavar='N',help='size of the patch X*Y where x=patch_size and y=patch_size (default: 512)')
parser.add_argument('--svs-dir',default='/home/data/gdc',metavar='S',help='SVS file to sample from if not using pre-saved coords (default: /data/luberjm/data/small/svs)')
parser.add_argument('--cancer-type',default='Bronchus and lung',metavar='S',help='Used to load the data corresponding to a specific type of cancer.')
parser.add_argument('--custom-coords-file',default='/home/mxn2498/projects/uta_cancer_search/custom_coords/patch_coords.data',metavar='S',help='add this flag to use a non-default coords file (default: patch_coords.data)')
parser.add_argument('--train-size',default='100',metavar='N',help='size of the training set (default: 100)')
parser.add_argument('--test-size',default='10',metavar='N',help='size of the training set, must be an even number (default: 10)')
parser.add_argument('--accelerator',default='gpu', metavar='S',help='gpu accelerator to use, use ddp for running in parallel (default: gpu)')
parser.add_argument('--logging-name',default='autoencoder', metavar='S',help='name to log this run under in tensorboard (default: autoencoder)')
parser.add_argument('--resnet',default='resnet18',metavar='S')
parser.add_argument('--enc-dim',default='512',metavar='N')
parser.add_argument('--latent-dim',default='256',metavar='N')
parser.add_argument('--first-conv',dest='first_conv',action='store_true')
parser.add_argument('--maxpool1',dest='maxpool1',action='store_true')
parser.add_argument('--read-coords',dest='read_coords',action='store_true',help='add this flag to read in previously sampled patch coordinates that pass QC from the default file \'patch_coords.data\'')
parser.add_argument('--write-coords', dest='write_coords', action='store_true',help='add this flag to write out sampled coordinates that pass QC to the default directory \'custom_coords/\', which can be preloaded to speed up training')
parser.add_argument('--coords-file-name',default='patch_coords.data',metavar='S',help='add this flag to determine the name of the custom_coords file.')

#these are not implemented yet but need to be in the future 
#parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                    help='how many batches to wait before logging training status')
#parser.add_argument('--seed', type=int, default=42, metavar='S',
#                    help='random seed (default: 42)')
args = parser.parse_args()


kwargs = {'batch_size':args.batch_size,'pin_memory':True,'num_workers':args.workers}


if __name__ == '__main__':
    transformations = transforms.Compose([transforms.ToPILImage(),transforms.CenterCrop(64),transforms.ToTensor(),transforms.Normalize(mean=[0.5937,0.5937,0.5937,0.5937], std=[0.0810,0.0810,0.0810,0.0810])])
    input_data = SvsDatasetFromFolder(args.svs_dir,args.cancer_type,args.patch_size,args.patches,args.workers,args.write_coords,args.coords_file_name,args.read_coords,args.custom_coords_file,transforms=transformations)
    data_train, data_other = random_split(input_data, [int(args.train_size), int(args.test_size)])
    data_test,data_val = random_split(data_other, [int(int(args.test_size)/2),int(int(args.test_size)/2)])
    train_loader = torch.utils.data.DataLoader(data_train,  **kwargs)
    test_loader = torch.utils.data.DataLoader(data_test,  **kwargs)
    val_loader = torch.utils.data.DataLoader(data_val, **kwargs)
    tb_logger = TensorBoardLogger('/home/mxn2498/projects/uta_cancer_search/logs', name=args.logging_name, log_graph=False)
    rddp = False
    if args.accelerator == "ddp":
        rddp = True
    trainer = pl.Trainer(max_epochs=args.epochs, replace_sampler_ddp=rddp, devices=args.gpus,logger=tb_logger,num_nodes=args.nodes,accelerator=args.accelerator,auto_lr_find=False,benchmark=True,fast_dev_run=False,gradient_clip_val=0.5,sync_batchnorm=True) #flush_logs_every_n_steps=1
    #autoencoder = AutoEncoder()
    #autoencoder = VanillaVAE()
    autoencoder = customVAE(enc_type=args.resnet,first_conv=args.first_conv,maxpool1=args.maxpool1,enc_out_dim=args.enc_dim,latent_dim=args.latent_dim)
    trainer.tune(autoencoder,train_loader,val_loader)
    trainer.fit(autoencoder,train_loader,val_loader)
    #trainer.save_checkpoint("/data/luberjm/models/50.ckpt")
    #trainer.fit(autoencoder,train_loader)
    #save model here
    fun = trainer.test(autoencoder,test_loader)
