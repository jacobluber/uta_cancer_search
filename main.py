#### Libraries

from os import makedirs
from os.path import exists
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from utils.Modules import CustomVAE
from utils.DataModules import GDCSVSDataModule

#### Functions and Classes

def main_func(args):
    """
    docstring goes here.
    """
    dict_args = vars(args)

    if args.everything_seed is not None:
        pl.seed_everything(args.everything_seed)
    
    # Iniatiating the DataModule
    data_module = GDCSVSDataModule(**dict_args)

    # Iniatiating the model
    model = CustomVAE(**dict_args)

    if not exists(args.logging_dir):
        makedirs(args.logging_dir)

    tb_logger = TensorBoardLogger(args.logging_dir, name=args.logging_name, log_graph=False)
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger)
    
    trainer.tune(model, data_module)
    trainer.fit(model, data_module)
    trainer.validate(model, data_module)
    trainer.test(model, data_module)
    
if __name__ == '__main__':
    parser = ArgumentParser()

    # Program Level Args

    # -> Main Function Args

    parser.add_argument(
        "--everything_seed",
        type = int,
        default = None,
        help = "Seed used with pl.seed_everything(). If provided, everything would be reproducible except the patching coordinates. [default: None]"
    )

    parser.add_argument(
        "--logging_dir",
        type = str,
        default = "./logs",
        help = "Address of the logs directory. [default: ./logs]"
    )

    parser.add_argument(
        "--logging_name",
        type = str,
        default = "experiment",
        help = "name of the current experiment. [default: experiment]"
    )
    
    # dataset specific args
    parser = GDCSVSDataModule.add_dataset_specific_args(parser)

    # model specific args
    parser = CustomVAE.add_model_specific_args(parser)

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    
    # parsing args
    args = parser.parse_args()
    
    # Calling the main function
    main_func(args)
    
    
    
    
    