#### Libraries

import torch
from datetime import datetime
from argparse import ArgumentParser

from pl_bolts.models.autoencoders.basic_vae.basic_vae_module import VAE
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.utils as vutils

#### Functions and Classes

class CustomVAE(VAE):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--input_height",
            type = int,
            default = 256,
            help = "Height of the images. [default: 64]"
        )

        parser.add_argument(
            "--enc_type",
            type = str,
            default = "resnet18",
            help = "Either resnet18 or resnet50. [default: resnet18]"
        )

        parser.add_argument(
            "--first_conv",
            action = "store_true",
            help = "Use standard kernel_size 7, stride 2 at start or replace it with kernel_size 3, stride 1 conv. [default: If the flag is not passed --> False]"
        )

        parser.add_argument(
            "--maxpool1",
            action = "store_true",
            help = "Use standard maxpool to reduce spatial dim of feat by a factor of 2. [default: If the flag is not passed --> False]"
        )

        parser.add_argument(
            "--enc_out_dim",
            type = int,
            default = 512,
            help = "Set according to the out_channel count of encoder used (512 for resnet18, 2048 for resnet50, adjust for wider resnets). [default: 512]",
        )

        parser.add_argument(
            "--kl_coeff",
            type = float,
            default = 0.1,
            help = "Coefficient for kl term of the loss. [default: 0.1]"
        )

        parser.add_argument(
            "--latent_dim",
            type = int,
            default = 256,
            help = "Dim of latent space. [default: 256]"
        )

        parser.add_argument(
            "--lr",
            type = float,
            default = 1e-4,
            help = "Learning rate for Adam. [default: 1e-4]"
        )

        return parser


    def __init__(
        self,
        input_height,
        enc_type,
        first_conv,
        maxpool1,
        enc_out_dim,
        kl_coeff,
        latent_dim,
        lr,
        *args,
        **kwargs,
    ):
        """
        docstring goes here
        """
        super(CustomVAE, self).__init__(
            input_height = int(input_height),
            enc_type = enc_type,
            first_conv = first_conv,
            maxpool1 = maxpool1,
            enc_out_dim = int(enc_out_dim),
            kl_coeff = kl_coeff,
            latent_dim = int(latent_dim),
            lr = lr,
            **kwargs,
        )

        # debugging
        self.example_input_array = torch.Tensor(1, 3, 64, 64)

        # saving hyperparameters to checkpoint
        self.save_hyperparameters()

        self.val_outs = []
        self.test_outs = []
        self.time = datetime.now() 
    

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, sync_dist=True)
        if self.global_rank == 0:
            if batch_idx == 0:
                self.val_outs = batch
        return loss    


    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, sync_dist=True)
        if self.global_rank == 0:
            if batch_idx == 0:
                self.test_outs = batch
        return loss 

    
    def predict_step(self, batch, batch_idx):
        pass

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


    def training_epoch_end(self, output):
        now = datetime.now()
        delta = now - self.time
        self.time = now
        tensorboard_logs = {'time_secs_epoch': delta.seconds}
        self.log_dict(tensorboard_logs, sync_dist=True)
    

    def validation_epoch_end(self, output):
        if self.global_rank == 0:
            x, y = self.val_outs
            z, x_hat, p, q = self._run_step(x) 

            if self.current_epoch == 0:
                vutils.save_image(
                    x,
                    f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/orig_{self.logger.name}_{self.current_epoch}.png",
                    normalize=True,
                    nrow=8
                )

            vutils.save_image(
                x_hat,
                f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/recons_{self.logger.name}_{self.current_epoch}.png",
                normalize=True,
                nrow=8
            )


    def test_epoch_end(self, output):
        if self.global_rank == 0:
            x, y = self.test_outs
            z, x_hat, p, q = self._run_step(x)

            vutils.save_image(
                x,
                f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/test_orig_{self.logger.name}_{self.current_epoch}.png",
                normalize=True,
                nrow=8
            )

            vutils.save_image(
                x_hat,
                f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/test_recons_{self.logger.name}_{self.current_epoch}.png",
                normalize=True,
                nrow=8
            )