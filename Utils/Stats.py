#### Libraries

from os import makedirs
from os.path import exists, join

import torch
import numpy as np

#### Functions and Classes

class DataLoaderStats:
    def __init__(self, loader, stats_dir):
        if not exists(stats_dir):
            self._create_dir(stats_dir)

        # Accumulating means and stds
        self.std = torch.tensor([0, 0, 0])
        self.mean = torch.tensor([0, 0, 0])

        for image_batch, _ in loader:
            # Last batches may be smaller than self.batch_size
            batch_size = image_batch.shape[0]
            for batch_id in range(batch_size):
                std, mean = torch.std_mean(image_batch[batch_id], dim=(1,2), unbiased=False)

                # Not the correct calculation, but a good enough estimate.
                #TODO->Find an efficient implementation of Welford's algorithm
                self.std = self.std + std
                self.mean = self.mean + mean
        
        print(len(loader.dataset))
        self.std = self.std / len(loader.dataset)
        self.mean = self.mean / len(loader.dataset)

        np.savetxt(join(stats_dir, "std.gz"), self.std.numpy())
        np.savetxt(join(stats_dir, "mean.gz"), self.mean.numpy())

        print(f"Stats are calculated for the training set and are saved in: {stats_dir}")


    def get_std(self):
        return self.std


    def get_mean(self):
        return self.mean
    

    def _create_dir(self, directory):
        if not exists(directory):
            makedirs(directory)
            print(f"directory created: {directory}")