## Libraries

from os import listdir
from os.path import join

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from matplotlib.image import imread

## Functions and Classes

class CIFAR10Dataset(Dataset):
   def __init__(
      self,
      data_dir,
      test_ratio,
      train,
      per_image_normalize,
      test_random_seed, 
      transform
   ):
      super(CIFAR10Dataset, self).__init__()

      self.data_dir = data_dir
      self.test_ratio = test_ratio
      self.test_random_seed = test_random_seed
      self.transform = transform
      self.train = train
      self.per_image_normalize = per_image_normalize

      self.images_adresses = []
        
      for image in listdir(self.data_dir):
         self.images_adresses.append(
            join(self.data_dir, image)
         )

      if self.test_random_seed is None:
         self.train_images_adresses, self.test_images_adresses = train_test_split(self.images_adresses, test_size=self.test_ratio)
      else:
         self.train_images_adresses, self.test_images_adresses = train_test_split(self.images_adresses, test_size=self.test_ratio, random_state=self.test_random_seed)


   def __len__(self):
      if self.train:
         return len(self.train_images_adresses)
      else:
         return len(self.test_images_adresses)

      
   def __getitem__(self, index):
      if self.train:
         image = imread(self.train_images_adresses[index])
      else:
         image = imread(self.test_images_adresses[index])

      trans = transforms.Compose([
         transforms.ToTensor()
      ])

      image = trans(image)
      image = image[:3, :, :]

      if self.per_image_normalize:
         std, mean = torch.std_mean(image, dim=(1,2), unbiased=False)
         norm_trans = transforms.Normalize(mean=mean, std=std)
         image = norm_trans(image)
      
      if self.transform is not None:
         image = self.transform(image)
        
      return image, image.size()