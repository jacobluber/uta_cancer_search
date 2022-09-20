from __future__ import print_function
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import join
from utils import *
from multiprocessing import Pool
import torch
import pyvips
import random
import pickle

class SvsDatasetFromFolder(Dataset):    
    def __init__(self, dataset_dir, patch_size, num_patches, num_workers, write_coords, read_coords, custom_coords_file, transforms=None):
        super(SvsDatasetFromFolder, self).__init__()
        self.imageFilenames = []
        self.transforms = transforms
        self.imageFilenames.extend(join(dataset_dir, x) for x in sorted(listdir(dataset_dir)))
        self.dirLength = len(self.imageFilenames)
        self.patchSize = patch_size
        self.numPatches = num_patches
        if read_coords:
            with open(custom_coords_file,'rb') as filehandle:
                self.patch_coords = pickle.load(filehandle)
                filehandle.close()
        if not read_coords:
            pool = Pool(processes=num_workers)
            print("pool")
            pool_out = pool.map(self._fetch_coords,self.imageFilenames)
            self.patch_coords = [elem for sublist in pool_out for elem in sublist]
            random.shuffle(self.patch_coords)
        if write_coords:
            with open('patch_coords.data','wb') as filehandle:
                pickle.dump(self.patch_coords,filehandle)
                filehandle.close()
        if not read_coords:
            assert len(self.patch_coords) == self.dirLength * self.numPatches               
    def _fetch_coords(self,fname):
        print(fname,flush=True)
        img = self._load_file(fname)
        patches = self._patching(img)
        dirs = [fname] * len(patches)
        return list(zip(dirs,patches))
    def _load_file(self,file):
        image = pyvips.Image.new_from_file(str(file))
        return image
    def _get_intersection(self,a_x,a_y,b_x,b_y): #tensors are row major
        if abs(a_x - b_x) < self.patchSize and abs(a_y - b_y) < self.patchSize:
            return True
        else:
            return False
    def _get_intersections(self,x,y,coords):
        if len(coords) == 0:
            return True
        else:
            ml = set(map(lambda b: self._get_intersection(b[0],b[1],x,y), coords))
            if True in ml:
                return False
            else: 
                return True
    def _filter_whitespace(self,tensor_3d):
        r = np.mean(np.array(tensor_3d[0]))
        g = np.mean(np.array(tensor_3d[1]))
        b = np.mean(np.array(tensor_3d[2]))
        channel_avg = np.mean(np.array([r,g,b]))
        if channel_avg < .82:
            return True
        else:
            return False
    def _img_to_tensor(self,img,x,y):
        t = img.crop(x,y,self.patchSize,self.patchSize)
        t_np = vips2numpy(t)
        #tt_np = transforms.ToTensor()(t_np)
        out_t = self.transforms(t_np)
        out_t = out_t[:3,:,:]
        return out_t
    def _patching(self, img):
        count = 0
        coords = []
        while count < self.numPatches: #[4, x , y] -> many [4, 512, 512]
                rand_i = random.randint(0,img.width-self.patchSize)
                rand_j = random.randint(0,img.height-self.patchSize)
                temp = self._img_to_tensor(img,rand_i,rand_j)
                if self._filter_whitespace(temp):
                    if self._get_intersections(rand_j,rand_i,coords):
                        coords.append((rand_i,rand_j))
                        count+=1
        return coords
    def __getitem__(self, index):
        fname, coord_tuple = self.patch_coords[index]
        img = self._load_file(fname)
        out = self._img_to_tensor(img,coord_tuple[0],coord_tuple[1])
        return out, out.size()
    def __len__(self):
        return len(self.patch_coords)

