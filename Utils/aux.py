#### Libraries

from os import makedirs
from os.path import exists, join
import pickle

import numpy as np

#### Constants

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

#### Functions and Classes

def vips2numpy(vi):
    return np.ndarray(
        buffer=vi.write_to_memory(),
        dtype=format_to_dtype[vi.format],
        shape=[vi.height, vi.width, vi.bands]
    )


def create_dir(directory):
    if not exists(directory):
        makedirs(directory)
        print(f"directory created: {directory}")


def save_transformation(transformation, filename, verbose=True):
    with open(filename, 'wb') as filehandle:
        pickle.dump(transformation, filehandle)
    if verbose:
        print(f"Transformations successfully saved as: {filename}")


def load_transformation(filename, verbose=True):
    with open(filename, 'rb') as filehandle:
        transformation = pickle.load(filehandle)
    if verbose:
        print(f"Transformations successfully loaded from: {filename}")
    return transformation


def save_latent_space(transformation, filename, verbose=False):
    with open(filename, 'wb') as filehandle:
        pickle.dump(transformation, filehandle)
    if verbose:
        print(f"Latent space successfully saved as: {filename}")


def load_latent_space(filename, verbose=False):
    with open(filename, 'rb') as filehandle:
        transformation = pickle.load(filehandle)
    if verbose:
        print(f"Latent space successfully loaded from: {filename}")
    return transformation