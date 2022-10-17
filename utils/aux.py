#### Libraries

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