#### Libraries

from os import listdir, remove
from os.path import join

from pyvips import Image

#### Functions and Classes

class Stitcher:
    def __init__(self, patches_directory, svs_directory):
        self.patches_directory = patches_directory
        self.svs_directory = svs_directory

    
    @property
    def parsed_names(self):
        parsed_names = {}

        for name in listdir(self.patches_directory):
            if name.endswith(".jpeg"):
                _, fname, coord = name.split("_")
                coord = coord.split(".jpeg")[0]

                if fname in parsed_names.keys():
                    parsed_names[fname].append(self._str_coord_to_tuple(coord))
                else:
                    parsed_names[fname] = [self._str_coord_to_tuple(coord)]

        return parsed_names


    def stitch(self):
        for fname, coords in self.parsed_names.items():
            # Assumes that top left is (0,0), left bottom is (0,max_j), top right is (max_i,max_j), and right bottom is (max_i,0)
            #   width = max_i * patch_size 
            #   height = max_j * patch_size
            max_i, max_j = self._max_i_j(coords)

            if (max_i, max_j) not in coords:
                print(f"not all patches available for reconstruction for image {fname}.svs")
                continue
            
            tiles = [Image.new_from_file(join(self.patches_directory, f"pred_{fname}_({i},{j}).jpeg"), access="sequential") for j in range(max_j + 1) for i in range(max_i + 1)]
            image = Image.arrayjoin(tiles, across=max_i + 1)
            image.write_to_file(join(self.svs_directory, f"pred_{fname}.tif"), compression="jpeg", tile=True)
            self._clean_directory(fname)

    
    def _max_i_j(self, coords):
        max_i = max([coord[0] for coord in coords])
        max_j = max([coord[1] for coord in coords])
        return max_i, max_j


    def _str_coord_to_tuple(self, str_coord):
        coord = str_coord.strip(")(").split(",")
        coord_list = [int(c) for c in coord]
        return tuple(coord_list)

    
    def _clean_directory(self, fname):
        for name in listdir(self.patches_directory):
            if name.startswith(f"pref_{fname}") and name.endswith(".jpeg"):
                remove(join(self.patches_directory, name))
