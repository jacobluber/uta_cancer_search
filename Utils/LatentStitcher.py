#### Libraries

from os import listdir, remove
from os.path import join

import pandas as pd

from Utils.aux import load_latent_space

#### Functions and Classes

class LatentStitcher:
    def __init__(self, latent_directory):
        self.latent_directory = latent_directory

    
    @property
    def parsed_names(self):
        parsed_names = {}

        for name in listdir(self.latent_directory):
            if name.endswith(".data"):
                _, fname, coord = name.split("_")
                coord = coord.split(".data")[0]

                if fname in parsed_names.keys():
                    parsed_names[fname].append(self._str_coord_to_tuple(coord))
                else:
                    parsed_names[fname] = [self._str_coord_to_tuple(coord)]

        return parsed_names


    def stitch(self):
        gdc_meta = pd.read_csv("/home/data/gdc/metadata.csv")
        results = pd.DataFrame(columns=["filename", "sampled_coords", "primary_site", "latent_value"])

        for fname, coords in self.parsed_names.items():
            meta = gdc_meta[gdc_meta["filename"] == fname + ".svs"]
            primary_site = meta["primary_site"].to_list()[0]
            for coord in coords:
                latent = load_latent_space(join(self.latent_directory, f"pred_{fname}_({coord[0]},{coord[1]}).data"))
                results.loc[len(results.index)] = [fname, coord, primary_site, latent.cpu()]
        
            self._clean_directory(fname)
        
        results.to_csv(join(self.latent_directory, "latent_spaces.csv"), index=False)


    def _str_coord_to_tuple(self, str_coord):
        coord = str_coord.strip(")(").split(",")
        coord_list = [int(c) for c in coord]
        return tuple(coord_list)

    
    def _clean_directory(self, fname):
        for name in listdir(self.latent_directory):
            if name.startswith(f"pred_{fname}") and name.endswith(".data"):
                remove(join(self.latent_directory, name))
