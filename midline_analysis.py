import numpy as np
from analysis import utils as autil
import matplotlib
#matplotlib.use('Qt5Agg')
#matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
#import util
import os
from analysis import plotting as aplt


def main():

    my_dir = "my_data/mask_output_Feb-17-2022_1216/masks/"
    dir_files = os.listdir(my_dir)

    save = True
    plotting = True

    dict_list = autil.get_attributes(my_dir)

    coords_x, coords_y = autil.get_coords(my_dir=my_dir, dict_list=dict_list)
    centroid = autil.get_centroid(dict_list=dict_list)
    midlines_x, midlines_y = autil.get_midlines(my_dir=my_dir, dict_list=dict_list)

    if plotting == True:
        #aplt.midline_animation(x=midlines_x,y=midlines_y)
        #aplt.midline_animation_centroid(x=midlines_x,y=midlines_y,centroid=centroid)
        aplt.body_midline_centroid(coords_x=coords_x,coords_y=coords_y,midlines_x=midlines_x,midlines_y=midlines_y,centroid=centroid)

    print("Finished main")

if __name__ == "__main__":

    main()
