import numpy as np
from analysis import utils as autil
import matplotlib
# matplotlib.use('Qt5Agg')
# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
# import util
import os
from analysis import plotting as aplt


def main():
    """TODO:
    - fast lengde på midline
    - midline begynner ved hodet og slutter ved halen
    - midline ender ikke alltid på 1
    - spline
    """
    my_dir = "my_data/mask_output_Feb-17-2022_1216/masks/"
    dir_files = os.listdir(my_dir)

    save = True
    plotting = True
    spline_length = 15

    dict_list = autil.get_attributes(my_dir)

    coords_x, coords_y = autil.get_coords(my_dir=my_dir, dict_list=dict_list)
    centroid = autil.get_centroid(dict_list=dict_list)
    midlines_x, midlines_y = autil.get_midlines(my_dir=my_dir, dict_list=dict_list)
    mean_length, std_length = autil.get_average_midline_length(midlines_x, midlines_y)

    new_midlines_x, new_midlines_y, new_centroid,\
    new_coords_x, new_coords_y = autil.axis_transformation(midlines_x, midlines_y, centroid, coords_x, coords_y)
    new_mean_length, new_std_length = autil.get_average_midline_length(new_midlines_x, new_midlines_y)

    spline_x, my_splines = autil.midline_spline(new_midlines_x,new_midlines_y,spline_length,interp_kind='linear')

    if plotting == True:
        #aplt.midline_animation(x=new_midlines_x, y=new_midlines_y)
        #aplt.midline_animation_centroid(x=midlines_x,y=midlines_y,centroid=centroid)
        #aplt.body_midline_centroid(coords_x=coords_x, coords_y=coords_y, midlines_x=midlines_x, midlines_y=midlines_y,
        #                           centroid=centroid)
        aplt.body_midline_centroid(coords_x=new_coords_x, coords_y=new_coords_y, midlines_x=new_midlines_x,
                                   midlines_y=new_midlines_y, centroid=new_centroid)
        #aplt.spline_plotting(my_splines,spline_x,new_coords_x,new_coords_y,new_midlines_x,new_midlines_y,new_centroid)

    print("Finished main")


if __name__ == "__main__":
    main()
