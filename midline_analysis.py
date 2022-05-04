import numpy as np
from analysis import utils as autil
#import matplotlib
#matplotlib.use('Qt5Agg')
# matplotlib.use('QtAgg')
#import matplotlib.pyplot as plt
# import util
import os
from analysis import plotting as aplt

def main():
    """TODO:
    - constant midline length constraint?
    - improve midline extreme points
    - make sure midline follows fish and doesn't always end at 1
    - smooth spline
    - read all of "Optimal undulatory swimming..." Maertens et. al
    - how to implement the factorial C

    Questions:
    - what is the unit axis in a fourier plot?

    - find period from simple point, or by plotting
    - find multiple periods (2 or 3) -> find average
    - find wave freq from least square method
    - fft for removing noise (can do in time domain as well)
        mirroring
        fft
        cut out higher freq.
        inverse fft to reconstruct original signal
        remove mirroring
    """

    #my_dir = "my_data/mask_output_Feb-17-2022_1216/masks/"
    my_dir = "my_data/mask_output_april_Apr-27-2022_1618/masks/"
    save_dir = "my_data/mask_output_april_Apr-27-2022_1618/"
    dir_files = os.listdir(my_dir)

    save = False#True
    plotting = True
    spline_length = 15
    T = 1 / 50  # sample spacing

    # Load attributes
    dict_list = autil.get_attributes(my_dir)
    coords_x, coords_y = autil.get_coords(my_dir=my_dir, dict_list=dict_list)
    centroid = autil.get_centroid(dict_list=dict_list)
    midlines_x, midlines_y = autil.get_midlines(my_dir=my_dir, dict_list=dict_list)


    # Some computations, plotting
    mean_length, std_length, l = autil.midline_statistics(midlines_x, midlines_y, plotting)
    new_midlines_x, new_midlines_y, new_centroid,\
    new_coords_x, new_coords_y = autil.axis_transformation(midlines_x, midlines_y, centroid, coords_x, coords_y)
    new_mean_length, new_std_length, new_l = autil.midline_statistics(new_midlines_x, new_midlines_y, plotting) # plots

    # Spline
    spline_x, my_splines = autil.midline_spline(new_midlines_x,new_midlines_y,spline_length,interp_kind='linear')

    ## Fourier analysis


    #       Choose only a few swimming periods, for example 5 periods:
    start_frame = 0
    end_frame = 79
    midlines_x = midlines_x[start_frame:end_frame,:]
    midlines_y = midlines_y[start_frame:end_frame,:]
    centroid = centroid[start_frame:end_frame]

    #       To verify
    #aplt.point_in_time(midlines_y, "midline_y end", save_dir=save_dir, pos=-1, save=False)

    f_x, f_y, N, f_y_c = autil.fourier_analysis_all(midlines_x,midlines_y, T, plotting)
    f_dom, new_y_vec, y_max_fourier = autil.fft_inverse_frequency_filter(f_x, f_y)
    xfit, yfit, polynomial = autil.curve_fit_second_order(x_vec=new_midlines_x[0,:],y_vec=y_max_fourier,order=2,output_length=100)

    #p_it, sample_match_x, sample_match_y = autil.find_position(midlines_x=midlines_x,midlines_y=midlines_y, sample_iteration=0,tol=0.1)
    y_displacement_time = autil.lateral_displacement(midlines_x,midlines_y,T)
    #y_displacement_fourier = autil.fourier_lateral_displacement(f_y, f_dom_arg)

    if plotting == True:
        #aplt.histogram(l, "midline length", save_dir, save=False)
        #aplt.histogram(new_l, "new midline length", save_dir, save=False)

        #aplt.point_in_time(midlines_y, "midline_y end", save_dir=save_dir, pos=-1, save=save)
        #aplt.point_in_time(midlines_x, "midline_x end", save_dir=save_dir, pos=-1, save=save)
        #aplt.point_in_time(midlines_y, "midline_y start", save_dir=save_dir, pos=0, save=save)
        #aplt.point_in_time(midlines_x, "midline_x start", save_dir=save_dir, pos=0, save=save)
        #aplt.point_in_time(centroid, "centroid_y", save_dir=save_dir, pos=-1, save=save)
        #aplt.point_in_time(centroid, "centroid_x", save_dir=save_dir, pos= 0, save=save)

        #aplt.all_point_in_time(midlines_y, "all_midline_y", save_dir=save_dir, save=save)
        #aplt.all_point_in_time(midlines_x, "all_midline_x", save_dir=save_dir, save=save)

        aplt.midlines_centroid_in_time(midlines_x,midlines_y,centroid, my_title="Midline points in time", save_dir=save_dir, save=save)


        #aplt.all_splines_plot(spline_x, my_splines, "all_splines", save_dir=save_dir, save=save)
        aplt.fourier_plot(f_x, f_y, N, "fourier_plot", save_dir=save_dir, save=save)
        aplt.local_maxima_plot(new_midlines_x[0,:],y_max_fourier, polynomial, mean_length, my_title="Midline y maxima in BL",save_dir=save_dir,save=save)
        #aplt.fourier_animation(f_x,f_y,N)

    print("Finished main")


if __name__ == "__main__":
    main()
