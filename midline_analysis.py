import numpy as np
from analysis import utils as autil
#from analysis import locomotion_extraction as loce
#import matplotlib
#matplotlib.use('Qt5Agg')
# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
# import util
import os
from analysis import plotting as aplt

from scipy.io import savemat


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

    save = False # plotting must also be true for save to work
    plotting = False
    spline_length = 15
    sample_spacing = 1 / 50  # sample spacing i.e. samples per second

    # Load attributes
    dict_list = autil.get_attributes(my_dir)
    coords_x, coords_y = autil.get_coords(my_dir=my_dir, dict_list=dict_list)
    centroid = autil.get_centroid(dict_list=dict_list)
    midlines_x, midlines_y = autil.get_midlines(my_dir=my_dir, dict_list=dict_list)

    time_dim, space_dim = midlines_x.shape
    time_vec = np.arange(0, time_dim, 1, dtype=float)
    # TODO: fix: time_vec/sample_spacing = 227 instead of 228
    time_vec = time_vec*sample_spacing # time vector in seconds


    # Some computations, plotting
    mean_length, std_length, l = autil.midline_statistics(midlines_x, midlines_y)
    new_midlines_x, new_midlines_y, new_centroid,\
    new_coords_x, new_coords_y = autil.axis_transformation(midlines_x, midlines_y, centroid, coords_x, coords_y)
    new_mean_length, new_std_length, new_l = autil.midline_statistics(new_midlines_x, new_midlines_y) # plots

    # Spline
    spline_x, my_splines = autil.midline_spline(new_midlines_x,new_midlines_y,spline_length,interp_kind='linear')

    ## Fourier analysis

    #       Choose only a few swimming periods, for example 5 periods:
    start_frame = 3
    end_frame = 83
    frame_len = end_frame-start_frame

    if plotting:
        aplt.frame_output(coords_x, coords_y, midlines_x, midlines_y, centroid, start_frame=start_frame, end_frame = end_frame, my_title="Starting and ending pos. for analysis", save=save, save_dir=save_dir)


    norm_x = new_midlines_x[start_frame:end_frame, :]
    norm_y = new_midlines_y[start_frame:end_frame, :]
    midlines_x = midlines_x[start_frame:end_frame, :]
    midlines_y = midlines_y[start_frame:end_frame, :]
    centroid = centroid[start_frame:end_frame]
    time_vec = time_vec[start_frame:end_frame]
    norm_x_vec = norm_x[0,:]

    if plotting:
        aplt.all_midlines_in_one(midlines_x, midlines_y, save_dir=save_dir, my_title="all midlines transverse",
                                 longitudinal_lines=False, save=save)
        aplt.all_midlines_in_one(midlines_x, midlines_y, save_dir=save_dir, my_title="all midlines transverse",
                                 longitudinal_lines=True, save=save)
        aplt.body_midline_centroid_animation(coords_x[start_frame:end_frame,:],coords_y[start_frame:end_frame,:],midlines_x,midlines_y,centroid)

    #       To verify
    #aplt.point_in_time(midlines_y, "midline_y end", save_dir=save_dir, pos=-1, save=False)

    f_x1, f_y, N, f_y_total, phase, f_x = autil.fourier_analysis_all(midlines_x,midlines_y, sample_spacing, plotting)
    f_dom, filtered_midlines_y, y_max_f, phase_dom = autil.fft_analysis(f_x1, f_y_total, midlines_y, phase, plotting)

    if plotting:
        aplt.fourier_plot(f_x, f_y, N, "fourier_plot", save_dir=save_dir, save=save)

    #aplt.body_midline_centroid_animation(coords_x[start_frame:end_frame, :], coords_y[start_frame:end_frame, :],
    #                                     midlines_x, filtered_midlines_y, centroid)

    phase_dom = autil.phase_filter(phase_dom)
    y_max_t = autil.lateral_displacement(mean_length,midlines_y)
    #y_displacement_fourier = autil.fourier_lateral_displacement(f_y, f_dom_arg)

    amp_x_f, amp_y_f, polynomial_f = autil.curve_fit(x_vec=new_midlines_x[0, :], y_vec=y_max_f, order=2, output_length=20)
    amp_x_t, amp_y_t, polynomial_t = autil.curve_fit(x_vec=new_midlines_x[0, :], y_vec=y_max_t, order=2, output_length=20)



    # TODO: find wavenumber
    wave_length, k = autil.wave_number(phase_dom, filtered_midlines_y, f_dom, sample_spacing, norm_x_vec)

    #thingy_xfit, thingy_yfit, Pm_vec, Pm = autil.combined_thingy(k, phase_dom, norm_x[0,:], plotting, "thingy_plot", save_dir, save=False)

    #steady_motion = autil.steady_motion(polynomial_t, f_dom[0], time_vec, Pm, norm_x[0,:])

    trav_index = autil.trav_index(midlines_y, sample_spacing)

    s2, s3, s4 = autil.G_approx(amp_y_t, norm_x[0,:], norm_y, k, wave_length)
    aplt.fish_characteristics(wave_length, s2, s3, s4, trav_index)

    if plotting:
        aplt.fft_evaluation2(midlines_y, filtered_midlines_y, time_vec, wait=0.2)

        #aplt.histogram(l, "midline length", save_dir, save=False)
        #aplt.histogram(new_l, "new midline length", save_dir, save=False)

        #aplt.point_in_time(midlines_y, "midline_y end", save_dir=save_dir, pos=-1, save=save)
        #aplt.point_in_time(midlines_x, "midline_x end", save_dir=save_dir, pos=-1, save=save)
        aplt.point_in_time(midlines_y, "midline_y mid", save_dir=save_dir, pos=10, save=save)
        #aplt.point_in_time(midlines_x, "midline_x start", save_dir=save_dir, pos=0, save=save)
        #aplt.point_in_time(centroid, "centroid_y", save_dir=save_dir, pos=-1, save=save)
        #aplt.point_in_time(centroid, "centroid_x", save_dir=save_dir, pos= 0, save=save)

        #aplt.all_point_in_time(midlines_y, "all_midline_y", save_dir=save_dir, save=save)
        #aplt.all_point_in_time(midlines_x, "all_midline_x", save_dir=save_dir, save=save)

        aplt.midlines_centroid_in_time(midlines_x,midlines_y,centroid, my_title="Midline points in time", save_dir=save_dir, save=save)


        #aplt.all_splines_plot(spline_x, my_splines, "all_splines", save_dir=save_dir, save=save)
        aplt.fourier_plot(f_x, f_y_total, N, "fourier_plot", save_dir=save_dir, save=save)

        aplt.local_maxima_plot(amp_x_t, amp_y_f, amp_y_f, y_max_f, y_max_t, polynomial_t, polynomial_f, save_dir, save)
        #aplt.fourier_animation(f_x,f_y,N)

    #for t in range(len(midlines_x)):
    #    loce.least_squares(midlines_x[t,:], midlines_y[t,:])

    #save=True
    if save:
        savemat(save_dir+'midlines_y.mat', {'my_data': midlines_y})
    print("Finished main")


if __name__ == "__main__":
    main()

