import numpy as np
import os
from util.utils import find_centroid
from scipy.interpolate import interp1d
import analysis.plotting as aplt
from scipy.fft import fft,fftfreq, ifft
import matplotlib.pyplot as plt


def get_attributes(my_dir):
    """Creates a time-sorted dictionary with all information, which can then be looped to access specifics"""

    dir_files = os.listdir(my_dir)
    i = 0
    dir_time_code = []
    for this_file in dir_files:
        # This sorts the files into the correct time, i.e. from 0 to t_end. Make sure the sign below is the correcct for
        # the naming nomenclature used in detectron2!
        dir_time_code.append([this_file.partition("_")[2], i])
        i += 1

    dir_time_code.sort()
    dict_list = {}

    for ii in range(len(dir_time_code)):
        time_code, element = dir_time_code[ii]

        dict_list[dir_files[element]] = {}
        dict_list[dir_files[element]]["time_code"] = time_code
        dict_list[dir_files[element]]["coords"] = np.load(my_dir + dir_files[element] + "/coords.npy")
        dict_list[dir_files[element]]["centroid"] = np.load(my_dir + dir_files[element] + "/centroid.npy")
        dict_list[dir_files[element]]["midline"] = np.load(my_dir + dir_files[element] + "/midline.npy")
        dict_list[dir_files[element]]["midline_angles"] = np.load(my_dir + dir_files[element] + "/midline_angles.npy")
        dict_list[dir_files[element]]["midline_angles_change.npy"] = np.load(
            my_dir + dir_files[element] + "/midline_angles_change.npy")

    return dict_list


def get_centroid(dict_list):
    centroid = np.zeros([len(dict_list), 2])
    i = 0

    for my_dict in dict_list:
        centroid[i] = dict_list[my_dict]["centroid"]
        i += 1

    return centroid


def get_midlines(my_dir, dict_list):
    """Creates a matrix of midlines, where each row is a midline for a given time"""

    dir_files = os.listdir(my_dir)
    midlines_x = np.zeros([len(dict_list), len(dict_list[dir_files[0]]["midline"])])
    midlines_y = np.zeros([len(dict_list), len(dict_list[dir_files[0]]["midline"])])
    i = 0

    for my_dict in dict_list:
        midlines_x[i, :] = dict_list[my_dict]["midline"][:, 0]
        midlines_y[i, :] = dict_list[my_dict]["midline"][:, 1]
        i += 1

    return midlines_x, midlines_y


def get_coords(my_dir, dict_list):
    """Creates a matrix of coords, where each row is a midline for a given time"""

    dir_files = os.listdir(my_dir)
    coords_x = np.zeros([len(dict_list), len(dict_list[dir_files[0]]["coords"])])
    coords_y = np.zeros([len(dict_list), len(dict_list[dir_files[0]]["coords"])])
    i = 0

    for my_dict in dict_list:
        coords_x[i, :] = dict_list[my_dict]["coords"][:, 0]
        coords_y[i, :] = dict_list[my_dict]["coords"][:, 1]
        i += 1

    return coords_x, coords_y


def axis_transformation(midlines_x, midlines_y, centroid, coords_x, coords_y):
    """Transformes the axes, the new origo is at the head of the midline

    As instream velocity only comes from the left, the direction of the axes never change, only the position,
    relative to the original frame.
    """

    rows, cols = midlines_x.shape
    c_rows, c_cols = coords_x.shape

    # TODO: best way to define x?
    midline_new_x_cols = np.linspace(0, 1, cols)
    midline_new_x = np.zeros([rows, cols])
    for i in range(rows):
        midline_new_x[i, :] = midline_new_x_cols
    new_centroid = np.zeros([rows, 2])
    midline_new_y = np.zeros([rows, cols])
    new_coords_x = np.zeros([c_rows, c_cols])
    new_coords_y = np.zeros([c_rows, c_cols])

    for r in range(rows):
        origo_x = midlines_x[r, 0]
        origo_y = midlines_y[r, 0]
        midline_new_x[r, 0] = 0
        midline_new_y[r, 0] = 0

        for c in range(1, cols):
            midline_new_x[r, c] = (midlines_x[r, c] - origo_x) / (np.max(coords_x[r, :]) - np.min(coords_x[r, :]))
            midline_new_y[r, c] = (midlines_y[r, c] - origo_y) / (np.max(coords_x[r, :]) - np.min(coords_x[r, :]))

    for r_c in range(c_rows):
        origo_x = midlines_x[r_c, 0]
        origo_y = midlines_y[r_c, 0]

        for c_c in range(c_cols):
            new_coords_x[r_c, c_c] = (coords_x[r_c, c_c] - origo_x) / (np.max(coords_x[r_c, :]) - np.min(coords_x[r_c, :]))
            new_coords_y[r_c, c_c] = (coords_y[r_c, c_c] - origo_y) / (np.max(coords_x[r_c, :]) - np.min(coords_x[r_c, :]))

        new_centroid[r_c, 0] = (centroid[r_c, 0] - origo_x) / (np.max(coords_x[r_c, :]) - np.min(coords_x[r_c, :]))
        new_centroid[r_c, 1] = (centroid[r_c, 1] - origo_y) / (np.max(coords_x[r_c, :]) - np.min(coords_x[r_c, :]))

    return midline_new_x, midline_new_y, new_centroid, new_coords_x, new_coords_y

def midline_statistics(midlines_x, midlines_y):
    """Approximates a mean midline length based on all midlines"""
    time, space = midlines_x.shape
    dl = np.zeros([time, space-1])
    l = np.zeros(time)
    mean_dl = np.zeros(time)
    std_dl = np.zeros(time)

    for t in range(time):
        for s in range(1,space):
            dl[t,s-1] = np.sqrt((midlines_x[t, s] - midlines_x[t, s-1])**2 + (midlines_y[t, s]- midlines_y[t, s-1])**2)

        l[t] = np.sum(dl[t,:])
        mean_dl[t] = np.mean(dl[t,:])
        std_dl[t] = np.std(dl[t,:])

    mean_dl_tot = np.mean(mean_dl)
    std_dl_tot = np.mean(std_dl)
    std_length = np.std(l)
    mean_length = np.mean(l)


    print("Property          | Mean    | Std. dev ")
    print("----------------------------------------")
    print("Midline length     {:3.2f}     {:1.2f}".format(mean_length,std_length))
    print("dL/dx               {:2.2f}     {:1.2f}".format(mean_dl_tot,std_dl_tot))
    print("----------------------------------------")
    return mean_length, std_length, l

def midline_spline(midlines_x, midlines_y, spline_length, interp_kind = 'cubic'):

    time, space = midlines_x.shape
    spline_x = np.linspace(0, 1, spline_length)
    my_splines = {}

    for t in range(time):
        my_splines[t] = interp1d(midlines_x[t,:], midlines_y[t,:], kind=interp_kind, fill_value='extrapolate')

    return spline_x, my_splines

def axis_transformation2(midlines_x, midlines_y, centroid, coords_x, coords_y, spline_x, my_splines, mean_length):
    """Keeps uniform midline length"""

    rows, cols = midlines_x.shape
    c_rows, c_cols = coords_x.shape

    # TODO: best way to define x?
    midline_new_x_cols = np.linspace(0, 1, cols)
    midline_new_x = np.zeros([rows, cols])
    for i in range(rows):
        midline_new_x[i, :] = midline_new_x_cols
    new_centroid = np.zeros([rows, 2])
    midline_new_y = np.zeros([rows, cols])
    new_coords_x = np.zeros([c_rows, c_cols])
    new_coords_y = np.zeros([c_rows, c_cols])

    for r in range(rows):
        origo_x = midlines_x[r, 0]
        origo_y = midlines_y[r, 0]
        midline_new_x[r, 0] = 0
        midline_new_y[r, 0] = 0

        for c in range(1, cols):
            midline_new_x[r, c] = (midlines_x[r, c] - origo_x) / (np.max(coords_x[r, :]) - np.min(coords_x[r, :]))
            midline_new_y[r, c] = (midlines_y[r, c] - origo_y) / (np.max(coords_x[r, :]) - np.min(coords_x[r, :]))

    for r_c in range(c_rows):
        origo_x = midlines_x[r_c, 0]
        origo_y = midlines_y[r_c, 0]

        for c_c in range(c_cols):
            new_coords_x[r_c, c_c] = (coords_x[r_c, c_c] - origo_x) / (np.max(coords_x[r_c, :]) - np.min(coords_x[r_c, :]))
            new_coords_y[r_c, c_c] = (coords_y[r_c, c_c] - origo_y) / (np.max(coords_x[r_c, :]) - np.min(coords_x[r_c, :]))

        new_centroid[r_c, 0] = (centroid[r_c, 0] - origo_x) / (np.max(coords_x[r_c, :]) - np.min(coords_x[r_c, :]))
        new_centroid[r_c, 1] = (centroid[r_c, 1] - origo_y) / (np.max(coords_x[r_c, :]) - np.min(coords_x[r_c, :]))

    return midline_new_x, midline_new_y, new_centroid, new_coords_x, new_coords_y

def fourier_analysis_all(midline_x, midline_y, T, plotting):
    """ FFT of time history of each point

    Removes mean position from signal so it doesn't disturb the results
    Returns absolute value of fourier analysis result
    """

    time_dim, space_dim = midline_x.shape
    N = time_dim # number of sample points
    f_x = fftfreq(N, T)[:N // 2]
    f_x1 = fftfreq(N, T)
    f_y = np.zeros([len(f_x1), space_dim])
    #f_y_total = np.zeros([len(f_x1), space_dim])
    f_y_total = np.empty([len(f_x1), space_dim]).astype(complex)

    f_y2 = np.zeros([len(f_x), space_dim])
    f_y_c = np.zeros([len(f_x), space_dim])
    phase = np.zeros([len(f_x), space_dim])
    a = np.zeros([time_dim, space_dim])

    for s in range(space_dim):
        y_vec = midline_y[:, s]

        #y_vec = y_vec - np.mean(y_vec)
        f_y_total[:,s] = np.fft.fft(y_vec)

        f_y[:,s] = 2.0 / N * np.abs(f_y_total[:,s])
        #f_y_c[:,s] = f_y1[0:N//2]
        phase[:,s] = np.angle(f_y_total[0:N//2, s])


    return f_x1, f_y, N, f_y_total, phase

def wave_number():

    #TODO: finish this
    print()

def fft_analysis(f_x, f_y_total, midlines_y, plotting):
    """ Analysis of fft signal, finds dominating freq., wave number, etc.

    Note that the dominant frequency might be due to the fish moving and so the lowest peak might need to be subtracted
    """
    wait = 1 #plotting animation

    time_it, space = f_y_total.shape
    #midlines_y_filtered = np.zeros([time_it, space])

    m_time, m_space = midlines_y.shape
    f_dom = np.zeros([space])
    max_y_ind = np.zeros([space]).astype(int)
    filtered_midlines_y = np.zeros([m_time, m_space])
    #filtered_midlines_y1 = {}
    old_y = np.zeros([m_time, m_space])
    max_y = np.zeros([space])
    T_dom = np.zeros([space])

    nfy_time, nfy_space = f_y_total.shape
    new_f_y = np.empty([nfy_time, nfy_space]).astype(complex)
    #new_f_y = np.zeros([nfy_time, nfy_space], dtype='complex')
    new_vec = np.empty([nfy_time//2]).astype(complex)
    new_f_y = new_f_y*0
    new_vec = new_vec*0

    scale_factor = 1.5
    high_pass = 0#2
    low_pass = 6

    for s in range(space):
        new_vec = new_vec * 0
        f_y = np.abs(f_y_total[1:len(f_y_total)//2, s])

        # Dominating frequency
        max_y[s] = f_y.max()
        max_y_ind[s] = f_y.argmax()+1
        f_dom[s] = f_x[max_y_ind[s]]
        T_dom[s] = 1/f_dom[s]
        phase_dom[s] = phase[]

        new_vec[high_pass:int(np.ceil(max_y_ind[s])*scale_factor)] = f_y_total[high_pass:int(np.ceil(max_y_ind[s])*scale_factor), s]

        new_vec[high_pass+1:max_y_ind[s]] = 0
        new_vec[low_pass:-1] = 0

        #new_fy = np.concatenate(new_vec,f_y_total[nfy_time//2+1, s],new_vec[-1:0:-1])
        new_fy = np.append(new_vec, f_y_total[nfy_time//2, s])
        new_fy[nfy_time//2] = 0
        new_fy = np.append(new_fy,new_vec[-1:0:-1])

        plt.plot(new_fy)
        plt.show()

        filtered_midlines_y[:, s] = ifft(new_fy)
        old_y[:,s] = ifft(f_y_total[:, s])


    if np.all(f_dom == f_dom[0]):
        print("All midline points have dominant frequency in y-dir as " + str(f_dom[0]) + " Hz")
        print("Index is : " + str(max_y_ind[0]))
    else:
        for freq in f_dom:
            print("Dominant frequency: " + str(freq))

    import time
    plotting=True
    if plotting:
        aplt.fft_evaluation(old_y,midlines_y,filtered_midlines_y,scale_factor, wait)
    plotting=False



    return f_dom, filtered_midlines_y, max_y


def lateral_displacement(mean_length, midlines_y):

    time_dim, space = midlines_y.shape
    y_displacement = np.zeros([space])

    for s in range(space):

        y_mean = np.mean(midlines_y[:,s])
        y_displacement[s] = (np.abs(np.max(midlines_y[:,s]))-y_mean)/mean_length

    return y_displacement

def curve_fit_second_order(x_vec, y_vec, body_length, order=2, output_length=100):
    """Curve fitting of a second degree equation, here  the amplitudes of each point along the midline"""


    #for t in range(time_dim):
    polynomial = np.polyfit(x_vec, y_vec, order)
    print("Polynomial for approximating midline amplitudes is given by: " + str(polynomial))
    xfit = np.linspace(min(x_vec), max(x_vec), output_length)
    yfit = np.polyval(polynomial, xfit)

    return xfit, yfit, polynomial

