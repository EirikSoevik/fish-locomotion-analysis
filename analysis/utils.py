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

def midline_statistics(midlines_x, midlines_y, plotting):
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


def fourier_analysis(midline_x, midline_y, std_length,T):

    x_vec = midline_x[:,-1]
    y_vec = midline_y[:,-1]

    N = len(x_vec) # number of sample points

    #T = 1/50 # sample spacing

    y_vec = y_vec - np.mean(y_vec)

    f_x = fftfreq(N,T)[:N//2]
    f_y = fft(y_vec)

    f_y_abs = 2.0 / N * np.abs(f_y[0:N // 2])
    print("plotting in fourier_analysis")
    aplt.fourier_plot(f_x,f_y,N)

    #aplt.fourier_animation(f_x,f_y,N)

    #aplt.fourier_plot(f_x,f_y,N)

    return f_x, f_y

def fourier_analysis_all(midline_x, midline_y, T, plotting):
    """ FFT of time history of each point

    Removes mean position from signal so it doesn't disturb the results
    Returns absolute value of fourier analysis result
    """

    time_dim, space_dim = midline_x.shape
    N = time_dim # number of sample points
    f_x = fftfreq(N, T)[:N // 2]
    f_y = np.zeros([len(f_x), space_dim])
    f_y_c = np.zeros([len(f_x), space_dim])
    phase = np.zeros([len(f_x), space_dim])

    for s in range(space_dim):
        x_vec = midline_x[:, s]
        y_vec = midline_y[:, s]

        y_vec = y_vec - np.mean(y_vec)
        f_y1 = np.fft.rfft(y_vec)
        f_y[:,s] = 2.0 / N * np.abs(f_y1[0:N // 2])
        f_y_c[:,s] = f_y1[0:N//2]
        phase[:,s] = np.angle(f_y1[0:N//2])


    return f_x, f_y, N, f_y_c


def fft_inverse_frequency_filter(f_x, f_y):
    """ Filters out higher frequencies in the frequency plane (after fourier analysis has been performed)

    Note that the dominant frequency might be due to the fish moving and so the lowest peak might need to be subtracted
    """

    time_it, space = f_y.shape
    #midlines_y_filtered = np.zeros([time_it, space])

    f_dom = np.zeros([space])
    max_y_it = np.zeros([space]).astype(int)
    new_y_vec = {}
    old_y = {}
    max_y = np.zeros([space])

    for s in range(space):
        max_y[s] = f_y[:,s].max()
        max_y_it[s] = f_y[:,s].argmax()
        f_dom[s] = f_x[max_y_it[s]]


        # filter out everything greater than 1.5 times dominant frequency
        new_y_vec[s] = ifft(f_y[0:int(np.ceil(max_y_it[s]*1.5)),s])
        old_y[s] = ifft(f_y[:,s])

    if np.all(f_dom == f_dom[0]):
        print("All midline points have dominant frequency in y-dir as " + str(f_dom[0]) + " Hz")
    else:
        for freq in f_dom:
            print("Dominant frequency: " + str(freq))

    return f_dom, new_y_vec, max_y

def rfft_2D(midlines_x,midlines_y,  std_length):
    """Input: mirrored midlines"""
    from scipy.fft import rfft2
    import matplotlib.cm as cm

    #f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, sharex='col', sharey='row')
    #xf = np.zeros((N,N))
    #xf = np.zeros((N,N))

    time, space = midlines_x.shape

    xf = np.zeros([time,space//2])
    yf = np.zeros([time,space])

    for t in range(time):
        yf[t,:] = fft(midlines_y[t, :])
        xf[t,:] = fftfreq(space, std_length)[:space//2]

        # plt.figure()
        # plt.plot(yf[t,:])
        # plt.show()
        # #time.sleep(1)
    print("Plotting in rfft_2D")
    aplt.fft_plot(xf,yf)

    return xf, yf

def approximate_steering():
    """Find y_1(x) = C(x²+gamma*x+beta)"""

def find_position(midlines_x, midlines_y, sample_iteration=0, tol=0.1):
    """Finds the same body position in time so that one period can be analysed"""

    time, space = midlines_x.shape

    sample_1 = np.vstack((midlines_x[sample_iteration,:], midlines_y[sample_iteration,:])).T
    sample_x = midlines_x[sample_iteration,:]
    sample_y = midlines_y[sample_iteration,:]
    p_it = -1

    for t in range(sample_iteration+1,time):
            #x_diff = sum(midlines_x[t,:]-sample_x)
            y_diff = sum(midlines_y[t,:]-sample_y)
            if y_diff<tol:
                p_it = t
                sample_match_x = midlines_x[t,:]
                sample_match_y = midlines_y[t,:]

    if p_it < 0:
        raise Exception("Could not find matching position in time")

    print("Start it: " + str(sample_iteration) + "end it: " + str(p_it))
    return p_it, sample_match_x, sample_match_y

def space_mirroring(midlines_x,midlines_y):
    """Mirrors midlines in the x-direction to be able to utilize fourier analysis"""
    time, space = midlines_x.shape

    midlines_x_mirrored = np.zeros([time, space*2])
    midlines_y_mirrored = np.zeros([time, space*2])

    for t in range(time):
        midlines_x_mirrored[t,0:space] = -np.flip(midlines_x[t,:])
        midlines_x_mirrored[t,space:2*space] = midlines_x[t,:]

        midlines_y_mirrored[t,0:space] = np.flip(midlines_y[t,:])
        midlines_y_mirrored[t,space:2*space] = midlines_y[t,:]

        # for s in range(space-1,-1,-1):
        #     midlines_x_mirrored[t, s] = -midlines_x[t, s]
        #     midlines_y_mirrored[t, s] = midlines_y[t, s]
        #
        # for s in range(space-1):
        #     midlines_x_mirrored[t, s] = midlines_x[t, s + space]
        #     midlines_y_mirrored[t, s] = midlines_y[t, s + space]

        # plt.figure()
        # plt.plot(midlines_x_mirrored[t,:],midlines_y_mirrored[t,:])
        # plt.show()

    return midlines_x_mirrored, midlines_y_mirrored

def lateral_displacement(midlines_x, midlines_y,T):

    time_dim, space = midlines_x.shape
    y_displacement = np.zeros([space])

    for s in range(space):

        y_mean = np.mean(midlines_y[:,s])
        y_displacement[s] = (np.abs(np.max(midlines_y[:,s]))-y_mean)/(np.max(midlines_x)-np.min(midlines_x))

    return y_displacement

def curve_fit_second_order(x_vec, y_vec, order=2, output_length=100):

    #time_dim, space_dim = x_vec.shape
    #xfit = {}
    #yfit = {}

    #for t in range(time_dim):
    polynomial = np.polyfit(x_vec, y_vec, order)
    print("Polynomial for approximating midline amplitudes is given by: " + str(polynomial))
    xfit = np.linspace(min(x_vec), max(x_vec), output_length)
    yfit = np.polyval(polynomial, xfit)

    return xfit, yfit, polynomial

