import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.interpolate import interp1d

def find_centroid(coords, outdir, save=True):
    """Find geometric center of input mask

    Assuming that due to fish anatomy the geometric center of mass can be used as an approximation for the
    real center of mass. Please note that both center of mass and center of geometry can be outside the body boundary,
    although for a fish this is not a likely scenario
    """

    # TODO: what to do with centroids outside of body boundary?
    # TODO: make up for distribution of weight changes in the longitudinal direction? COG might be shifted forwards

    centroid = coords.mean(axis=0)
    if save:
        np.save(outdir + "centroid.npy", centroid)
    return centroid


def body_boundary_filter(my_array, coords, remove_points=False, lin_interp=True):
    """TODO: FIX IT
    Filters out midline points if they are outside the body boundary

    Returns filtered midline and a bool telling if the midline has been tweaked or not.
    Two modes, either remove points and shorten array, or replace with average of neighbours
    """

    nan_count = 0
    for i in range(len(my_array)):
        point = Point(my_array[i])
        polygon = Polygon(coords)
        if not polygon.contains(point):
            my_array[i] = np.nan
            nan_count += 1

    if nan_count == 0:
        print("No points outside of body boundary")
        return None, False

    print("Found points outside body boundary")
    count = 0
    if remove_points:
        my_array_filtered = np.zeros([len(my_array) - nan_count, 2])
        for i in range(len(my_array)):
            if not np.isnan(my_array[i, 0]).any():
                my_array_filtered[count] = my_array[i]
                count += 1
            else:
                print("removing point outside body boundary")

    # TODO: make more robust as it simply averages neighbouring points without checking
    #  wether they are in the body as well or not

    if lin_interp is True:
        my_array_filtered = my_array
        for i in range(len(my_array)):
            if np.isnan(my_array[i].any()):
                print("replacing point outside body boundary with lin. interpolation of neighbours"
                      " PS! Functionality not guaranteed to be perfect!")
                my_array_filtered[i] = np.mean(my_array_filtered[i - 2:i + 2])

    return my_array_filtered, True

def midline_least_squares(midlines_x,midlines_y):

    time, space = midlines_x.shape

    for t in range(time):
        midline = np.vstack([midlines_x[t, :],midlines_y[t, :]]).T
        #m, c = np.linalg.lstsq(midline,)
        #TODO: fix this?

def midline_interpolate(midlines_x, midlines_y):

    time, space = midlines_x.shape

def midline_spline(midline, spline_length, interp_kind = 'cubic'):

    my_spline = interp1d(midline[:,0], midline[:,1], kind=interp_kind)#, fill_value='extrapolate')
    spline_x = np.linspace(midline[:,0].min(),midline[:,1].max(),spline_length)

    return spline_x, my_spline

def midline_rib_approximation(coords, new_len):
    """Divides the fish in longitudinal sections and finds the mid-point of y-values to create midline

    Important: Does not account for conservation of length, does not start or end at the end points of the fish

    Divides fish into "new_len" parts in the x-direction, for each part it finds all coords in this batch, and takes
    the mean of the x- and y-positions
    """

    x_start = np.floor(coords[:, 0].min())
    x_end = np.ceil(coords[:, 0].max())
    #x_step = np.floor((x_end - x_start) / new_len)
    #x_new = np.arange(x_start, x_end, x_step)
    x_new = np.linspace(x_start,x_end,new_len+1)
    midline = np.zeros([new_len, 2])
    print("Start, end, len", str(x_start), str(x_end), str(len(x_new)))
    #print("len(x_new):",len(x_new))



    for i in range(len(x_new) - 1):
        start = x_new[i]
        end = x_new[i + 1]
        batch_y = []
        batch_x = []
        batch_count = 0

        for j in coords:
            if j[0] >= start and j[0] <= end:
                batch_y.append(j[1])
                batch_x.append(j[0])
                batch_count += 1

        if batch_count == 0:
            midline[i] = np.nan
            print("i: " + str(i))
            print("midline[i] = nan")
        elif batch_count > 0:
            midline[i, 1] = np.mean(batch_y)
            midline[i, 0] = np.mean(batch_x)
        else:
            print("Something wrong!")
            breakpoint()

    #midline[0]
    #midline[-1]
    #   calculate mean of x points, get 2 mean y values (lower, higher)
    #   calculate mid point between y values

    # filter values that are too far off?
    #filtered_midline, filtered_check = body_boundary_filter(my_array=midline, coords=coords, remove_points=False,
    #                                                        lin_interp=True)
    # print(midline)

    #if filtered_check is True:
    #    return filtered_midline
    #elif filtered_check is False:
    #    return midline
    return midline

def midline_angles(midline):
    # TODO: check which way positive angles correspond to

    angles = np.zeros(len(midline) - 1)
    angle_change = np.zeros(len(midline) - 2)

    for i in range(len(midline) - 1):
        angles[i] = np.arctan((midline[i + 1, 1] - midline[i, 1]) / (midline[i, 0] - midline[i + 1, 0]))

    for i in range(len(midline) - 2):
        angle_change[i] = angles[i + 1] - angles[i]

    return angles, angle_change
