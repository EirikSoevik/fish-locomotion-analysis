import math
from multiprocessing.pool import ThreadPool as Pool
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import PCA
from pykdtree.kdtree import KDTree
from masbpy import io_npy
import numpy as np
import util.plotting as uplt
from masbpy.ma_mp import MASB as MASB_mp
from masbpy.ma import MASB
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import warnings

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

    nan_count = 0
    for i in range(len(my_array)):
        point = Point(my_array[i])
        polygon = Polygon(coords)
        if not polygon.contains(point):
            my_array[i] = np.nan
            nan_count += 1

    count = 0
    if remove_points == True:
        my_array_filtered = np.zeros([len(my_array) - nan_count, 2])
        for i in range(len(my_array)):
            if not np.isnan(my_array[i, 0]).any():
                my_array_filtered[count] = my_array[i]
                count += 1
            else:
                print("removing point outside body boundary")

    # TODO: make more robust as it simply averages neighbouring points without checking wether they are in the body as well or not
    if lin_interp == True:
        my_array_filtered = np.zeros([len(my_array), 2])
        for i in range(len(my_array)):
            if np.isnan(my_array[i].any()):
                print("replacing point outside body boundary with lin. interpolation of neighbours"
                      " PS! Functionality not guaranteed to be perfect!")
                my_array_filtered[count] = np.mean(my_array_filtered[count-2:count+2])

    return my_array_filtered


def midline_rib_approximation(coords, new_len, save):
    x_start = np.ceil(coords[:, 0].min())
    x_end = np.floor(coords[:, 0].max())
    x_step = np.floor((x_end - x_start) / new_len)
    x_new = np.arange(x_start, x_end, x_step)
    midline = np.zeros([new_len, 2])

    for i in range(len(x_new) - 1):
        start = x_new[i]
        end = x_new[i + 1]
        batch_y = []
        batch_x = []
        batch_count = 0

        for j in coords:
            if j[0] > start and j[0] < end:
                batch_y.append(j[1])
                batch_x.append(j[0])
                batch_count += 1

        if batch_count > 0:
            midline[i, 1] = np.mean(batch_y)
            midline[i, 0] = np.mean(batch_x)
        elif batch_count == 0:
            midline[i] = np.nan

    #   calculate mean of x points, get 2 mean y values (lower, higher)
    #   calculate mid point between y values

    # filter values that are too far off?
    filtered_midline = body_boundary_filter(my_array=midline,coords=coords,remove_points=False,lin_interp=True)
    # print(midline)

    return midline


def midline_angles(midline, save):
    # TODO: check which way positive angles correspond to

    angles = np.zeros(len(midline) - 1)
    angle_change = np.zeros(len(midline) - 2)

    for i in range(len(midline) - 1):
        angles[i] = np.arctan((midline[i + 1, 1] - midline[i, 1]) / (midline[i, 0] - midline[i + 1, 0]))

    for i in range(len(midline) - 2):
        angle_change[i] = angles[i + 1] - angles[i]

    return angles, angle_change
