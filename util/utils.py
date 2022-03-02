import math
from multiprocessing.pool import ThreadPool as Pool
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import PCA
from pykdtree.kdtree import KDTree
from masbpy import io_npy
import numpy as np

from masbpy.ma_mp import MASB as MASB_mp
from masbpy.ma import MASB


def compute_normal(neighbours):
    pca = PCA(n_components=2)
    pca.fit(neighbours)
    plane_normal = pca.components_[1]  # this is a normalized normal
    # print(plane_normal)
    return plane_normal


# matplotlib.use('Qt5Agg')

def normal_smoother(D, angles, angle_threshold, batch_size):
    """this smooths noise in the normal vector"""

    angle_error = np.zeros(len(D['normals']))
    for i in range(len(D['normals'])):
        batch_error = 0
        batch_mean = np.mean(angles[i:i + batch_size])
        batch_std = np.std(angles[i:i + batch_size])

        # for j in range(i, i + batch_size):
        #    batch_sum +=
        # angle_error[i] +=


def normal_noise_detector(D, angles, angle_threshold, batch_size):
    """this calculates noise in the normal vector"""

    angle_error = np.zeros(len(D['normals']))
    for i in range(len(D['normals'])):
        batch_error = 0
        batch_mean = np.mean(angles[i:i + batch_size])
        batch_std = np.std(angles[i:i + batch_size])

        # for j in range(i,i+batch_size):
        #    batch_sum +=
        # angle_error[i] +=


def find_centroid(coord, outdir, save=True):
    '''Find geometric center of input mask

    Assuming that due to fish anatomy the geometric center of mass can be used as an approximation for the
    real center of mass. Please note that both center of mass and center of geometry can be outside the body boundary,
    although for a fish this is not a likely scenario
    '''

    centroid = coord.mean(axis=0)
    if save: np.save(outdir + "centroid.npy", centroid)
    return centroid


def normal_angles(D, outdir, save=True):
    """Calculates the angles in radians of every instance of the normal vector."""
    angles = np.zeros(len(D['normals']))
    for i in range(len(D['normals'])):
        angles[i] = D['normals'][i, 1]
        if D['normals'][i, 0] < 0:
            angles += math.pi
    if save: np.save(outdir + "normal_angles.npy", angles)
    return angles


def elliptic_geom_maker(N_geom, outdir, a=5, b=10, origin_x=0, origin_y=0, save=True):
    """Creates an ellipse with width a and height b, with a focal point in origin

    N_geom is the number of points you want. a is in x-dir, b in y-dir, origin coords
    give the center of the ellipse
    """

    T = np.linspace(0, 2 * np.pi, N_geom)
    x = np.zeros(len(T))
    y = np.zeros(len(T))
    for i in range(len(T)):
        x[i] = origin_x + a * np.cos(T[i])
        y[i] = origin_y + b * np.sin(T[i])

    elliptic_coords = np.vstack((x, y)).T
    # if save: np.save(outdir+"coords.npy",elliptic_coords)
    print("Creating elliptic geometry with N=" + str(N_geom))
    return elliptic_coords


def array_decimator(outdir, input_array, N, save=True):
    """Reduces the number of elements in a vector by a factor N, saves and returns the new vector"""

    output_array = np.zeros([len(input_array) // N, 2])
    for i in range(len(output_array)):
        output_array[i, 0] = input_array[N * i, 0]
        output_array[i, 1] = input_array[N * i, 1]

    if save:
        np.save(outdir + "new_coords", output_array)
    return output_array


# def equalize_array(outdir, array_to_model, array_to_change, save=True):
#
#    new_len =

def lin_interpol(outdir, input_vector, new_len, save_name, save=True):
    '''Unfinished'''
    old_len, dimensions  = input_vector.shape

    batch_size = old_len//new_len
    new_vec = np.zeros([new_len, dimensions])

    for i in range(len(new_vec)-batch_size):
        new_vec[i,0] = np.mean(input_vector[i:i+batch_size,0])
        new_vec[i,1] = np.mean(input_vector[i:i+batch_size,1])
    return new_vec

def decimate(outdir, input_vector, new_len, save_name, save=True):

    old_len, dim = input_vector.shape
    new_vec = np.zeros([new_len, dim])
    r = old_len//new_len # ratio coefficient

    for i in range(new_len):
        new_vec[i,0] = input_vector[i*r, 0]
        new_vec[i,1] = input_vector[i*r,1]

    if save == True:
        np.save(outdir+save_name,new_vec)

    return new_vec

import util.plotting as uplt

def k_decimations(input_vector, new_len):
    '''Picks out only new_len terms from input_vector, but iterates through so that you get many vectors'''
    old_len, dim = input_vector.shape
    k = old_len//new_len # ratio coefficient
    k_new_vecs = np.zeros([k,new_len,dim])
    k_vec_elements = np.zeros([k,new_len])

    for r in range(k-1,-1,-1):
        new_vec = np.zeros([new_len, dim])
        for i in range(new_len):
            new_vec[i,0] = input_vector[i*k-r,0]
            new_vec[i,1] = input_vector[i*k-r,1]
            k_vec_elements[r,i] = i*k-r

        k_new_vecs[r,:,:] = new_vec

    #if save == True:
    #    np.save(outdir+save_name,new_vec)

    return k_new_vecs, k, k_vec_elements

def get_elements(input_vec,elements_vec):
    '''Gives a subset of an array from a list of element positions'''
    output_vec = np.zeros(len(elements_vec))
    for i in range(len(elements_vec)):
        output_vec[i] = elements_vec[i]
    return output_vec


def k_ma_iterator(input_coords, input_normals, N_ma, outdir, save, N_normals):

    # input_vector = coords_normals
    k_coords, k_c, k_coords_elements = k_decimations(input_coords,N_ma)
    k_normals, k_n, k_normals_elements = k_decimations(input_normals, N_ma)

    D_k = [dict() for i in range(k_c)]

    for k in range(k_c):
        D_k[k]['coords'] = k_coords[k]
        D_k[k]['normals'] = k_normals[k]
        D_k[k], D_k[k]['max_r'] = compute_ma_my_func(D_k[k], outfile=outdir)
        D_k[k]['centroid'] = find_centroid(D_k[k]['coords'], outdir=outdir)
        D_k[k]['angles'] = normal_angles(D=D_k[k], outdir=outdir, save=save)
        uplt.all_in_four_2(D_k[k],k,N_ma,N_normals,D_k[k]['centroid'], original_coords=input_coords, original_normals=input_normals)

    return D_k



def array_decimator_set_len(outdir, input_array, new_len, save_name="", save=True):
    '''Sets new size of array using univariate spline from scikit-learn'''

    # Linear length along the line:
    distance = np.cumsum(np.sqrt(np.sum(np.diff(input_array, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]

    # for s in range(25,50,5):
    # Build a list of the spline function, one for each dimension:
    k = 1
    s = 50
    splines = [UnivariateSpline(distance, coords, k=k, s=s) for coords in input_array.T]

    # Computed the spline for the asked distances:
    alpha = np.linspace(0, 1, new_len)
    points_fitted = np.vstack(spl(alpha) for spl in splines).T

    if save == True and save_name != "":
        np.save(outdir + save_name, points_fitted)
    elif save_name == "":
        raise Exception("No name given to save variable")

    return points_fitted


def compute_normals_my_func(coord, outfile, k=10, save=True):
    """Computes normal vectors for input coordinates, saves them and returns them."""

    kd_tree = KDTree(coord)
    neighbours = kd_tree.query(coord, k + 1)[1]
    neighbours = coord[neighbours]

    p = Pool()
    normals = p.map(compute_normal, neighbours)
    normals = np.array(normals, dtype=np.float32)

    # if save:
    #    np.save(outfile, normals)
    return normals


def compute_ma_my_func(dict, outfile, denoise_absmin=None, denoise_delta=None, denoise_min=None, detect_planar=None,
                       save=True):
    """computes medial axis balls when coordinates and normal vectors are given, uses ma utility

    dict is datadictionary containing at least ['coords'] and ['normals']
    max_r is calculated based on the largest differences in x- and y-direction, differences in
    diagonals are not accounted for
    """

    radius_factor = 1.5
    x_min = dict['coords'][:, 0].min()
    x_max = dict['coords'][:, 0].max()
    y_min = dict['coords'][:, 1].min()
    y_max = dict['coords'][:, 1].max()
    max_r_x = (x_max - x_min) * radius_factor
    max_r_y = (y_max - y_min) * radius_factor
    max_r = max_r_y if (max_r_y >= max_r_x) else max_r_x

    ma = MASB(dict, max_r=max_r)
    ma.compute_balls()
    dict = ma.D
    if save:
        io_npy.write_npy(outfile, datadict=dict)

    return dict, max_r


def compute_ma_my_func_mp(dict, outfile, denoise=None, denoise_delta=None, detect_planar=None, save=True):
    """computes medial axis balls when coordinates and normal vectors are given, uses ma_mp utility

    dict is datadictionary containing at least ['coords'] and ['normals']
    max_r is calculated based on the largest differences in x- and y-direction, differences in
    diagonals are not accounted for
    """

    radius_factor = 1.5
    x_min = dict['coords'][:, 0].min()
    x_max = dict['coords'][:, 0].max()
    y_min = dict['coords'][:, 1].min()
    y_max = dict['coords'][:, 1].max()
    max_r_x = (x_max - x_min) * radius_factor
    max_r_y = (y_max - y_min) * radius_factor
    max_r = max_r_y if (max_r_y >= max_r_x) else max_r_x
    max_r_y = (y_max - y_min) * 1.5  # largest radius is 50% larger than largest difference in y values

    ma = MASB_mp(dict, max_r=max_r, denoise=denoise, denoise_delta=denoise_delta, detect_planar=detect_planar)
    ma.compute_balls()
    dict = ma.D
    if save:
        io_npy.write_npy(outfile, datadict=dict)
    return dict, max_r
