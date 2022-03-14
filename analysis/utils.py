import numpy as np
import os
from util.utils import find_centroid

def get_attributes(my_dir):
    """Creates a time-sorted dictionary with all information, which can then be looped to access specifics"""

    dir_files = os.listdir(my_dir)
    i = 0
    dir_time_code = []
    for this_file in dir_files:
        dir_time_code.append([this_file.partition("=")[2], i])
        i += 1

    dir_time_code.sort()
    dict_list = {}

    for i in range(len(dir_time_code)):
        time_code, element = dir_time_code[i]

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

def