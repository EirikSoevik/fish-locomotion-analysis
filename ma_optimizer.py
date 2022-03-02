# ma_optimizer is a script for optimizing the parameters of the ma.py/compute_ma.py parameters

import sys, argparse

# Choose one of the two under:
from masbpy.ma_mp import MASB
#from masbpy.ma import MASB

from masbpy import io_ply, io_npy, metacompute
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

MY_FILE = "mask_optimizer_dir/" # fish geometry and normals
#MY_FILE = "simple_example/"


#def my_plotter(max_r, denoise_absmin, denoise_delta,denoise_min, detect_planar):
def my_plotter():

    fig, axs = plt.subplots(2)
    #fig.suptitle((" max_r: " + str(max_r) + " denoise_absmin : " + str(denoise_absmin) + " denoise_delta: " + str(
    #    denoise_delta) + " denoise_min : " + str(denoise_min) + " det_planar: " + str(detect_planar)))
    axs[0].quiver(datadict['coords'][:, 0], datadict['coords'][:, 1], datadict['normals'][:, 0], datadict['normals'][:, 1])
    #axs[1].plot(datadict['ma_coords_in'][:,0],datadict['ma_coords_in'][:,1])
    axs[1].scatter(datadict['coords'][:,0],datadict['coords'][:,1])
    #plt.quiver(datadict['coords'][:, 0], datadict['coords'][:, 1], datadict['normals'][:, 0], datadict['normals'][:, 1])
    #plt.plot(datadict['ma_coords_in'][:,0],datadict['ma_coords_in'][:,1])
    axs[0].set_title("Coordinates and normal vectors")
    axs[1].set_title("Coordinates")
    fig.tight_layout()
    plt.show()

def ma_in_plotter():

    fig, axs = plt.subplots(2)
    #fig.suptitle((" max_r: " + str(max_r) + " denoise_absmin : " + str(denoise_absmin) + " denoise_delta: " + str(
    #    denoise_delta) + " denoise_min : " + str(denoise_min) + " det_planar: " + str(detect_planar)))
    axs[0].quiver(datadict['coords'][:, 0], datadict['coords'][:, 1], datadict['normals'][:, 0], datadict['normals'][:, 1])
    axs[1].scatter(datadict['ma_coords_in'][:,0],datadict['ma_coords_in'][:,1])
    #axs[1].scatter(datadict['coords'][:,0],datadict['coords'][:,1])
    #plt.quiver(datadict['coords'][:, 0], datadict['coords'][:, 1], datadict['normals'][:, 0], datadict['normals'][:, 1])
    #plt.plot(datadict['ma_coords_in'][:,0],datadict['ma_coords_in'][:,1])
    axs[0].set_title("Coordinates and normal vectors")
    axs[1].set_title("Internal coordinates i.e. midline")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":

    # for max_r in range(1, 602, 200):
    #     for denoise_absmin in range(1,6,2):
    #         for denoise_delta in range(1,6,2):
    #             for denoise_min in range(1,6,2):
    #                 for detect_planar in True,False:
    #                     ma = MASB(datadict, max_r=max_r,denoise_absmin=denoise_absmin,denoise_delta=denoise_delta,denoise_min=denoise_min,detect_planar=detect_planar)
    #                     ma.compute_balls()
    #                     #print(" max_r: " + str(max_r) + " denoise_absmin : " + str(denoise_absmin) + " denoise_delta: " + str(denoise_delta) + " denoise_min : " + str(denoise_min) + " det_planar: " + str(detect_planar))
    #                     my_plotter(datadict, max_r, denoise_absmin, denoise_delta, denoise_min, detect_planar)

    print("Processing file " + MY_FILE)

    print("Calculating normals")

    N = 40 # Reduce array to 1/Nth size
    coords = np.load(MY_FILE+"original_coords.npy")
    normals = np.load(MY_FILE+"manually_edited_normals.npy")
    new_len = int(len(coords)/N)
    new_coords = np.zeros([new_len,2])
    new_normals = np.zeros([new_len,2])
    for i in range(len(new_coords)):
        new_coords[i,0] = coords[N*i,0]
        new_coords[i,1] = coords[N*i,1]
        new_normals[i,0] = normals[N*i,0]
        new_normals[i,1] = normals[N*i,1]


    for i in new_normals:
        print(i)


    #new_normals[15] = new_normals[10]
    plt.quiver(new_coords[:,0], new_coords[:,1],new_normals[:,0], new_normals[:,1])
    plt.title("Decimated coordinates by 1/" + str(N))
    plt.show()



    np.save(MY_FILE+"coords.npy",new_coords)
    np.save(MY_FILE+"normals.npy",new_normals)
    # input args after sys.executable: file to run, input file, output dir
    #subprocess.call([sys.executable,'compute_normals.py', MY_FILE + "coords.npy", MY_FILE, '-k=10'])

    datadict = {}
    datadict['coords'] = np.load(MY_FILE + 'coords.npy')
    datadict['normals'] = np.load(MY_FILE + 'normals.npy')

    my_plotter()

    print("Calculating medial axis points")
    # input args after sys.executable: file to run, input file, output dir
    #subprocess.call([sys.executable, 'compute_ma.py',MY_FILE, MY_FILE])
    #ma = MASB(datadict, max_r=args.max_radius, denoise_absmin=None, denoise_delta=None, denoise_min=args.denoise,
    #          detect_planar=args.planar)
    #ma = MASB(datadict,max_r=500,denoise=1,denoise_delta=1,detect_planar=1)
    ma = MASB(datadict,max_r=500)
    ma.compute_balls()


    print("Finished computing, now plotting")

    datadict['ma_coords_in'] = np.load(MY_FILE + 'ma_coords_in.npy')
    ma_in_plotter()
    print("Finished plotting")
