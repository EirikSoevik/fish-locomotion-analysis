import matplotlib.pyplot as plt
import numpy as np

def set_plot_position(x=3100,y=100,dx=2000,dy=900):
    """Sets plot position the desired position"""
    # set plot position
    # default is on the 3rd right screen, experiment with the values to get what you want, or comment out
    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner for example:
    mngr.window.setGeometry(x,y,dx,dy)

def normal_quiver(coords,normals,k,print_coords=False):
    plt.quiver(coords[:,0],coords[:,1],normals[:,0],normals[:,1])
    if print_coords: plt.scatter(coords[:,0],coords[:,1])
    plt.title("Normals printed on top of coordinates, N=" + str(len(normals)) + " k="+str(k))
    plt.axis("square")
    set_plot_position()
    plt.show()


def coords(coords):
    plt.scatter(coords[:,0],coords[:,1])
    plt.title("Coordinates, N=" + str(len(coords)))
    plt.axis("square")
    set_plot_position()
    plt.show()


def internals(datadict):
    plt.scatter(datadict['ma_coords_in'][:,0],datadict['ma_coords_in'][:,1])
    plt.title("Internal points")
    plt.axis("square")
    set_plot_position()
    plt.show()


def all_in_one(datadict):
    plt.scatter(datadict['coords'][:, 0], datadict['coords'][:, 1], label='coords')
    plt.scatter(datadict['ma_coords_in'][:, 0], datadict['ma_coords_in'][:, 1],label='internal points')
    plt.quiver(datadict['coords'][:,0],datadict['coords'][:,1],datadict['normals'][:,0],datadict['normals'][:,1], label='normal vectors')
    plt.scatter(datadict['ma_coords_out'][:,0],datadict['ma_coords_out'][:,1],label='external points')
    plt.title('Coords, normals, internal and external points')
    plt.legend()
    set_plot_position()
    plt.show()


def externals(datadict):
    plt.scatter(datadict['ma_coords_out'][:,0],datadict['ma_coords_out'][:,1])
    plt.title('External points')
    set_plot_position()
    plt.show()


def linearly(datadict, key):
    plt.plot(datadict[key], label=key)
    plt.title(key)
    set_plot_position()
    plt.show()


def all_in_four(datadict,D,k,N_ma, N_normals,centroid):

    fig, axs = plt.subplots(2,2)
    axs[0,0].scatter(datadict['coords'][:,0], datadict['coords'][:,1], label='coords')
    axs[0,0].scatter(centroid[0],centroid[1], label='centroid')
    axs[0,0].quiver(datadict['coords'][:, 0], datadict['coords'][:, 1], datadict['normals'][:, 0], datadict['normals'][:, 1], pivot='tip', label='normal vectors')
    axs[0,0].set_title('Coordinates, length='+str(len(datadict['coords']))+', N='+str(N_ma))
    axs[0,0].legend()

    axs[0,1].scatter(centroid[0], centroid[1], label='centroid')
    axs[0,1].quiver(D['coords_normals'][:,0],D['coords_normals'][:,1],D['normals_normals'][:,0],D['normals_normals'][:,1], pivot='tip', label='normal vectors')
    axs[0,1].scatter(D['coords_normals'][:,0], D['coords_normals'][:,1])
    axs[0,1].set_title('Normal vectors, k=' + str(k) + ' n = ' + str(N_normals))

    axs[1,0].scatter(datadict['ma_coords_in'][:,0], datadict['ma_coords_in'][:,1], label='internal coords')
    axs[1,0].scatter(datadict['ma_coords_out'][:, 0], datadict['ma_coords_out'][:, 1], label='external coords')
    axs[1,0].set_title('Internal coordinates')
    axs[1,0].legend()

    axs[1,1].scatter(centroid[0], centroid[1], label='centroid')
    axs[1,1].scatter(datadict['coords'][:, 0], datadict['coords'][:, 1], label='coords')
    axs[1,1].scatter(datadict['ma_coords_in'][:, 0], datadict['ma_coords_in'][:, 1], label='internal points')
    axs[1,1].quiver(datadict['coords'][:, 0], datadict['coords'][:, 1], datadict['normals'][:, 0], datadict['normals'][:, 1],  pivot='tip', label='normal vectors')
    axs[1,1].scatter(datadict['ma_coords_out'][:, 0], datadict['ma_coords_out'][:, 1], label='external points')
    axs[1,1].legend()
    axs[1,1].set_title('All together')
    set_plot_position()
    fig.tight_layout()
    plt.show()

def all_in_four_2(datadict,neighbours,N_ma, N_normals,centroid, original_coords, original_normals):

    fig, axs = plt.subplots(2,2)
    plt.tight_layout()
    axs[0,0].scatter(datadict['coords'][:,0], datadict['coords'][:,1], label='coords')
    axs[0,0].scatter(centroid[0],centroid[1], label='centroid')
    axs[0,0].quiver(datadict['coords'][:, 0], datadict['coords'][:, 1], datadict['normals'][:, 0], datadict['normals'][:, 1], pivot='tip', label='normal vectors')
    axs[0,0].set_title('Coordinates, length='+str(len(datadict['coords']))+', N='+str(N_ma))
    axs[0,0].set_aspect('equal',adjustable='datalim')
    axs[0,0].legend()

    axs[0,1].scatter(centroid[0], centroid[1], label='centroid')
    axs[0,1].quiver(original_coords[:,0],original_coords[:,1],original_normals[:,0],original_normals[:,1], pivot='tip', label='normal vectors')
    axs[0,1].scatter(original_coords[:,0],original_coords[:,1],original_normals[:,0])
    axs[0,1].set_title('Normal vectors, k=' + str(neighbours) + ' n = ' + str(N_normals))
    axs[0,1].set_aspect('equal', adjustable='datalim')

    axs[1,0].scatter(datadict['ma_coords_in'][:,0], datadict['ma_coords_in'][:,1], label='internal coords')
    axs[1,0].scatter(datadict['ma_coords_out'][:, 0], datadict['ma_coords_out'][:, 1], label='external coords')
    axs[1,0].set_title('Internal coordinates')
    axs[1,0].legend()
    axs[1,0].set_aspect('equal', adjustable='datalim')

    axs[1,1].scatter(centroid[0], centroid[1], label='centroid')
    axs[1,1].scatter(datadict['coords'][:, 0], datadict['coords'][:, 1], label='coords')
    axs[1,1].scatter(datadict['ma_coords_in'][:, 0], datadict['ma_coords_in'][:, 1], label='internal points')
    axs[1,1].quiver(datadict['coords'][:, 0], datadict['coords'][:, 1], datadict['normals'][:, 0], datadict['normals'][:, 1],  pivot='tip', label='normal vectors')
    #axs[1,1].scatter(datadict['ma_coords_out'][:, 0], datadict['ma_coords_out'][:, 1], label='external points')
    axs[1,1].legend()
    axs[1,1].set_title('All together')
    axs[1,1].set_aspect('equal', adjustable='datalim')
    set_plot_position()

    plt.show()

def midlines(midlines_unfiltered, midlines, midlines_interpolated, coords):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(midlines_unfiltered[:, 0], midlines_unfiltered[:, 1])
    plt.scatter(midlines[:, 0], midlines[:, 1])
    plt.scatter(coords[:, 0], coords[:, 1])
    plt.plot(midlines_interpolated[:,0],midlines_interpolated[:,1])
    plt.title("All midlines with any point outside body boundary removed")
    ax.set_aspect('equal', adjustable='datalim')
    plt.show()

def rib_approx(coords,rib_midline,midline_angles,midline_angles_change):
    fig, ax= plt.subplots(2,1)

    ax[0].plot(rib_midline[:, 0], rib_midline[:, 1], '-o')
    ax[0].scatter(coords[:, 0], coords[:, 1])
    #plt.plot(midlines_interpolated[:, 0], midlines_interpolated[:, 1])
    plt.title("Rib midlines")

    ax[1].plot(midline_angles)
    ax[1].plot(midline_angles_change)

    ax[0].set_aspect('equal', adjustable='datalim')
    set_plot_position()
    plt.show()

def spline_plot(my_spline,coords, spline_x):
    fig, ax = plt.subplots(1)

    ax.plot(spline_x, my_spline(spline_x), '-o')
    ax.scatter(coords[:, 0], coords[:, 1])
    #plt.plot(midlines_interpolated[:, 0], midlines_interpolated[:, 1])
    plt.title("Rib midlines")


    ax.set_aspect('equal', adjustable='datalim')
    set_plot_position()
    plt.show()


def print_dictionary_info(dict,max_r):
    #print("Max radius = " + '{%4.2f}'.format(max_r))
    print('Max radius = %.1f' %max_r)
    print("Length of coords vector: " + str(len(dict['coords'])))
    print("Length of normals vector: " + str(len(dict['normals'])))
    print("Length of coords_normals vector: " + str(len(dict['coords_normals'])))
    print("Length of normals_normals vector: " + str(len(dict['normals_normals'])))
    print("Length of internal points: " + str(len(dict['ma_coords_in'])))
    print("Length of external points: " + str(len(dict['ma_coords_out'])))
    #print("Internals:   Externals: ")
    #print(D['ma_coords_out'])
    #for i, o in zip(D['ma_coords_in'],D['ma_coords_out']):
    #    print('{:=3.1f}'.format(i[0]), '{:=3.1f}'.format(i[1]),'{:3.1f}'.format(o[0]),'{:3.1f}'.format(o[1]))
    nan_count = 0
    for i in dict['ma_coords_out']:
        if np.isnan(i[0]):
            nan_count += 1
        if np.isnan(i[1]):
            nan_count += 1
    print("NaN percentage in external points: " + str((nan_count/(len(dict['ma_coords_out']*2)))))


