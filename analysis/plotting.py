import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib.animation as FuncAnimation


""" Plots available:

    - midline_animation(x,y)
    - midline_animation_centroid(x, y,centroid)
    - body_midline_centroid(coords_x, coords_y, midlines_x, midlines_y, centroid)
    - spline_plotting(my_splines,spline_x,coords_x, coords_y, midlines_x, midlines_y, centroid)
    - all_splines_plot(spline_x, my_splines, my_title, save_dir, save=False)
    - fourier_plot(f_x,f_y,N, my_title, save_dir, save=False)
    - fourier_animation(f_x, f_y,N)
    - fft_plot(fx, fy)
    - point_in_time(my_array, my_title, save_dir, pos=-1,save=False)
    - all_point_in_time(my_array, my_title, save_dir,save=False)
    - midlines_centroid_in_time(midline_x, midline_y,centroid)
    - histogram(l, my_title, save_dir, save=False)
    - simple_plot(vec, my_title, save_dir, save=False)
"""

def set_plot_position(x=3100,y=100,dx=2000,dy=900):
    """Sets plot position the desired position"""
    # set plot position
    # default is on the 3rd right screen, experiment with the values to get what you want, or comment out
    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner for example:
    mngr.window.setGeometry(x,y,dx,dy)

def midline_animation(x, y):

    rows, col = x.shape
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.set_aspect('equal', adjustable='datalim')

    plt.ion()
    figure, ax = plt.subplots(figsize=(10, 8))
    line1, = ax.plot(x[0,:],y[0,:])
    ax.set_aspect('equal',adjustable='datalim')
    plt.title("Midline frame 0 of " + str(rows))
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    set_plot_position()

    for i in range(1,rows):
        line1.set_xdata(x[i,:])
        line1.set_ydata(y[i,:])
        plt.title("Midline frame " + str(i) +" of " + str(rows))
        figure.canvas.draw()

        figure.canvas.flush_events()

        time.sleep(0.1)

def midline_animation_centroid(x, y,centroid):

    rows, col = x.shape
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.set_aspect('equal', adjustable='datalim')

    plt.ion()
    figure, ax = plt.subplots(figsize=(10, 8))
    line1, = ax.plot(x[0,:],y[0,:])
    line2, = ax.plot(centroid[0,0], centroid[0,1],'ro')
    ax.set_aspect('equal',adjustable='datalim')
    plt.title("Midline")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    set_plot_position()

    for i in range(1,rows):

        line1.set_xdata(x[i,:])
        line1.set_ydata(y[i,:])
        line2.set_xdata(centroid[i,0])
        line2.set_ydata(centroid[i,1])

        figure.canvas.draw()

        figure.canvas.flush_events()

        time.sleep(0.1)

def body_midline_centroid(coords_x, coords_y, midlines_x, midlines_y, centroid):


    rows, col = midlines_x.shape
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.set_aspect('equal', adjustable='datalim')

    plt.ion()
    figure, ax = plt.subplots(figsize=(10, 8))
    line1, = ax.plot(midlines_x[0,:],midlines_y[0,:])
    line2, = ax.plot(centroid[0,0], centroid[0,1],'ro')
    line3, = ax.plot(coords_x[0,:], coords_y[0,:])
    ax.set_aspect('equal',adjustable='datalim')
    plt.title("Midline")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    set_plot_position()

    for i in range(1,rows):

        line1.set_xdata(midlines_x[i,:])
        line1.set_ydata(midlines_y[i,:])
        line2.set_xdata(centroid[i,0])
        line2.set_ydata(centroid[i,1])
        line3.set_xdata(coords_x[i,:])
        line3.set_ydata(coords_y[i,:])

        figure.canvas.draw()

        figure.canvas.flush_events()

        time.sleep(0.1)

def spline_plotting(my_splines,spline_x,coords_x, coords_y, midlines_x, midlines_y, centroid):


    rows, col = midlines_x.shape
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.set_aspect('equal', adjustable='datalim')

    plt.ion()
    figure, ax = plt.subplots(figsize=(10, 8))
    line1, = ax.plot(midlines_x[0,:],midlines_y[0,:])
    line2, = ax.plot(centroid[0,0], centroid[0,1],'ro')
    line3, = ax.plot(coords_x[0,:], coords_y[0,:])
    line4, = ax.plot(spline_x, my_splines[0](spline_x))
    ax.set_aspect('equal',adjustable='datalim')
    plt.title("Midline spline, i= " + str(0))
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    set_plot_position()

    for i in range(1,rows):

        line1.set_xdata(midlines_x[i,:])
        line1.set_ydata(midlines_y[i,:])
        line2.set_xdata(centroid[i,0])
        line2.set_ydata(centroid[i,1])
        line3.set_xdata(coords_x[i,:])
        line3.set_ydata(coords_y[i,:])
        line4.set_ydata(my_splines[i](spline_x))
        line4.set_xdata(spline_x)
        plt.title("Midline spline, i= " + str(i))
        figure.canvas.draw()

        figure.canvas.flush_events()

        time.sleep(0.1)

    figure.canvas.flush_events()

def all_splines_plot(spline_x, my_splines, my_title, save_dir, save=False):

    plt.figure()

    for i in range(len(my_splines)):
        #plt.plot(spline_x,my_splines[i])
        plt.plot(spline_x, my_splines[i](spline_x))

    plt.title("all splines")
    plt.show()

    if save:
        plt.savefig(save_dir+my_title+".png")

def fourier_plot(f_x,f_y,N, my_title, save_dir, save=False):

    plt.figure()
    #plt.plot(f_x, 2.0 / N * np.abs(f_y[0:N // 2]))
    plt.plot(f_x, f_y)
    plt.xlabel("f_x")
    plt.ylabel("f_y")
    plt.title("f_X/f_y fourier plot")
    set_plot_position()
    plt.show()
    if save:
        plt.savefig(save_dir+my_title+".png")


def fourier_animation(f_x, f_y,N):

    freq_dim, space_dim = f_y.shape

    plt.ion()
    figure, ax = plt.subplots(figsize=(10, 8))
    line1, = ax.plot(f_x,f_y[:,0])
    #ax.set_aspect('equal',adjustable='datalim')
    plt.xlim([0,max(f_x)*1.1])
    plt.ylim([0, f_y.max()*1.1])
    plt.title("Midline point fourier analysis for index 0")
    plt.xlabel("f [Hz]")
    plt.ylabel("|f_y|")
    set_plot_position()

    for s in range(1,space_dim):
        line1.set_xdata(f_x)
        line1.set_ydata(f_y[:,s])
        figure.canvas.draw()
        plt.title("Midline point fourier analysis for index " + str(s))
        figure.canvas.flush_events()

        time.sleep(1)

    figure.canvas.flush_events()

def fft_plot(fx, fy):

    ftime, N = fy.shape

    plt.ion()
    figure, ax = plt.subplots(figsize=(10, 8))
    #line1, = ax.plot(fx[0,:],fy[0,:])
    line1, = ax.plot(fx[0, :], 2.0 / N * np.abs(fy[0, 0:N//2]))

    #ax.set_aspect('equal',adjustable='datalim')
    plt.title("Fourier analysis of midline for frame: 0" )
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    set_plot_position()


    #2.0 / N * np.abs(f_y[i, 0:N // 2])
    for i in range(1,ftime):
        #line1.set_xdata(fx[i,:])
        #line1.set_ydata(fy[i,:])
        line1.set_xdata(fx[i,:])
        line1.set_ydata(2.0 / N * np.abs(fy[i, 0:N//2]))
        figure.canvas.draw()
        plt.title("Fourier analysis of midline for frame: " + str(i))
        figure.canvas.flush_events()

        time.sleep(0.1)

    figure.canvas.flush_events()

def point_in_time(my_array, my_title, save_dir, pos=-1,save=False):
    """Plot a single midline point in time to evaluate swimming period"""

    plt.figure()
    plt.plot(my_array[:, pos])
    plt.title(my_title + " : position " + str(pos))
    plt.xlabel("frame from start, [s * 1/50]")
    plt.ylabel("Position")
    plt.show()
    if save:
        plt.savefig(save_dir+my_title+".png")

def all_point_in_time(my_array, my_title, save_dir,save=False):
    """Plot a single midline point in time to evaluate swimming period"""

    plt.figure()
    plt.plot(my_array)
    plt.title(my_title)
    plt.xlabel("frame from start, [s * 1/50]")
    plt.ylabel("Position")
    plt.show()
    if save:
        plt.savefig(save_dir+my_title+".png")

def midlines_centroid_in_time(midline_x, midline_y,centroid, my_title, save_dir, save=False):

    fig, axs = plt.subplots(2,2)
    plt.tight_layout()

    axs[0, 0].plot(midline_x)
    axs[0, 0].set_title("midline x")
    axs[0, 0].set(ylabel="x-direction")
    axs[0, 0].set(xlabel="timeframes")

    axs[0, 1].plot(midline_y)
    axs[0, 1].plot(centroid[:,1],'rv')
    axs[0, 1].set_title("midline y")
    axs[0, 1].set(ylabel="y-direction")
    axs[0, 1].set(xlabel="timeframes")

    axs[1, 0].plot(centroid[:,0])
    axs[1, 0].set_title("centroid x")
    axs[1, 0].set(xlabel="timeframes")
    axs[1, 0].set(ylabel="x-direction")

    axs[1, 1].plot(centroid[:,1])
    axs[1, 1].set_title("centroid y")
    axs[1, 1].set(xlabel="timeframes")
    axs[1, 1].set(ylabel="y-direction")

    plt.show()

    if save:
        plt.savefig(save_dir+my_title+".png")


def histogram(l, my_title, save_dir, save=False):
    plt.figure()
    n, bins, patches = plt.hist(l)
    plt.title(my_title)
    set_plot_position()
    plt.show()

    if save:
        plt.savefig(save_dir+my_title+".png")

def simple_plot(x_vec, y_vec, my_title, save_dir, save=False):

    plt.figure()
    plt.plot(x_vec, y_vec)
    plt.title(my_title)
    set_plot_position()
    plt.show()

    if save:
        plt.savefig(save_dir+my_title+".png")

def local_maxima_plot(new_x_vec, y_max_fourier, polynomial, mean_length, my_title, save_dir, save=False):

    y_vec = y_max_fourier/mean_length
    yfit = np.polyval(polynomial, new_x_vec)/mean_length

    plt.figure()
    plt.plot(new_x_vec, y_vec, label='Fourier Amplitudes')
    plt.plot(new_x_vec, yfit, color='blue', label='Best Fit - FFT amplitudes')
    plt.legend()
    plt.title(my_title)
    set_plot_position()
    plt.show()




    if save:
        plt.savefig(save_dir+my_title+".png")
