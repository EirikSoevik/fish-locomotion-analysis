import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib.animation as FuncAnimation

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
    plt.title("Midline")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    set_plot_position()

    for i in range(1,rows):
        line1.set_xdata(x[i,:])
        line1.set_ydata(y[i,:])

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

def all_splines_plot(spline_x, my_splines):

    plt.figure()

    for i in range(len(my_splines)):
        #plt.plot(spline_x,my_splines[i])
        plt.plot(spline_x, my_splines[i](spline_x))
        plt.title("all splines")
        plt.show()

def fourier_plot(f_x,f_y,N):

    plt.figure()
    #plt.plot(f_x, 2.0 / N * np.abs(f_y[0:N // 2]))
    plt.plot(f_x, f_y)
    plt.xlabel("f_x")
    plt.ylabel("f_y")
    plt.title("f_X/f_y fourier plot")
    set_plot_position()
    plt.show()

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