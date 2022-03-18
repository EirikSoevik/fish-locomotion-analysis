import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib.animation as FuncAnimation

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
    plt.title("Midline spline")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")

    for i in range(1,rows):

        line1.set_xdata(midlines_x[i,:])
        line1.set_ydata(midlines_y[i,:])
        line2.set_xdata(centroid[i,0])
        line2.set_ydata(centroid[i,1])
        line3.set_xdata(coords_x[i,:])
        line3.set_ydata(coords_y[i,:])
        line4.set_ydata(my_splines[i](spline_x))
        line4.set_xdata(spline_x)
        figure.canvas.draw()

        figure.canvas.flush_events()

        time.sleep(0.1)

    figure.canvas.flush_events()