"""
docstring
"""


import h5py
import lineid_plot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection


def pseudo_voigt(x_data, amplitude, center, sigma, fraction):
    """
    Function that calculates a pseudo-voigt distribution over a given x_data range 
    """
    sigma_g = sigma/(np.sqrt(2*np.log(2)))
    gaussian = ((((1-fraction)*(amplitude))/(sigma_g*np.sqrt(2*np.pi)))
                *np.exp((-((x_data-center)**2))/(2*(sigma_g**2))))
    lorentzian = ((fraction*amplitude)/(np.pi))*(sigma/(((x_data-center)**2)+sigma**2))
    pseudo_voigt = gaussian+lorentzian
    return pseudo_voigt


def plot_component(ax, hdf5_file, key, peak_number, color=None):
    """docstring"""
    # open hdf5 file
    hdf5 = h5py.File(hdf5_file, 'r')
    # extract wavenumber data
    x_data = list(hdf5[key+'/wavenumber'])
    peak_list = list(hdf5[key].keys())[:-3]
    # extract tuple from dataset
    peak_params = list(list(hdf5['{}/{}'.format(key, peak_list[peak_number-1])])[0])
    fraction, sigma, center, amplitude = peak_params[0:4]
    # calculate pseudo voigt distribution from peak_params
    y_data = pseudo_voigt(x_data, amplitude, center, sigma, fraction)
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']
    else:
        color = color
    ax.plot(x_data, y_data, linestyle='--', color=color)
    ax.fill_between(x_data, y_data, alpha=0.3, color=color)
    hdf5.close()


def plot_components(ax, hdf5_file, key, peak_list):
    """docstring"""
    # will turn int or float into list
    if isinstance(peak_list, (int, float)):
        peak_list = [peak_list]
    else:
        pass
    for _, peak_number in enumerate(peak_list):
        plot_component(ax, hdf5_file, key, peak_number)


def plot_fit(hdf5_filename, key, color='blue'):
    """
    This function produces a graph of a spectra contained within an hdf5 file along with labels
    showing the center location of each pseudo-Voigt profile and their corresponding dataset key.

    Args:
        hdf5_filename (str): the filename and location of an existing hdf5 file containing the
                             spectra of interest.
        key (str): the hdf5 key which contains the datasets describing the existing fit and the
                   raw spectra data.

    Returns:
        fig (matplotlib.figure.Figure): Returns the figure so that the plot can be customized
                                        as needed.
        ax (matplotlib.axes._axes.Axes): Returns the figure axes so that the plot can be
                                         customized as needed.
    """
    # handling input errors
    if not isinstance(hdf5_filename, str):
        raise TypeError('Passed value of `hdf5_filename` is not a string! Instead, it is: '
                        + str(type(hdf5_filename)))
    if not hdf5_filename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError('`hdf5_filename` is not type = .hdf5! Instead, it is: '
                        + hdf5_filename.split('/')[-1].split('.')[-1])
    if not isinstance(key, str):
        raise TypeError('Passed value of `key` is not a string! Instead, it is: '
                        + str(type(key)))
    # open .hdf5
    hdf5 = h5py.File(hdf5_filename, 'r')
    # extract spectra data
    x_data = np.asarray(list(hdf5['{}/wavenumber'.format(key)]))
    y_data = np.asarray(list(hdf5['{}/counts'.format(key)]))
    residuals = np.asarray(list(hdf5['{}/residuals'.format(key)]))
    # extract fitted peak center values
    peak_centers = []
    peak_labels = []
    for _, peak in enumerate(list(hdf5[key])[:-3]):
        center = hdf5['{}/{}'.format(key, peak)][0][2]
        peak_centers.append(center)
        peak_labels.append(peak)
    # plot spectra and peak labels
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]},
                               figsize=(15,6))
    # lock the scale so that additional plots do not warp the labels
    ax1.set_autoscale_on(False)
    # plot labels
    ax2.set_xlabel('Wavenumber ($cm^{-1}$)', fontsize=14)
    ax1.set_xlim(min(x_data), max(x_data))
    ax1.set_ylim(0-0.2*max(y_data), 1.1*max(y_data))
    ax1.set_ylabel('Counts', fontsize=14, labelpad=20)
    ax2.set_ylabel('Residuals', fontsize=14, labelpad=12)
    # plot data
    ax1.plot(x_data, y_data, color='red')
    fit = y_data+residuals
    ax1.plot(x_data, fit, color='blue', linewidth=1.5, linestyle='--')
    ax2.plot(x_data, residuals, color='teal')
    lineid_plot.plot_line_ids(x_data, y_data, peak_centers, peak_labels,
                              box_axes_space=0.12, plot_kwargs={'linewidth':0.75}, ax=ax1)
    # set facecolor
    ax1.set_facecolor=(.95, .95, .95)
    ax2.set_facecolor=(.95, .95, .95)
    # scale residuals plot symmetrically about zero
    ylim = max(abs(min(residuals)), abs(max(residuals)))
    ax2.set_ylim(-ylim, ylim)
    # add grid lines to residual plot
    ax2.grid(which='major', axis='y', linestyle='-')
    # force tick labels for top plot
    ax1.tick_params(axis='both', which='both', labelsize=10, labelbottom=True)
    # add title
    plt.title('{} spectra from {}'.format(key, hdf5_filename.split('/')[-1]), fontsize=18, pad=350)
    # add custom legend
    ax1.text(0.2, 0.4, 'Data', color='Red', fontsize=12, transform=plt.gcf().transFigure)
    ax1.text(0.19, 0.4, '(       ,         )', color='black', fontsize=14, transform=plt.gcf().transFigure)
    ax1.text(0.23, 0.4, 'Model', color='blue', fontsize=12, transform=plt.gcf().transFigure)
    hdf5.close()
    return fig, ax1, ax2


def polygon_under_graph(xlist, ylist):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
    """
    return [(xlist[0], 0.), *zip(xlist, ylist), (xlist[-1], 0.)]


def plot_temp(hdf5_filename, temp):
    """
    docstring
    """
    # open hdf5_file
    hdf5 = h5py.File(hdf5_filename, 'r')
    # intialize 3D plot
    fig = plt.figure(figsize=(20,6))
    ax = fig.add_subplot(111, projection='3d')
    # plot raw spectra data from hdf5 file
    for _, time in enumerate(list(hdf5['{}C'.format(temp)].keys())):
        x_data = list(hdf5['{}C/{}/wavenumber'.format(temp, time)])
        y_data = list(hdf5['{}C/{}/counts'.format(temp, time)]) 
        ax.plot(x_data, y_data, zs=int(time[:-1]), zdir='y', c='blue', linewidth=0.75, alpha=0.7)
    # assign orientation and labels
    ax.view_init(30, -100)
    ax.set_xlabel('Wavenumber', fontsize=12, labelpad=20)
    ax.set_ylabel('Residence Time', fontsize=12, labelpad=10)
    ax.set_zlabel('Counts', fontsize=12, labelpad=10)
    ax.set_title('Spectra @ {}C'.format(temp), fontsize=18)
    hdf5.close()
    return fig, ax


def plot_3D_component(ax, hdf5_filename, temp, peak_number):
    """
    docstring
    """
    # open hdf5_file
    hdf5 = h5py.File(hdf5_filename, 'r')
    # plot pseudo-voigt profiles
    for _, time in enumerate(list(hdf5['{}C'.format(temp)].keys())):
        key = '{}C/{}'.format(temp, time)
        # extract wavenumber data
        x_data = list(hdf5[key+'/wavenumber'])
        peak_list = list(hdf5[key].keys())
        peak_name = peak_list[peak_number-1]
        # extract pseudo voigt parameters
        peak_params = list(hdf5['{}/{}'.format(key, peak_name)][0])
        fraction, sigma, center, amplitude = peak_params[0:4]
        # calculate pseudo voigt distribution from peak_params
        y_data = pseudo_voigt(x_data, amplitude, center, sigma, fraction)
        # assign verticies for polygon to represent area under curve
        verts = []
        verts.append(polygon_under_graph(x_data, y_data))
        # plot polygons
        poly = PolyCollection(verts, facecolors='r', alpha=.6)
        ax.add_collection3d(poly, zs=int(time[:-1]), zdir='y')
    # apply title
    if peak_number < 10:
        ax.set_title('Pseudo-Voigt profile for Peak_0{} @ {}C'.format(peak_number, temp),
                     fontsize=18, pad=35)
    else:
        ax.set_title('Pseudo-Voigt profile for Peak_{} @ {}C'.format(peak_number, temp),
                     fontsize=18, pad=35)    
    # close hdf5 file
    hdf5.close()
    return