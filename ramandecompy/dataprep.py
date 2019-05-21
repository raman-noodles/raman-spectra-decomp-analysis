"""
This model contains a set of functions that allow for the creation an management of hdf5
(heirarchical data format 5) files for storing Raman spectroscopy data and values associated
with analyzing that data. This module interacts closely with the spectrafit.py module also
included with this package.

Developed by the Raman-Noodles team (2019 DIRECT Cohort, University of Washington)
"""


import h5py
import lineid_plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from ramandecompy import spectrafit



def new_hdf5(new_filename):
    """
    This function creates a new hdf5 file in the active directory taking as the sole
    argument a string name for the file.

    Args:
        new_filename (str): Filename for the new hdf5 file such that the created empty
                            file will have the name new_filename.hdf5.

    Returns:
        None
    """
    # handling input errors
    if not isinstance(new_filename, str):
        raise TypeError('Passed value of `filename` is not a string! Instead, it is: '
                        + str(type(new_filename)))
    # w- mode will create a file and fail if the file already exists
    hdf5 = h5py.File('{}.hdf5'.format(new_filename), 'w-')
    hdf5.close()


def add_calibration(hdf5_filename, data_filename, label=None):
    """
    This function adds Raman calibration data to an existing hdf5 file. It uses the
    spectrafit.fit_data function to fit the data before saving the fit result and
    the raw data to the hdf5 file.

    Args:
        hdf5_filename (str): the filename and location of an existing hdf5 file to add the
                             calibration data too.
        data_filename (str): the filename and location of raw Raman spectroscopy data in
                             either the form of an .xlsx or a .csv with the wavenumber data
                             in the 1st column and the counts data in the 2nd column. These
                             files should contain only the wavenumber and counts data
                             (no column labels).
        label (str): (optional) The function will by default use the data_filename (filename
                     only, not the location) as the label for the data stored in the hdf5.
                     This optional argument allows a custom label in the form of a string to
                     be used instead.

    Returns:
        None
    """
    # handling input errors
    if not isinstance(hdf5_filename, str):
        raise TypeError('Passed value of `cal_filename` is not a string! Instead, it is: '
                        + str(type(hdf5_filename)))
    if not hdf5_filename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError('`cal_filename` is not type = .hdf5! Instead, it is: '
                        + hdf5_filename.split('/')[-1].split('.')[-1])
    if not isinstance(data_filename, str):
        raise TypeError('Passed value of `data_filename` is not a string! Instead, it is: '
                        + str(type(data_filename)))
    # r+ is read/write mode and will fail if the file does not exist
    cal_file = h5py.File(hdf5_filename, 'r+')
    if data_filename.split('.')[-1] == 'xlsx':
        data = pd.read_excel(data_filename, header=None, names=('wavenumber', 'counts'))
    elif data_filename.split('.')[-1] == 'csv':
        data = pd.read_csv(data_filename, header=None, names=('wavenumber', 'counts'))
    else:
        print('data file type not recognized')
    # ensure that the data is listed from smallest wavenumber first
    if data['wavenumber'][:1].values > data['wavenumber'][-1:].values:
        data = data.iloc[::-1]
        data.reset_index(inplace=True, drop=True)
    else:
        pass
    # peak detection and data fitting
    fit_result = spectrafit.fit_data(data['wavenumber'].values, data['counts'].values)
    # write data to .hdf5 using custom label if provided
    if label is not None:
        cal_file['{}/wavenumber'.format(label)] = data['wavenumber']
        cal_file['{}/counts'.format(label)] = data['counts']
        for i, _ in enumerate(fit_result):
            if i < 9:
                cal_file['{}/Peak_0{}'.format(label, i+1)] = fit_result[i]
            else:
                cal_file['{}/Peak_{}'.format(label, i+1)] = fit_result[i]
    else:
        label = (data_filename.split('/')[-1]).split('.')[0]
        cal_file['{}/wavenumber'.format(label)] = data['wavenumber']
        cal_file['{}/counts'.format(label)] = data['counts']
        for i, _ in enumerate(fit_result):
            cal_file['{}/Peak_{}'.format(label, i+1)] = fit_result[i]
    cal_file.close()


def add_experiment(hdf5_filename, exp_filename):
    """
    This function adds Raman experimental data to an existing hdf5 file. It uses the
    spectrafit.fit_data function to fit the data before saving the fit result and
    the raw data to the hdf5 file. The data_filename must be in a standardized format
    to interact properly with this function. It must take the form anyname_temp_time.xlsx
    (or .csv) since this function will parse the the temp and time from the filename to
    label the data and fit result in the hdf5 file.

    Args:
        hdf5_filename (str): the filename and location of an existing hdf5 file to add the
                             experiment data too.
        data_filename (str): the filename and location of raw Raman spectroscopy data in
                             either the form of an .xlsx or a .csv with the wavenumber data
                             in the 1st column and the counts data in the 2nd column. These
                             files should contain only the wavenumber and counts data
                             (no column labels).

    Returns:
        None
    """
    # handling input errors
    if not isinstance(hdf5_filename, str):
        raise TypeError('Passed value of `hdf5_filename` is not a string! Instead, it is: '
                        + str(type(hdf5_filename)))
    if not hdf5_filename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError('`hdf5_filename` is not type = .hdf5! Instead, it is: '
                        + hdf5_filename.split('/')[-1].split('.')[-1])
    if not isinstance(exp_filename, str):
        raise TypeError('Passed value of `data_filename` is not a string! Instead, it is: '
                        + str(type(exp_filename)))
    # confirm exp_filename is correct format (can handle additional decimals in exp_filename
    label = '.'.join(exp_filename.split('/')[-1].split('.')[:-1])
    if len(label.split('_')) < 2:
        raise ValueError("""Passed value of `exp_filename` inapproprate. exp_filename must contain
        at least one '_', preferably of the format somename_temp_time.xlsx (or .csv)""")
    # r+ is read/write mode and will fail if the file does not exist
    exp_file = h5py.File(hdf5_filename, 'r+')
    if exp_filename.split('.')[-1] == 'xlsx':
        data = pd.read_excel(exp_filename, header=None, names=('wavenumber', 'counts'))
    elif exp_filename.split('.')[-1] == 'csv':
        data = pd.read_csv(exp_filename, header=None, names=('wavenumber', 'counts'))
    else:
        print('data file type not recognized')
    # ensure that the data is listed from smallest wavenumber first
    if data['wavenumber'][:1].values > data['wavenumber'][-1:].values:
        data = data.iloc[::-1]
        data.reset_index(inplace=True, drop=True)
    else:
        pass
    # peak detection and data fitting
    fit_result = spectrafit.fit_data(data['wavenumber'].values, data['counts'].values)
    # extract experimental parameters from filename
    specs = exp_filename.split('/')[-1].split('.')[:-1]
    if len(specs) > 1:
        spec = ''
        for _, element in enumerate(specs):
            spec = str(spec+element)
        specs = spec
    specs = specs.split('_')
    time = specs[-1]
    temp = specs[-2]
    # write data to .hdf5
    exp_file['{}/{}/wavenumber'.format(temp, time)] = data['wavenumber']
    exp_file['{}/{}/counts'.format(temp, time)] = data['counts']
    for i, _ in enumerate(fit_result):
        if i < 9:
            exp_file['{}/{}/Peak_0{}'.format(temp, time, i+1)] = fit_result[i]
        else:
            exp_file['{}/{}/Peak_{}'.format(temp, time, i+1)] = fit_result[i]
    exp_file.close()


def adjust_peaks(hdf5_file, key, add_list=None, drop_list=None, plot_fits=False):
    """
    Function that allows the user to manually add or remove peaks from the automatic spectra
    fitting by inputing an add_list and/or a drop_list. The function pulls some data from
    the existing fit and overwrites it with the new results.

    Args:
        hdf5_filename (str): the filename and location of an existing hdf5 file that contains
                             the fit the user wants to adjust.
        key (str): the hdf5 key which contains the datasets describing the existing fit and the
                   raw spectra data.
        add_list (list): (optional) list of wavenumbers values where peaks should be added
                         to the fit.
        drop_list (list): (optional) list of keys (str) corresponding to the datasets of peaks
                          in the existing fit that should be dropped from the new model.
        plot_fits (boolean): (optional) if set equal to True, the fit result will be plotted
                             for visual inspection by the user.

    Returns:
        None
    """
    # handling input errors
    if not isinstance(hdf5_file, str):
        raise TypeError('Passed value of `hdf5_file` is not a string! Instead, it is: '
                        + str(type(hdf5_file)))
    if not hdf5_file.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError('`hdf5_file` is not type = .hdf5! Instead, it is: '
                        + hdf5_file.split('/')[-1].split('.')[-1])
    if not isinstance(key, str):
        raise TypeError('Passed value of `key` is not a string! Instead, it is: '
                        + str(type(key)))
    if add_list is None:
        pass
    else:
        if not isinstance(add_list, list):
            raise TypeError('Passed value of `add_list` is not a list! Instead, it is: '
                            + str(type(add_list)))
    if drop_list is None:
        pass
    else:
        if not isinstance(drop_list, list):
            raise TypeError('Passed value of `drop_list` is not a list! Instead, it is: '
                            + str(type(drop_list)))
    if not isinstance(plot_fit, bool):
        raise TypeError('Passed value of `drop_list` is not a list! Instead, it is: '
                        + str(type(plot_fit)))
    hdf5 = h5py.File(hdf5_file, 'r+')
    # extract raw x-y data
    x_data = np.asarray(hdf5['{}/{}'.format(key, 'wavenumber')])
    y_data = np.asarray(hdf5['{}/{}'.format(key, 'counts')])
    # extract peak center and height locations from hdf5
    peaks = []
    for _, peak in enumerate(list(hdf5[key])[:-2]):
        peaks.append((list(hdf5['{}/{}'.format(key, peak)])[2],
                      list(hdf5['{}/{}'.format(key, peak)])[5]))
    # drop desired tuples from peaks
    if drop_list is not None:
        drop_index = []
        for _, name in enumerate(drop_list):
            drop_index.append(int(name.split('_')[-1])-1)
        for i, index in enumerate(drop_index):
            peaks.pop(index-i)
    else:
        pass
    if add_list is not None:
        # interpolate data
        comp_int = interpolate.interp1d(x_data, y_data, kind='cubic')
        # iterate through add_list
        peaks_add = []
        for _, guess in enumerate(add_list):
            height = comp_int(int(guess))
            peaks_add.append((int(guess), int(height)))
    else:
        peaks_add = []
    # build new model
    fit_result = spectrafit.build_custom_model(x_data, y_data, peaks, peaks_add, plot_fits)
    # delete old fit data
    del hdf5['300C/25s']
    # write data to .hdf5
    hdf5['{}/wavenumber'.format(key)] = x_data
    hdf5['{}/counts'.format(key)] = y_data
    for i, _ in enumerate(fit_result):
        if len(fit_result[i]) == 7:
            if i < 9:
                hdf5['{}/Peak_0{}'.format(key, i+1)] = fit_result[i][:6]
            else:
                hdf5['{}/Peak_{}'.format(key, i+1)] = fit_result[i][:6]
        elif len(fit_result[i]) == 8:
            if i < 9:
                hdf5['{}/Peak_0{}*'.format(key, i+1)] = fit_result[i][:6]
            else:
                hdf5['{}/Peak_{}*'.format(key, i+1)] = fit_result[i][:6]
        else:
            print('fit_result for Peak_{} contains an inappropriate number of values'.format(i))
    hdf5.close()


def view_hdf5(filename):
    """
    This function prints out a display of the contents of any hdf5 file. It prints the filename
    followed by a list of the groups and datasets in a familiar directory/file format. Groups
    (folders appear bold) while datasets (files) appear in a standard font.

    Args:
        filename (str): the filename and location of an existing hdf5 file for inspection.

    Returns:
        None
    """
        # handling input errors
    if not isinstance(filename, str):
        raise TypeError('Passed value of `filename` is not a string! Instead, it is: '
                        + str(type(filename)))
    if not filename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError('`filename` is not type = .hdf5! Instead, it is: '
                        + filename.split('/')[-1].split('.')[-1])
    # pring groups and datasets in first three layers
    print('**** {} ****'.format(filename))
    hdf5 = h5py.File(filename, 'r')
    for _, layer_1 in enumerate(list(hdf5.keys())):
        if isinstance(hdf5[layer_1], h5py.Group):
            print('\033[1m{}\033[0m'.format(layer_1))
            for _, layer_2 in enumerate(list(hdf5[layer_1].keys())):
                if isinstance(hdf5['{}/{}'.format(layer_1, layer_2)], h5py.Group):
                    print('|    \033[1m{}\033[0m'.format(layer_2))
                    for _, layer_3 in enumerate(list(hdf5['{}/{}'.format(layer_1, layer_2)])):
                        if isinstance(hdf5['{}/{}/{}'.format(layer_1, layer_2, layer_3)],
                                      h5py.Group):
                            print('|    |    \033[1m{}\033[0m/...'.format(layer_3))
                        else:
                            print('|    |    {}'.format(layer_3))
                else:
                    print('|    {}'.format(layer_2))
        else:
            print('{}'.format(layer_1))
    hdf5.close()


def plot_fit(hdf5_filename, key, color='blue'):
    """
    This function produces a graph of a spectra contained within an hdf5 file along with labels
    showing the center location of each pseudo-Voigt profile and their corresponding dataset key.

    Args:
        hdf5_filename (str): the filename and location of an existing hdf5 file containing the
                             spectra of interest.
        key (str): the hdf5 key which contains the datasets describing the existing fit and the
                   raw spectra data.
        color (str): (optional) a different color can be used to produce the plot.

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
    x_data = list(hdf5['{}/wavenumber'.format(key)])
    y_data = list(hdf5['{}/counts'.format(key)])
    # extract fitted peak center values
    peak_centers = []
    peak_labels = []
    for _, peak in enumerate(list(hdf5[key])[:-2]):
        peak_centers.append(list(hdf5['{}/{}'.format(key, peak)])[2])
        peak_labels.append(peak)
    # plot spectra and peak labels
    fig, ax = lineid_plot.plot_line_ids(x_data, y_data, peak_centers, peak_labels,
                                        box_axes_space=0.12, plot_kwargs={'linewidth':0.75})
    fig.set_size_inches(15, 5)
    # lock the scale so that additional plots do not warp the labels
    ax.set_autoscale_on(False)
    # reset the data plot color
    plt.gca().get_lines()[0].set_color(color)
    plt.xlabel('wavenumber ($cm^{-1}$)', fontsize=14)
    plt.xlim(min(x_data), max(x_data))
    plt.ylabel('counts', fontsize=14)
    plt.title('{} spectra from {}'.format(key, hdf5_filename.split('/')[-1]), fontsize=18, pad=80)
    hdf5.close()
    return fig, ax
