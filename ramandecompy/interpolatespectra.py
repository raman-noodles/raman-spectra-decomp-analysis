"""This function takes in compounds calibration data files to generate interpolated data
made from randomized combinations of calibration peak data, and, using spectrafit,
identifies peaks found in the fed-in known calibration peak spectra. It returns DataFrames
with peak fitting descriptors to be used in either molar decomposition
or machine learning applications.
 """
import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
import lineid_plot
import pandas as pd
from scipy import interpolate
from ramandecompy import spectrafit
from ramandecompy import peakidentify
from ramandecompy import dataprep
from ramandecompy import datavis
from ramandecompy import dataimport


def keyfinder(hdf5_filename):
    """
    This function prints out a display of the list containing hdf5 key with
    two layers. It is generally used with the large file of all the experimental
    data from the dataimport function

    Args:
        hdf5_filename (str): the filename and location of an existing hdf5 file for inspection.

    Returns:
        keys (list): list containing hdf5 key with two layers.
    """
    if not isinstance(hdf5_filename, str):
        raise TypeError("""Passed value of `hdf5_filename` is not a string!
        Instead, it is: """+ str(type(hdf5_filename)))
    if not hdf5_filename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError("""`hdf5_filename` is not type = .hdf5!
        Instead, it is: """+ hdf5_filename.split('/')[-1].split('.')[-1])
    keys = []
    hdf5 = h5py.File(hdf5_filename, 'r')
    for _, layer_1 in enumerate(list(hdf5.keys())):
        if isinstance(hdf5[layer_1], h5py.Group):
    #         print('\033[1m{}\033[0m'.format(layer_1))
            for _, layer_2 in enumerate(list(hdf5[layer_1].keys())):
                if isinstance(hdf5['{}/{}'.format(layer_1, layer_2)], h5py.Group):
    #                 print('|    \033[1m{}\033[0m'.format(layer_2))
                    keys.append('{}/{}'.format(layer_1, layer_2))
                    for _, layer_3 in enumerate(list(hdf5['{}/{}'.format(layer_1,
                                                                         layer_2)])):
                        if isinstance(hdf5['{}/{}/{}'.format(layer_1, layer_2,
                                                             layer_3)],
                                      h5py.Group):
    #                         print('|    |    \033[1m{}\033[0m/...'.format(layer_3))
                            pass
                        else:
                            pass
    #                         print('|    |    {}'.format(layer_3))
                else:
    #                 print('|    {}'.format(layer_2))
                    keys.append('{}/{}'.format(layer_1, layer_2))
        else:
            pass
    #         print('{}'.format(layer_1))
    hdf5.close()
    return keys
def generate_spectra_dataset(hdf5_filename, target_compound, spectra_count):
    """
    docstring
    """
    hdf5 = h5py.File(hdf5_filename, 'r')
    # get list of compounds from hdf5 file
    compound_list = list(hdf5.keys())
    # create list of interpolated spectra
    interp_list = []
    for compound in compound_list:
        # interpolate
        tuple_list = interp_and_norm(hdf5_filename, compound)
        interp_list.append(tuple_list)
    # identify index of target_compound
    target_index = [i for i, compound in enumerate(compound_list) if target_compound in compound][0]
    # create list of spectra
    x_data = []
    y_data = []
    label = []
    for j in range(spectra_count): 
        # apply scaling to interpolated list
        for i, tuple_list in enumerate(interp_list):
            if i == 0:
                # apply scaling
                scaled_tuple_list = apply_scaling(tuple_list, j, i, target_index)
                summed_tuples = scaled_tuple_list
            else:
                # apply scaling
                scaled_tuple_list = apply_scaling(tuple_list, j, i, target_index)    
                summed_tuples = summed_tuples + scaled_tuple_list
        # sort by wavenumber
        combined = sorted(summed_tuples)
        # add by like
        same_x = {x:0 for x, _ in combined}
        for name, num in combined:
            same_x[name] += num
        sum_combined = list(map(tuple, same_x.items()))
        # unzip
        x_combined, y_combined = zip(*sum_combined)
        # set as arrays
        x_combined = np.asarray(x_combined)
        y_combined = np.asarray(y_combined)
        # plots the spectra, will remove from final rev
        plt.plot(x_combined, y_combined)
        # export data with label (0 = no target, 1 = yes target)
        x_data.append(x_combined)
        y_data.append(y_combined)
        label.append(j % 2)
    hdf5.close()
    return x_data, y_data, label
def interpolatedfit(hdf5_filename, key, x_data, y_data, label, num):
    """
    This function adds interpolated Raman calibration data to an existing hdf5 file. It uses the
    spectrafit.fit_data function to fit the data before saving the fit result and
    the raw data to the hdf5 file. It returns a DataFrame which contains the peak fitted data
    and peak descriptors of the interpolated spectra for each iteration of spectra_counts for
    each target compound in the compound list of the calibration file.

    Args:
        hdf5_filename (str): the filename and location of an existing hdf5 file to add the
                             generated interpolated spectra data to.
        key (string): key within `hdf5_filename` of generated interpolated spectra data file
        x_data (list like): The x-values of the generated interpolated spectra
                            for which the model will be fit.
        y_data (list like): The y-values of the generated interpolated spectra
                            for which the model will be fit.
        label (int): The binary values of the generated interpolated spectra
                            that describe 1 if target compound fitted and 0 if not.
        num (int): iteration of spectra counted.

    Returns:
        df (DataFrame): DataFrame which contains the peak fitted data and peak descriptors
                        of the classified interpolated spectra for each iteration of spectra_counts
                        for each target compound in the compound list of the calibration file. 
    """
    # handling input errors
    if not isinstance(hdf5_filename, str):
        raise TypeError('Passed value of `hdf5_filename` is not a string! Instead, it is: '
                        + str(type(hdf5_filename)))
    if not hdf5_filename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError('`hdf5_filename` is not type = .hdf5! Instead, it is: '
                        + hdf5_filename.split('/')[-1].split('.')[-1])
    if not isinstance(key, str):
        raise TypeError("""Passed value of `key` is not an string!
        Instead, it is: """ + str(type(key)))
    if not isinstance(x_data, (list, np.ndarray)):
        raise TypeError('Passed value of `x_data` is not a list or numpy.ndarray! Instead, it is: '
                        + str(type(x_data)))
    if not isinstance(y_data, (list, np.ndarray)):
        raise TypeError('Passed value of `y_data` is not a list or numpy.ndarray! Instead, it is: '
                        + str(type(y_data)))
    if not isinstance(label, int):
        raise TypeError('Passed value of `label` is not a int! Instead, it is: '
                        + str(type(label)))
    if not isinstance(num, int):
        raise TypeError("""Passed value of `num` is not an int!
        Instead, it is: """ + str(type(num)))
    # r+ is read/write mode and will fail if the file does not exist
    exp_file = h5py.File(hdf5_filename, 'r+')
    # peak detection and data fitting   
    fit_result, residuals = spectrafit.fit_data(x_data[num], y_data[num])
    # write data to .hdf5
    exp_file['{}/{}/wavenumber'.format(key, num)] = x_data[num]
    exp_file['{}/{}/counts'.format(key, num)] = y_data[num]
    exp_file['{}/{}/residuals'.format(key, num)] = residuals
    print('{}/{}/residuals'.format(key, num))
    # extract experimental parameters from filename
    for i, result in enumerate(fit_result):
        # create custom datatype
        my_datatype = np.dtype([('fraction', np.float),
                        ('center', np.float),
                        ('sigma', np.float),
                        ('amplitude', np.float),
                        ('fwhm', np.float),
                        ('height', np.float),
                        ('area under the curve', np.float)])
        if i < 9:
            dataset = exp_file.create_dataset('{}/{}/Peak_0{}'.format(key, num, i+1), (1,), dtype=my_datatype)
        else:
            dataset = exp_file.create_dataset('{}/{}/Peak_{}'.format(key, num, i+1), (1,), dtype=my_datatype)
        # apply data to tuple
        data = tuple(result[:7])
        data_array = np.array(data, dtype=my_datatype)
        # write new values to the blank dataset
        dataset[...] = data_array
    print("""Data from fit with compound pseudo-Voigt model.
          Results saved to {}.""".format(hdf5_filename))
    exp_file.close()
    df = pd.DataFrame(data)
    return df
def interp_and_norm(hdf5_filename, compound):
    """
    docstring
    """
    # open hdf5_file
    hdf5 = h5py.File(hdf5_filename, 'r')
    # interpolate spectra
    x_data = np.asarray(hdf5['{}/wavenumber'.format(compound)])
    y_data = np.asarray(hdf5['{}/counts'.format(compound)])
    interp_spectra = interpolate.interp1d(x_data, y_data, kind='cubic')
    # get integer values across x_range, protecting against edge cases
    x_range = np.arange(int(min(x_data)+1), int(max(x_data)))
    y_interp = interp_spectra(x_range)
    # normalize y_interp values
    y_interp_norm = y_interp/max(y_interp)
    # zip x and y values into tuples
    tuple_list = list(zip(x_range, y_interp_norm))
    # close hdf5 file
    hdf5.close()
    return tuple_list

def apply_scaling(tuple_list, j, i ,target_index):
    """
    docstring
    """
    # unpack tuple_list
    x_data, y_data = zip(*tuple_list)
    y_data = np.asarray(y_data)
    # alternate including target_compound or not
    # if j is odd include target_compound
    if j % 2 == 1:
        if i == target_index:
            y_data_scaled = y_data*np.random.uniform(0.1, 1)
        else:
            y_data_scaled = y_data*np.random.uniform()
    # if j is even no target_compound
    elif j % 2 == 0:
        if i == target_index:
            y_data_scaled = y_data*0
        else:
            y_data_scaled = y_data*np.random.uniform() 
    else:
        pass
    # repack tuple_list
    scaled_tuple_list = list(zip(x_data, y_data_scaled))
    return scaled_tuple_list

def combined_interpolatedfit(hdf5_interpfilename, hdf5_calfilename, spectra_count):
    """
    This function combines the interpolated spectra generated previously and spectrafits them.
    It returns a list of DataFrames which contains the peak fitted data and peak descriptors of the 
    interpolated spectra for each iteration of spectra_counts for each target compound in 
    the compound list of the calibration file. Therefore, this function iterates over of spectra_counts
    for each target compound in the compound list of the calibration file

    Args:
        hdf5_interpfilename (str): the filename and location of an existing hdf5 file to add the
                             interpolated data to.
        hdf5_calfilename (str): the filename and location of an existing hdf5 file with
                             calibration data to compare.
        spectra_count (int): number of spectra to be counted.
                             

    Returns:
        frames (list): list of DataFrames which contain the peak fitted datapeak descriptors
        of the classified interpolated spectra for each iteration of spectra_counts for
        each target compound in the compound list of the calibration file.
    """
    if not isinstance(hdf5_interpfilename, str):
        raise TypeError("""Passed value of `hdf5_interpfilename` is not a string!
        Instead, it is: """+ str(type(knownhdf5_filename)))
    if not hdf5_interpfilename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError("""`hdf5_interpfilename` is not type = .hdf5!
        Instead, it is: """+ hdf5_interpfilename.split('/')[-1].split('.')[-1])
    if not isinstance(hdf5_calfilename, str):
        raise TypeError("""Passed value of `hdf5_calfilename` is not a string!
        Instead, it is: """+ str(type(hdf5_calfilename)))
    if not hdf5_calfilename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError("""`hdf5_calfilename` is not type = .hdf5!
        Instead, it is: """+ hdf5_calfilename.split('/')[-1].split('.')[-1])
    if not isinstance(spectra_count, int):
        raise TypeError("""Passed value of `spectra_count` is not an int!
        Instead, it is: """ + str(type(spectra_count)))
    hdf5 = h5py.File(hdf5_calfilename, 'r+')
    # get list of compounds from hdf5 file
    y_data_list = []
    x_data_list = []
    frames = []
    compound_list = list(hdf5.keys())
    print(compound_list)
    for _, target_compound in enumerate(compound_list):
        # Generate interpolated spectra
        x_data, y_data, labels = generate_spectra_dataset(hdf5_calfilename, target_compound, spectra_count)
        y_data_list.append(y_data)
        x_data_list.append(x_data)
        for i, label in enumerate(labels):
            # Generate peak fitted dataframes and descriptors
            interpdf = interpolatedfit(hdf5_interpfilename, 'interp_'+target_compound, x_data, y_data, label, i) 
            frames.append(interpdf)
    return frames