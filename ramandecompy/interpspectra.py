"""This function takes in compounds from a dictionary from shoyu, and, using spectrafit,
identifies peaks found in both the fed-in known spectra, as well as the unknown spectra
to be analyzed. From that identification, it then classifies the peaks in the unknown
spectra based on the fed-in known spectra.
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
    seconds = []
    hdf5 = h5py.File(hdf5_filename, 'r')
    for _, layer_1 in enumerate(list(hdf5.keys())):
        if isinstance(hdf5[layer_1], h5py.Group):
    #         print('\033[1m{}\033[0m'.format(layer_1))
            for _, layer_2 in enumerate(list(hdf5[layer_1].keys())):
                if isinstance(hdf5['{}/{}'.format(layer_1, layer_2)], h5py.Group):
    #                 print('|    \033[1m{}\033[0m'.format(layer_2))
                    seconds.append('{}/{}'.format(layer_1, layer_2))
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
                    seconds.append('{}/{}'.format(layer_1, layer_2))
        else:
            pass
    #         print('{}'.format(layer_1))
    hdf5.close()
    return seconds
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
def combine_experiment(hdf5_filename, key, x_data, y_data, labels, num):
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
#     if not isinstance(exp_filename, str):
#         raise TypeError('Passed value of `data_filename` is not a string! Instead, it is: '
#                         + str(type(exp_filename)))
#     # confirm exp_filename is correct format (can handle additional decimals in exp_filename
#     label = '.'.join(exp_filename.split('/')[-1].split('.')[:-1])
#     if len(label.split('_')) < 2:
#         raise ValueError("""Passed value of `exp_filename` inapproprate. exp_filename must contain
#         at least one '_', preferably of the format somename_temp_time.xlsx (or .csv)""")
    # r+ is read/write mode and will fail if the file does not exist
    exp_file = h5py.File(hdf5_filename, 'r+')
#     if exp_filename.split('.')[-1] == 'xlsx':
#         data = pd.read_excel(exp_filename, header=None, names=('wavenumber', 'counts'))
#     elif exp_filename.split('.')[-1] == 'csv':
#         data = pd.read_csv(exp_filename, header=None, names=('wavenumber', 'counts'))
#     else:
#         print('data file type not recognized')
    # ensure that the data is listed from smallest wavenumber first
#     if x_data[:1].values > x_data[-1:].values:
#         data = data.iloc[::-1]
#         data.reset_index(inplace=True, drop=True)
#     else:
#         pass
    # peak detection and data fitting
    
    fit_result, residuals = spectrafit.fit_data(x_data[num], y_data[num])
    # write data to .hdf5
    exp_file['{}/{}/wavenumber'.format(key, num)] = x_data[num]
    exp_file['{}/{}/counts'.format(key, num)] = y_data[num]
    exp_file['{}/{}/residuals'.format(key, num)] = residuals
    print('{}/{}/residuals'.format(key, num))
#     fit_result, residuals = spectrafit.fit_data(data['wavenumber'].values, data['counts'].values)
    # extract experimental parameters from filename
#     specs = exp_filename.split('/')[-1].split('.')[:-1]
#     if len(specs) > 1:
#         spec = ''
#         for _, element in enumerate(specs):
#             spec = str(spec+element)
#         specs = spec
#     specs = specs.split('_')
#     time = specs[-1]
#     temp = specs[-2]
    
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

def interpolated_spectra(hdf5_interpfilename, hdf5_calfilename, spectra_count):
    """
    docstring
    """
    hdf5 = h5py.File(hdf5_calfilename, 'r+')
    # get list of compounds from hdf5 file
    y_data_list = []
    x_data_list = []
    frames = []
    compound_list = list(hdf5.keys())
    print(compound_list)
    for _, target_compound in enumerate(compound_list):
        x_data, y_data, labels = generate_spectra_dataset(hdf5_calfilename, target_compound, spectra_count)
        y_data_list.append(y_data)
        x_data_list.append(x_data)
        for i, label in enumerate(labels):
            interpdf = combine_experiment(hdf5_interpfilename, 'interp_'+target_compound, x_data, y_data, label, i) 
            frames.append(interpdf)
    return frames