"""
Module used to unit test the functionality and outputs of the interpolatespectra.py module
"""

import os
import h5py
import numpy as np
from ramandecompy import interpolatespectra
from ramandecompy import dataprep


HDF5_FILENAME = 'ramandecompy/tests/test_files/test_calibration.hdf5'
TARGET_COMPOUND = 'water'
HDF5 = h5py.File(HDF5_FILENAME, 'r')
X_DATA = np.asarray(HDF5['{}/wavenumber'.format(TARGET_COMPOUND)])
Y_DATA = np.asarray(HDF5['{}/counts'.format(TARGET_COMPOUND)])
TUPLE_LIST = list(zip(X_DATA, Y_DATA))

HDF5_INTERPFILENAME = 'ramandecompy/tests/test_files/interpolated_spectra_calibration_file.hdf5'
HDF5_CALFILENAME = 'ramandecompy/tests/test_files/peakidentify_calibration_test.hdf5'
HDF5_2 = h5py.File(HDF5_CALFILENAME, 'r+')


def test_interp_and_norm():
    """
    A function that tests that the interpolatespectra.interp_and_norm function is behaving
    as expected.
    """
    tuple_list = interpolatespectra.interp_and_norm(HDF5_FILENAME, TARGET_COMPOUND)
    assert isinstance(tuple_list, list), '`tuple_list` is not a list'
    assert isinstance(tuple_list[0], tuple), 'first element of `tuple_list` is not a tuple'
    assert isinstance(tuple_list[0][0], np.int64), 'first element of tuple is not a np.int64'
    x_data, y_data = zip(*tuple_list)
    assert max(y_data) <= 1, 'spectra was not normalized correctly'
    assert len(x_data) == len(y_data), 'x and y data lengths do not match'
    try:
        interpolatespectra.interp_and_norm(4.2, TARGET_COMPOUND)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        interpolatespectra.interp_and_norm('hdf5.txt', TARGET_COMPOUND)
    except TypeError:
        print('A .txt was passed to the function, and was handled well with a TypeError.')
    try:
        interpolatespectra.interp_and_norm(HDF5_FILENAME, [TARGET_COMPOUND])
    except TypeError:
        print('A list was passed to the function, and was handled well with a TypeError.')


def test_apply_scaling():
    """
    A function that tests that the interpolatespectra.apply_scaling function is behaving
    as expected.
    """
    # and odd value for j means the compound should be present if i and target_index match
    j = 7
    i = 3
    target_index = 3
    scaled_tuple_list = interpolatespectra.apply_scaling(TUPLE_LIST, j, i, target_index)
    assert len(scaled_tuple_list) == len(TUPLE_LIST), 'scaled data not the same size as input data'
    assert isinstance(scaled_tuple_list, list), '`scaled_tuple_list` is not a list'
    assert isinstance(scaled_tuple_list[0], tuple), """
    first element of `scaled_tuple_list` is not a tuple"""
    try:
        interpolatespectra.apply_scaling(4.2, j, i, target_index)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        interpolatespectra.apply_scaling([1, 2, 3, 4], j, i, target_index)
    except TypeError:
        print("""A list not containing tuples was passed to the function,
         and was handled well with a TypeError.""")
    try:
        interpolatespectra.apply_scaling(TUPLE_LIST, True, i, target_index)
    except TypeError:
        print('A boolean was passed to the function, and was handled well with a TypeError.')
    try:
        interpolatespectra.apply_scaling(TUPLE_LIST, j, 3, target_index)
    except TypeError:
        print('An int was passed to the function, and was handled well with a TypeError.')
    try:
        interpolatespectra.apply_scaling(TUPLE_LIST, j, i, 4.2)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')


def test_generate_spectra_dataset():
    """
    A function that tests that the interpolatespectra.generate_spectra_dataset
    function is behaving as expected.
    """
    spectra_count = 20
    x_data, y_data, label = interpolatespectra.generate_spectra_dataset(HDF5_FILENAME,
                                                                        TARGET_COMPOUND,
                                                                        spectra_count)
    assert len(x_data) == 20, 'incorrect number of spectra generated (x_data)'
    assert len(y_data) == 20, 'incorrect number of spectra generated (y_data)'
    assert len(label) == 20, 'incorrect number of spectra generated (label)'
    try:
        interpolatespectra.generate_spectra_dataset(4.2,
                                                    TARGET_COMPOUND,
                                                    spectra_count)
    except TypeError:
        print('A float was passed to the function and was handled well with a TypeError')
    try:
        interpolatespectra.generate_spectra_dataset('file.txt',
                                                    TARGET_COMPOUND,
                                                    spectra_count)
    except TypeError:
        print('A .txt was passed to the function and was handled well with a TypeError')
    try:
        interpolatespectra.generate_spectra_dataset(HDF5_FILENAME,
                                                    7,
                                                    spectra_count)
    except TypeError:
        print('An int was passed to the function and was handled well with a TypeError')
    try:
        interpolatespectra.generate_spectra_dataset(HDF5_FILENAME,
                                                    TARGET_COMPOUND,
                                                    -1)
    except ValueError:
        print('A negative int was passed to the function and was handled well with a TypeError')
    try:
        interpolatespectra.generate_spectra_dataset(HDF5_FILENAME,
                                                    TARGET_COMPOUND,
                                                    [1, 2, 3])
    except TypeError:
        print('A list was passed to the function and was handled well with a TypeError')


def test_keyfinder():
    """
    This function tests the operation of the keyfinder
    function in interpolatespectra.py
    """
    # run function
    interpolatespectra.keyfinder('ramandecompy/tests/test_files/test_experiment.hdf5')
    exp_file = h5py.File('ramandecompy/tests/test_files/test_experiment.hdf5', 'r')
    # make assertions
    assert len(exp_file) == 1, 'incorrect number of 1st order groups'
    assert list(exp_file.keys())[0] == '300C', '1st order group name incorrect'
    assert len(exp_file['300C']) == 5, 'incorrect number of 2nd order groups'
    assert list(exp_file['300C'].keys())[0] == '25s', '2nd order group name incorrect'
    assert '300C/25s/wavenumber' in exp_file, 'x data (wavenumber) not stored correctly'
    assert '300C/25s/counts' in exp_file, 'y data (counts) not stored correctly'
    assert len(exp_file['300C/25s']) == 21, 'incorrect number of peaks + raw_data stored'
    # test inputs
    try:
        interpolatespectra.keyfinder(4.2)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.keyfinder('test.txt')
    except TypeError:
        print('A .txt was passed to the function, and it was handled well with a TypeError.')


def test_interpolatedfit():
    """
    This function tests the operation of the interpolatedfit
    function in interpolatespectra.py
    """
    dataprep.new_hdf5('ramandecompy/tests/test_files/interpolated_spectra_calibration_file')
    spectra_count = 2
    # get list of compounds from hdf5 file
    y_data_list = []
    x_data_list = []
    hdf5 = h5py.File(HDF5_CALFILENAME, 'r')
    compound_list = list(HDF5_2.keys())
    hdf5.close()
    key = 'water'
    for _, target_compound in enumerate(compound_list):
        x_data, y_data, labels = interpolatespectra.generate_spectra_dataset(HDF5_CALFILENAME,
                                                                             target_compound,
                                                                             spectra_count)
        y_data_list.append(y_data)
        x_data_list.append(x_data)
    # example i and label value
    i = 1
    label = 1
    # Run function
    for j, label_test in enumerate(labels):
        interpolatespectra.interpolatedfit(HDF5_INTERPFILENAME, key, x_data, y_data, j)
    # test inputs
    try:
        interpolatespectra.interpolatedfit(4.2, key, x_data, y_data, i)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.interpolatedfit('test.txt', key, x_data, y_data, i)
    except TypeError:
        print('A .txt was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.interpolatedfit(HDF5_INTERPFILENAME, 4.2, x_data, y_data, i)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.interpolatedfit(HDF5_INTERPFILENAME,
                                           'test.txt', x_data, y_data, i)
    except TypeError:
        print('A .txt was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.interpolatedfit(HDF5_INTERPFILENAME, key, 4.2, y_data, i)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.interpolatedfit(HDF5_INTERPFILENAME, key, 'x_data', y_data, i)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.interpolatedfit(HDF5_INTERPFILENAME, key, x_data, 4.2, i)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.interpolatedfit(HDF5_INTERPFILENAME, key, x_data, 'y_data', i)
    except TypeError:
        print('A .txt was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.interpolatedfit(HDF5_INTERPFILENAME,
                                           key, x_data, y_data, 4.2)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.interpolatedfit(HDF5_INTERPFILENAME, key, x_data, y_data,'num')
    except TypeError:
        print('A .txt was passed to the function, and it was handled well with a TypeError.')
    os.remove('ramandecompy/tests/test_files/interpolated_spectra_calibration_file.hdf5')
