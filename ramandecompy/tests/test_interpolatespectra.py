"""
Module used to unit test the functionality and outputs of the interpolatespectra.py module
"""
import math
import h5py
import os
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
from ramandecompy import interpolatespectra

# dataprep.new_hdf5('ramandecompy/tests/test_files/interpolated_spectra_calibration_file')
hdf5_interpfilename = 'ramandecompy/tests/test_files/interpolated_spectra_calibration_file.hdf5'
interphdf5 = h5py.File(hdf5_interpfilename, 'r+')
hdf5_calfilename = 'ramandecompy/tests/test_files/peakidentify_calibration_test.hdf5'
hdf5 = h5py.File(hdf5_calfilename, 'r+')
def test_keyfinder():
    """
    This function tests the operation of the keyfinder
    function in interpolatespectra.py
    """
    # run function
    interpolatespectra.keyfinder('ramandecompy/tests/test_files/dataimport_experiment_files.hdf5')
    exp_file = h5py.File('ramandecompy/tests/test_files/dataimport_experiment_files.hdf5', 'r')
    # make assertions
    assert len(exp_file) == 10, 'incorrect number of 1st order groups'
    assert list(exp_file.keys())[0] == '300C', '1st order group name incorrect'
    assert len(exp_file['300C']) ==5, 'incorrect number of 2nd order groups'
    assert list(exp_file['300C'].keys())[0] == '25s', '2nd order group name incorrect'
    assert '300C/25s/wavenumber' in exp_file, 'x data (wavenumber) not stored correctly'
    assert '300C/25s/counts' in exp_file, 'y data (counts) not stored correctly'
    assert len(exp_file['300C/25s']) == 19, 'incorrect number of peaks + raw_data stored'
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
    # first a function that will return a normalized interpolated spectra
    hdf5_interpfilename = 'ramandecompy/tests/test_files/interpolated_spectra_calibration_file.hdf5'
    hdf5_calfilename = 'ramandecompy/tests/test_files/peakidentify_calibration_test.hdf5'
    spectra_count = 1
    hdf5 = h5py.File(hdf5_calfilename, 'r+')
    interphdf5 = h5py.File(hdf5_interpfilename, 'r+')
    # get list of compounds from hdf5 file
    y_data_list = []
    x_data_list = []
    compound_list = list(hdf5.keys())
    key = 'water'
    for _, target_compound in enumerate(compound_list):
        x_data, y_data, labels = interpolatespectra.generate_spectra_dataset(hdf5_calfilename, target_compound, spectra_count)
        y_data_list.append(y_data)
        x_data_list.append(x_data)
    # Run function
    for i, label in enumerate(labels):
        interpolatespectra.interpolatedfit(hdf5_interpfilename, key, x_data, y_data, label, i)
    # test inputs
    try:
        interpolatespectra.interpolatedfit(4.2, key, x_data, y_data, label, i)
    except TypeError:
        print('A float was passed to the function as `hdf5_filename`, and it was handled well with a TypeError.')
    try:
        interpolatespectra.interpolatedfit('test.txt', key, x_data, y_data, label, i)
    except TypeError:
        print('A .txt was passed to the function as `hdf5_filename`, and it was handled well with a TypeError.')
    try:
        interpolatespectra.interpolatedfit(hdf5_filename, 4.2, x_data, y_data, label, i)
    except TypeError:
        print('A float was passed to the function as `key`, and it was handled well with a TypeError.')
    try:
        interpolatespectra.interpolatedfit(hdf5_filename, 'test.txt', x_data, y_data, label, i)
    except TypeError:
        print('A .txt was passed to the function as `key`, and it was handled well with a TypeError.')
    try:
        interpolatespectra.interpolatedfit(hdf5_filename, key, 4.2, y_data, label, i)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.interpolatedfit(hdf5_filename, key, 'x_data', y_data, label, i)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.interpolatedfit(hdf5_filename, key, x_data, 4.2, label, i)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.interpolatedfit(hdf5_filename, key, x_data, 'y_data', label, i)
    except TypeError:
        print('A .txt was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.interpolatedfit(hdf5_filename, key, x_data, y_data, 4.2, len(labels))
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.interpolatedfit(hdf5_filename, key, x_data, y_data, labels, 'num')
    except TypeError:
        print('A .txt was passed to the function, and it was handled well with a TypeError.')

    # make assertions

def test_combined_interpolatedfit():
    """
    This function tests the operation of the combined_interpolatedfit
    function in interpolatespectra.py
    """
    # first a function that will return a normalized interpolated spectra
    hdf5_interpfilename = 'ramandecompy/tests/test_files/interpolated_spectra_calibration_file.hdf5'
    hdf5_calfilename = 'ramandecompy/tests/test_files/peakidentify_calibration_test.hdf5'
    spectra_count = 1
    # run function
    interpolatespectra.combined_interpolatedfit(hdf5_interpfilename, hdf5_calfilename, spectra_count)
    # test inputs
    try:
        interpolatespectra.combined_interpolatedfit(4.2, hdf5_calfilename, spectra_count)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.combined_interpolatedfit('test.txt', hdf5_calfilename, spectra_count)
    except TypeError:
        print('A .txt was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.combined_interpolatedfit(hdf5_interpfilename, 4.2, spectra_count)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.combined_interpolatedfit(hdf5_interpfilename, 'test.txt', spectra_count)
    except TypeError:
        print('A .txt was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.combined_interpolatedfit(hdf5_interpfilename, hdf5_calfilename, 4.2)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        interpolatespectra.combined_interpolatedfit(hdf5_interpfilename, hdf5_calfilename, 'test.txt')
    except TypeError:
        print('A .txt was passed to the function, and it was handled well with a TypeError.')
    
    # make assertions
    
interphdf5.close()
hdf5.close()
# os.remove('ramandecompy/tests/test_files/interpolated_spectra_calibration_file.hdf5')