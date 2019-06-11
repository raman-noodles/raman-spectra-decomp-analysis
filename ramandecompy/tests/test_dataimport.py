"""
Module used to unit test the functionality and outputs of the dataimport.py module
"""
import os
import h5py
from ramandecompy import dataprep
from ramandecompy import dataimport



def test_data_import():
    """
    This function tests the operation of the peak_assignment
    function in peakidentify.py
    """
    dataprep.new_hdf5('exp_test')
    dataprep.add_experiment('exp_test.hdf5', 'ramandecompy/tests/test_files/FA_3.6wt%_300C_25s.csv')
    exp_file = h5py.File('exp_test.hdf5', 'r')
    # test generated file
    assert len(exp_file) == 1, 'incorrect number of 1st order groups'
    assert list(exp_file.keys())[0] == '300C', '1st order group name incorrect'
    assert len(exp_file['300C']) ==1, 'incorrect number of 2nd order groups'
    assert list(exp_file['300C'].keys())[0] == '25s', '2nd order group name incorrect'
    assert '300C/25s/wavenumber' in exp_file, 'x data (wavenumber) not stored correctly'
    assert '300C/25s/counts' in exp_file, 'y data (counts) not stored correctly'
    assert len(exp_file['300C/25s']) == 19, 'incorrect number of peaks + raw_data stored'
    # test inputs
    try:
        dataprep.add_experiment(4.2, 'ramandecompy/tests/test_files/CarbonMonoxide_Baseline_Calibration.xlsx')
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.add_experiment('exp_test.hdp5', 4.2)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.add_experiment('exp_test.txt', 'ramandecompy/tests/test_files/CarbonMonoxide_Baseline_Calibration')
    except TypeError:
        print('A .txt file was passed to the function, and it was handled will with a TypeError.')
    os.remove('exp_test.hdf5')
