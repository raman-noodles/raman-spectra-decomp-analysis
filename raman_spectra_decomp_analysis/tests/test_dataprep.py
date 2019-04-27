"""docstring"""
import os
import h5py
from ramannoodles import dataprep


def test_new_hdf5():
    """docstring"""
    dataprep.new_hdf5('function_test')
    hdf5 = h5py.File('function_test.hdf5', 'r')
    assert isinstance(hdf5, h5py._hl.files.File), 'output type is not a h5py._hl.files.File'
    assert len(hdf5) == 0, 'output h5py._hl.files.File is not empty'
    # test inputs
    try:
        dataprep.new_hdf5(4.2)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    os.remove('function_test.hdf5')
        
        
def test_add_calibration():
    """docstring"""
    dataprep.new_hdf5('test')
    dataprep.add_calibration('test.hdf5',
                             'ramannoodles/tests/test_files/Methane_Baseline_Calibration.xlsx',
                             label='Methane')
    cal_file = h5py.File('test.hdf5', 'r')
    assert list(cal_file.keys())[0] == 'Methane', 'custom label not applied correctly'
    assert len(cal_file) == 1, 'more than one first order group assigned to test.hdf5'
    assert len(cal_file['Methane']) == 3, 'more then 1 peak was stored'
    assert 'Methane/wavenumber' in cal_file, 'x data (wavenumber) not stored correctly'
    assert 'Methane/counts' in cal_file, 'y data (counts) not stored correctly'
    # test that function assigns filename correctly as compound label
    dataprep.new_hdf5('test1')
    dataprep.add_calibration('test1.hdf5',
                             'ramannoodles/tests/test_files/Methane_Baseline_Calibration.xlsx')
    cal_file1 = h5py.File('test1.hdf5', 'r')
    assert list(cal_file1.keys())[0] == 'Methane_Baseline_Calibration', 'filename label not applied correctly'
    # test inputs
    try:
        dataprep.add_calibration(4.2, 'ramannoodles/tests/test_files/CarbonMonoxide_Baseline_Calibration.xlsx')
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.add_calibration('test.hdp5', 4.2)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.add_calibration('test.txt', 'ramannoodles/tests/test_files/CarbonMonoxide_Baseline_Calibration')
    except TypeError:
        print('A .txt file was passed to the function, and it was handled will with a TypeError.')
    os.remove('test.hdf5')
    os.remove('test1.hdf5')
    
    
def test_add_experiment():
    """docstring"""
    dataprep.new_hdf5('exp_test')
    dataprep.add_experiment('exp_test.hdf5', 'ramannoodles/tests/test_files/FA_3.6wt__300C_25s.csv')
    exp_file = h5py.File('exp_test.hdf5', 'r')
    # test generated file
    assert len(exp_file) == 1, 'incorrect number of 1st order groups'
    assert list(exp_file.keys())[0] == '300C', '1st order group name incorrect'
    assert len(exp_file['300C']) ==1, 'incorrect number of 2nd order groups'
    assert list(exp_file['300C'].keys())[0] == '25s', '2nd order group name incorrect'
    assert '300C/25s/wavenumber' in exp_file, 'x data (wavenumber) not stored correctly'
    assert '300C/25s/counts' in exp_file, 'y data (counts) not stored correctly'
    assert len(exp_file['300C/25s']) == 18, 'incorrect number of peaks + raw_data stored'
    # test inputs
    try:
        dataprep.add_experiment(4.2, 'ramannoodles/tests/test_files/CarbonMonoxide_Baseline_Calibration.xlsx')
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.add_experiment('test.hdp5', 4.2)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.add_experiment('test.txt', 'ramannoodles/tests/test_files/CarbonMonoxide_Baseline_Calibration')
    except TypeError:
        print('A .txt file was passed to the function, and it was handled will with a TypeError.')
    os.remove('exp_test.hdf5')


def test_view_hdf5():
    """docstring"""
    # test inputs
    try:
        dataprep.view_hdf5(4.2)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.view_hdf5('test.txt')
    except TypeError:
        print('A .txt was passed to the function, and it was handled well with a TypeError.')


def test_plot_fit():
    """docstring"""
    hdf5_filename = 'ramannoodles/tests/test_files/dataprep_experiment.hdf5'
    key = '300C/25s'
    hdf5 = h5py.File(hdf5_filename, 'r')
    dataprep.plot_fit(hdf5_filename, key)
    #test inputs
    try:
        dataprep.plot_fit(4.2, '300C/25s')
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.plot_fit('test.txt', '300C/25s')
    except TypeError:
        print('A .txt was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.plot_fit('test.txt', 4.2)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    assert key in hdf5, 'input key does not exist within the specified .hdf5 file'
