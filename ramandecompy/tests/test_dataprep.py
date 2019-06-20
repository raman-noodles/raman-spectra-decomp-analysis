"""docstring"""
import os
import h5py
from ramandecompy import dataprep


def test_new_hdf5():
    """
    A function that tests that there are no errors in the `new_hdf5` function from dataprep.
    """
    dataprep.new_hdf5('function_test')
    # test inputs
    try:
        dataprep.new_hdf5(4.2)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    os.remove('function_test.hdf5')


def test_add_calibration():
    """
    A function that tests the `add_calibration` function from dataprep. It first tests that no
    errors occur when the function is run before testing the output to ensure that the calibration
    compound was sucessfully added and labeled appropriately. It checks that the proper number of
    peaks were saved as well as the wavenumber, counts, and residuals. It tests both the custom and
    automatic labeling functionality before finally ensuring that input errors are handled well.
    """
    dataprep.new_hdf5('test')
    dataprep.add_calibration('test.hdf5',
                             'ramandecompy/tests/test_files/Methane_Baseline_Calibration.xlsx',
                             label='Methane')
    cal_file = h5py.File('test.hdf5', 'r')
    assert list(cal_file.keys())[0] == 'Methane', 'custom label not applied correctly'
    assert len(cal_file) == 1, 'more than one first order group assigned to test.hdf5'
    assert len(cal_file['Methane']) == 4, 'more then 1 peak was stored'
    assert 'Methane/wavenumber' in cal_file, 'x data (wavenumber) not stored correctly'
    assert 'Methane/counts' in cal_file, 'y data (counts) not stored correctly'
    assert 'Methane/residuals' in cal_file, 'residuals not stored correctly'
    # test that function assigns filename correctly as compound label
    dataprep.new_hdf5('test1')
    dataprep.add_calibration('test1.hdf5',
                             'ramandecompy/tests/test_files/Methane_Baseline_Calibration.xlsx')
    cal_file1 = h5py.File('test1.hdf5', 'r')
    assert list(cal_file1.keys())[0] == 'Methane_Baseline_Calibration', """
    filename label not applied correctly"""
    # test inputs
    try:
        dataprep.add_calibration(4.2, """ramandecompy/tests/test_files/
        CarbonMonoxide_Baseline_Calibration.xlsx""")
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.add_calibration('test.hdp5', 4.2)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.add_calibration('test.txt', """ramandecompy/tests/
        test_files/CarbonMonoxide_Baseline_Calibration""")
    except TypeError:
        print('A .txt file was passed to the function, and it was handled will with a TypeError.')
    os.remove('test.hdf5')
    os.remove('test1.hdf5')


def test_add_experiment():
    """
    A function that tests the `add_experiment` function from dataprep. It first tests that no
    errors occur when the function is run before testing the output to ensure that the
    experimental data was sucessfully added and labeled appropriately. It checks that the
    proper number of peaks were saved as well as the wavenumber, counts, and residuals.
    Lastly it ensures that input errors are handled well.
    """
    dataprep.new_hdf5('exp_test')
    dataprep.add_experiment('exp_test.hdf5',
                            'ramandecompy/tests/test_files/FA_3.6wt%_300C_25s.csv')
    exp_file = h5py.File('exp_test.hdf5', 'r')
    # test generated file
    assert len(exp_file) == 1, 'incorrect number of 1st order groups'
    assert list(exp_file.keys())[0] == '300C', '1st order group name incorrect'
    assert len(exp_file['300C']) == 1, 'incorrect number of 2nd order groups'
    assert list(exp_file['300C'].keys())[0] == '25s', '2nd order group name incorrect'
    assert '300C/25s/wavenumber' in exp_file, 'x data (wavenumber) not stored correctly'
    assert '300C/25s/counts' in exp_file, 'y data (counts) not stored correctly'
    assert '300C/25s/residuals' in exp_file, 'residuals not stored correctly'
    assert len(exp_file['300C/25s']) == 19, 'incorrect number of peaks + raw_data stored'
    # test inputs
    try:
        dataprep.add_experiment(4.2, """ramandecompy/tests/test_files/
        CarbonMonoxide_Baseline_Calibration.xlsx""")
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.add_experiment('exp_test.hdp5', 4.2)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.add_experiment('exp_test.txt', """ramandecompy/tests/
        test_files/CarbonMonoxide_Baseline_Calibration""")
    except TypeError:
        print('A .txt file was passed to the function, and it was handled will with a TypeError.')
    os.remove('exp_test.hdf5')


def test_adjust_peaks():
    """
    A function that tests the `adjust_peaks` function from dataprep. The function first looks to
    see that no errors occur when running the function before then checking to ensure that input
    errors are handled well.
    """
    # generate test hdf5 file
    dataprep.new_hdf5('exp_test')
    dataprep.add_experiment('exp_test.hdf5', 'ramandecompy/tests/test_files/FA_3.6wt%_300C_25s.csv')
    # peaks to add and drop form auto-fitting
    add_list = [1270, 1350, 1385]
    drop_list = ['Peak_01']
    dataprep.adjust_peaks('exp_test.hdf5', '300C/25s', add_list, drop_list, plot_fits=True)
    try:
        dataprep.adjust_peaks(4.2, '300C/25s', add_list, drop_list, plot_fits=True)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.adjust_peaks('exp_test.txt', '300C/25s', add_list, drop_list, plot_fits=True)
    except TypeError:
        print('A .txt was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.adjust_peaks('exp_test.hdf5', ['300C/25s', '400C/25s'],
                              add_list, drop_list, plot_fits=True)
    except TypeError:
        print('A list was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.adjust_peaks('exp_test.hdf5', '300C/25s', 'add_list', drop_list, plot_fits=True)
    except TypeError:
        print('A str was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.adjust_peaks('exp_test.hdf5', '300C/25s', add_list, 'drop_list', plot_fits=True)
    except TypeError:
        print('A str was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.adjust_peaks('exp_test.hdf5', '300C/25s', add_list, drop_list, plot_fits=3)
    except TypeError:
        print('An int was passed to the function, and it was handled well with a TypeError.')
    os.remove('exp_test.hdf5')


def test_view_hdf5():
    """
    A function that tests the `view_hdf5` function from dataprep. The function first looks to
    see that no errors occur when running the function before then checking to ensure that input
    errors are handled well.
    """
    # test inputs
    dataprep.view_hdf5('ramandecompy/tests/test_files/dataprep_experiment.hdf5')
    try:
        dataprep.view_hdf5(4.2)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        dataprep.view_hdf5('test.txt')
    except TypeError:
        print('A .txt was passed to the function, and it was handled well with a TypeError.')
