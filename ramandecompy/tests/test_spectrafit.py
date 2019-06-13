"""
This is the unit test module for spectrafit.py
"""

import pickle
import numpy as np
import pandas as pd
import lmfit
from ramandecompy import spectrafit


data_filename = 'ramandecompy/tests/test_files/Hydrogen_Baseline_Calibration.xlsx'
DATA = pd.read_excel(data_filename, header=None, names=('x', 'y'))
X_TEST = DATA['x'].values
Y_TEST = DATA['y'].values


def test_peak_detect():
    """
    Test function that confirms spectrafit.peak_detect behaves as expected. It confirms that
    the outputs are the correct types, that all detected peaks are within the data range, that
    the list of peak indeces (peak_list[0]) has the same length as peaks, and that input
    errors are handled.
    """
    peaks, peak_list = spectrafit.peak_detect(X_TEST, Y_TEST)
    assert isinstance(peaks, list), 'expected output is list'
    assert isinstance(peaks[0], tuple), 'first peak data is not a tuple'
    for i, _ in enumerate(peaks):
        assert min(X_TEST) <= peaks[i][0] <= max(X_TEST), """
        Peak {} center is outside data range""".format(i)
    assert 0 <= peaks[0][1] <= max(Y_TEST), '1st peak maximum is outside acceptable range'
    assert len(peaks) == len(peak_list[0]), """
    Number of peak indeces different than number of peak data"""
    try:
        spectrafit.peak_detect(1.1, Y_TEST)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.peak_detect(X_TEST, 2.1)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.peak_detect(X_TEST, Y_TEST, height='one')
    except TypeError:
        print('A str was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.peak_detect(X_TEST, Y_TEST, prominence='one')
    except TypeError:
        print('A str was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.peak_detect(X_TEST, Y_TEST, distance='one')
    except TypeError:
        print('A str was passed to the function, and was handled well with a TypeError.')


def test_set_params():
    """
    Test function that confirms spectrafit.set_params behaves as expected. It confirms that
    the output types are correct, that the number of parameters is proportional to the number
    of peaks, and that input type errors are handled.
    """
    peaks = spectrafit.peak_detect(X_TEST, Y_TEST)[0]
    mod, pars = spectrafit.set_params(peaks)
    assert isinstance(mod, (lmfit.models.PseudoVoigtModel, lmfit.model.CompositeModel)), 'mod is not a lmfit CompositeModel'
    assert isinstance(pars, lmfit.parameter.Parameters), 'pars are not lmfit Parameters'
    assert len(pars) == 6*len(peaks), 'incorrect ratio of parameters to peaks'
    try:
        spectrafit.set_params(1.1)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.set_params([1, 2, 3, 4])
    except TypeError:
        print("""A list of ints was passed to the function,
         and it was handled well with a TypeError.""")


def test_model_fit():
    """
    Test function that confirms spectrafit.model_fit behaves as expected. It confirms that
    the output types are correct, that the size of the fit data matches the input, that the
    the number of output values is equal to the number of input parameters, and that input
    type errors are handled.
    """
    peaks = spectrafit.peak_detect(X_TEST, Y_TEST)[0]
    mod, pars = spectrafit.set_params(peaks)
    out = spectrafit.model_fit(X_TEST, Y_TEST, mod, pars)
    assert isinstance(out, lmfit.model.ModelResult), 'output is not a lmfit ModelResult'
    assert len(out.best_fit) == len(Y_TEST), 'size of fit incorrect'
    assert isinstance(out.best_values, dict), 'out.best_values is not a dictionary'
    assert len(out.values) == len(pars), 'number of output values not equal to number of parameters'
    try:
        spectrafit.model_fit(1.2, Y_TEST, mod, pars)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.model_fit(X_TEST, 1.3, mod, pars)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.model_fit(X_TEST, Y_TEST, [1, 2, 3, 4], pars)
    except TypeError:
        print('A list was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.model_fit(X_TEST, Y_TEST, mod, 1.4)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.model_fit(X_TEST, Y_TEST, mod, pars, report='yup!')
    except TypeError:
        print('A string was passed to the function, and was handled well with a TypeError.')


def test_plot_fit():
    """
    Test function that confirms spectrafit.plot_fit behaves as expected. This function has no
    outputs so it only tests to ensure that the input types and values are correct.
    """
    peaks = spectrafit.peak_detect(X_TEST, Y_TEST)[0]
    mod, pars = spectrafit.set_params(peaks)
    out = spectrafit.model_fit(X_TEST, Y_TEST, mod, pars)
    spectrafit.plot_fit(X_TEST, Y_TEST, out, plot_components=True)
    try:
        spectrafit.plot_fit(1.2, Y_TEST, out)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.plot_fit(X_TEST, 1.3, out)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.plot_fit(X_TEST, Y_TEST, 1.4)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.plot_fit(X_TEST, Y_TEST, [1, 2, 3, 4])
    except TypeError:
        print('A list was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.plot_fit(X_TEST, Y_TEST, out, plot_components='yup!')
    except TypeError:
        print('A str was passed to the function, and was handled well with a TypeError.')


def test_export_fit_data():
    """
    Test function that confirms spectrafit.export_fit_data behaves as expected. It confirms that the
    output type is correct, that the output shape is correct, that the number of peaks in the report
    is correct, and the input type errors are handled.
    """
    peaks = spectrafit.peak_detect(X_TEST, Y_TEST)[0]
    mod, pars = spectrafit.set_params(peaks)
    out = spectrafit.model_fit(X_TEST, Y_TEST, mod, pars)
    fit_peak_data, residuals = spectrafit.export_fit_data(X_TEST, Y_TEST, out)
    assert isinstance(fit_peak_data, list), 'output is not a list'
    assert np.asarray(fit_peak_data).shape == (int(len(out.values)/6), 7), """
    output is not the correct shape"""
    assert len(fit_peak_data) == int(len(out.values)/6), 'incorrect number of peaks exported'
    try:
        spectrafit.export_fit_data(X_TEST, 'out')
    except TypeError:
        print('A str was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.export_fit_data(4.2, out)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
        
        
def test_fit_data():
    """
    docstring
    """
    fit_result, residuals = spectrafit.fit_data(X_TEST, Y_TEST)
    assert isinstance(fit_result, list), 'output is not a list'
    for i,peak in enumerate(fit_result):
        assert isinstance(peak, list), 'output element {} is not a np.ndarray'.format(i)
        assert len(peak) in [7, 8], 'output element {} contains an incorrect number of values ({})'.format(i, len(peak))
    assert len(fit_result) == 4, 'output contains an incorrect amount of detected peaks'
    try:
        spectrafit.fit_data(X_TEST, 'Y_TEST')
    except TypeError:
        print('A str was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.fit_data(4.2, Y_TEST)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')


def test_build_custom_model():
    """
    docstring
    """
    # first build a custom peak list that only contains the pseudo voight descriptors for the first 3 peaks
    peaks =[(0.68764166, 4.52683284, 355.65041041, 8506.9345801, 9.05366569, 687.05133171, 8424.94459088),
            (0.57765067, 4.40443189, 587.33331331, 21649.13312358, 8.80886378, 1878.91168914, 21593.07349362),
            (0.65921506, 4.44539185, 816.00734735, 3733.9967507, 8.8907837, 310.71145822, 3726.8698975)]
    # not the "speculated" center and height location of the 4th peak
    peaks_add = [(1035, 256)]
    fit_result, residuals = spectrafit.build_custom_model(X_TEST, Y_TEST, peaks, peaks_add, plot_fits=True)
    assert len(fit_result) == 4, '4th peak was not successfully added to the model'
    assert isinstance(fit_result, list), '`fit_result` is not a list'
    for i, element in enumerate(fit_result):
        assert isinstance(element, list), 'element with index {} in fit_result is not a list'.format(i)
        if i <= 2:
            assert len(element) == 7, 'index {} in fit_result should have length 7'.format(i)
        else:
            assert len(element) == 8, 'index {} in fit_result should have length 8'.format(i)
            assert element[-1] == 'user_added', 'user_added tag not successfully appended' 
    assert len(residuals) == len(X_TEST), 'size of `residuals` does not match input data'
    assert isinstance(residuals[0], np.float64), 'the 1st element in `residuals` is not a np.float64'
    try:
        spectrafit.build_custom_model(4.2, Y_TEST, peaks, peaks_add, plot_fits=True)
    except TypeError:
            print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.build_custom_model(X_TEST, 'Y_TEST', peaks, peaks_add, plot_fits=True)
    except TypeError:
            print('A string was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.build_custom_model(X_TEST, Y_TEST, (1, 2, 3, 4, 5, 6), peaks_add, plot_fits=True)
    except TypeError:
            print('A tuple was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.build_custom_model(X_TEST, Y_TEST, peaks, peaks_add, plot_fits=1)
    except TypeError:
            print('An int was passed to the function, and was handled well with a TypeError.')