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
    fit_peak_data = spectrafit.export_fit_data(X_TEST, out)
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
    """docstring"""
    fit_result = spectrafit.fit_data(X_TEST, Y_TEST)
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
    
