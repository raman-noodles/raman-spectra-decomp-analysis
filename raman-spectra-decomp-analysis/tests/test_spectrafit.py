"""
This is the unit test module for spectrafit.py
"""

import pickle
import numpy as np
import lmfit
from ramannoodles import spectrafit


# define gaussian function used to generate test data (fake spectra)
def gaussian(x_input, var_a, var_b, var_c):
    """Simple function to produce a gaussian distribution"""
    y_output = var_a*np.exp(-((x_input-var_b)**2)/(2*(var_c**2)))
    return y_output


# generate test data
X_TEST = np.arange(500, 3500, 5)
# generate 6 gaussians
GAUSS1 = gaussian(X_TEST, 0.9, 2000, 100)
GAUSS2 = gaussian(X_TEST, 0.5, 1000, 200)
GAUSS3 = gaussian(X_TEST, 0.4, 2500, 1000)
GAUSS4 = gaussian(X_TEST, 0.7, 3200, 50)
GAUSS5 = gaussian(X_TEST, 0.7, 2200, 20)
GAUSS6 = gaussian(X_TEST, 0.7, 4000, 500)
# add gaussians to create fake spectra
Y_TEST = GAUSS1 + GAUSS2 + GAUSS3 + GAUSS4 + GAUSS5 + GAUSS6
# normalize test spectra
Y_TEST = [(Y_TEST[i] - min(Y_TEST))/(max(Y_TEST)-min(Y_TEST)) for i in range(len(Y_TEST))]
Y_TEST = np.asarray(Y_TEST)

# open spectra library
SHOYU_DATA_DICT = pickle.load(open('raman_spectra/shoyu_data_dict.p', 'rb'))


def test_subtract_baseline():
    """
    Test function that confirms spectrafit.subtract_baseline behaves as expected. It confirms
    that the output type is correct, that the output length matches that of the input, and
    that input errors are handled.
    """
    y_data = spectrafit.subtract_baseline(Y_TEST)
    assert isinstance(y_data, np.ndarray), 'output is not a numpy array'
    assert len(y_data) == len(Y_TEST), 'output length different from input'
    try:
        spectrafit.subtract_baseline(4.2)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        spectrafit.subtract_baseline(Y_TEST, plot=True)
    except ValueError:
        print('plot=True but no x_data was provided, and was handled well with a ValueError.')


def test_peak_detect():
    """
    Test function that confirms spectrafit.peak_detect behaves as expected. It confirms that
    the outputs are the correct types, that all detected peaks are within the data range, that
    the list of peak indeces (peak_list[0]) has the same length as peaks, and that input
    errors are handled.
    """
    y_test = spectrafit.subtract_baseline(Y_TEST)
    peaks, peak_list = spectrafit.peak_detect(X_TEST, y_test)
    assert isinstance(peaks, list), 'expected output is list'
    assert isinstance(peaks[0], tuple), 'first peak data is not a tuple'
    for i, _ in enumerate(peaks):
        assert min(X_TEST) <= peaks[i][0] <= max(X_TEST), """
        Peak {} center is outside data range""".format(i)
    assert 0 <= peaks[0][1] <= 1, '1st peak maximum is outside acceptable range'
    assert len(peaks) == len(peak_list[0]), """
    Number of peak indeces different than number of peak data"""
    try:
        spectrafit.peak_detect(1.1, y_test)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.peak_detect(X_TEST, 2.1)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.peak_detect(X_TEST, y_test, height='one')
    except TypeError:
        print('A str was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.peak_detect(X_TEST, y_test, prominence='one')
    except TypeError:
        print('A str was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.peak_detect(X_TEST, y_test, distance='one')
    except TypeError:
        print('A str was passed to the function, and was handled well with a TypeError.')


def test_set_params():
    """
    Test function that confirms spectrafit.set_params behaves as expected. It confirms that
    the output types are correct, that the number of parameters is proportional to the number
    of peaks, and that input type errors are handled.
    """
    y_test = spectrafit.subtract_baseline(Y_TEST)
    peaks = spectrafit.peak_detect(X_TEST, y_test)[0]
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
    y_test = spectrafit.subtract_baseline(Y_TEST)
    peaks = spectrafit.peak_detect(X_TEST, y_test)[0]
    mod, pars = spectrafit.set_params(peaks)
    out = spectrafit.model_fit(X_TEST, y_test, mod, pars)
    assert isinstance(out, lmfit.model.ModelResult), 'output is not a lmfit ModelResult'
    assert len(out.best_fit) == len(y_test), 'size of fit incorrect'
    assert isinstance(out.best_values, dict), 'out.best_values is not a dictionary'
    assert len(out.values) == len(pars), 'number of output values not equal to number of parameters'
    try:
        spectrafit.model_fit(1.2, y_test, mod, pars)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.model_fit(X_TEST, 1.3, mod, pars)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.model_fit(X_TEST, y_test, [1, 2, 3, 4], pars)
    except TypeError:
        print('A list was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.model_fit(X_TEST, y_test, mod, 1.4)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.model_fit(X_TEST, y_test, mod, pars, report='yup!')
    except TypeError:
        print('A string was passed to the function, and was handled well with a TypeError.')


def test_plot_fit():
    """
    Test function that confirms spectrafit.plot_fit behaves as expected. This function has no
    outputs so it only tests to ensure that the input types and values are correct.
    """
    y_test = spectrafit.subtract_baseline(Y_TEST)
    peaks = spectrafit.peak_detect(X_TEST, y_test)[0]
    mod, pars = spectrafit.set_params(peaks)
    out = spectrafit.model_fit(X_TEST, y_test, mod, pars)
    spectrafit.plot_fit(X_TEST, y_test, out, plot_components=True)
    try:
        spectrafit.plot_fit(1.2, y_test, out)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.plot_fit(X_TEST, 1.3, out)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.plot_fit(X_TEST, y_test, 1.4)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.plot_fit(X_TEST, y_test, [1, 2, 3, 4])
    except TypeError:
        print('A list was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.plot_fit(X_TEST, y_test, out, plot_components='yup!')
    except TypeError:
        print('A str was passed to the function, and was handled well with a TypeError.')


def test_export_fit_data():
    """
    Test function that confirms spectrafit.export_fit_data behaves as expected. It confirms that the
    output type is correct, that the output shape is correct, that the number of peaks in the report
    is correct, and the input type errors are handled.
    """
    y_test = spectrafit.subtract_baseline(Y_TEST)
    peaks = spectrafit.peak_detect(X_TEST, y_test)[0]
    mod, pars = spectrafit.set_params(peaks)
    out = spectrafit.model_fit(X_TEST, y_test, mod, pars)
    fit_peak_data = spectrafit.export_fit_data(out)
    assert isinstance(fit_peak_data, list), 'output is not a list'
    assert np.asarray(fit_peak_data).shape == (int(len(out.values)/6), 6
), """
    output is not the correct shape"""
    assert len(fit_peak_data) == int(len(out.values)/6), 'incorrect number of peaks exported'
    try:
        spectrafit.export_fit_data(mod)
    except TypeError:
        print('A str was passed to the function, and was handled well with a TypeError.')


def test_compound_report():
    """
    Test function that confirms spectrafit.compound_report behaves as expected.
    """
    compound = SHOYU_DATA_DICT['WATER']
    data = spectrafit.compound_report(compound)
    assert len(data) == 5, 'more values in return than expected'
    assert len(data[0]) == 3, 'more than three peaks detected for WATER'
    assert isinstance(data, tuple), 'output data type is not a tuple'
    try:
        spectrafit.compound_report('water')
    except TypeError:
        print('A str was passed to the function, and was handled well with a TypeError.')



def test_data_report():
    """
    Test function that confirms spectrafit.data_report behaves as expected.
    """
    compound = SHOYU_DATA_DICT['WATER']
    data = spectrafit.data_report(compound['x'], compound['y'])
    assert len(data) == 5, 'more values in return than expected'
    assert len(data[0]) == 3, 'more than three peaks detected for WATER'
    assert isinstance(data, tuple), 'output data type is not a tuple'
    try:
        spectrafit.data_report(1.1, compound['y'])
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
    try:
        spectrafit.data_report(compound['x'], 1.2)
    except TypeError:
        print('A float was passed to the function, and was handled well with a TypeError.')
