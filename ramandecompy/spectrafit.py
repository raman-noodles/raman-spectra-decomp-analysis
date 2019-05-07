"""
This module allows for baseline subtraction using polynomial subtraction at a user-specified
degree, peak detection using scipy.signal find_peaks module, and then utilizes
Lorentzian fitting of spectra data, enabling extraction of full-width, half-max peak data.
Note that Lorentzian fitting was chosen explicitly as it is the proper descriptor of peak
shapes from Raman spectra.

Developed by the Raman-Noodles team.
"""


import matplotlib.pyplot as plt
import numpy as np
import lmfit
from sklearn.metrics import auc
from lmfit.models import PseudoVoigtModel
from peakutils.baseline import baseline
from scipy.signal import find_peaks


def peak_detect(x_data, y_data, height=None, prominence=None, distance=None):
    """
    Function that utilizes scipy to find peak maxima from input spectral data. Default
    detection parameters are chosen for the user based upon values that worked well during
    initial testing of the function; however, the option remains to adjust the parameters
    to achieve the best fit, if the user so chooses.
    WARNING: This function may return unexpected results or unreliable results for data
    that contains NaNs. Please remove any NaN values prior to passing data.

    Args:
        x_data (list like): The x-values of the spectra from which peaks will be detected.
        y_data (list like): The y-values of the spectra from which peaks will be detected.
        height (float): (Optional) The minimum floor of peak-height below which all peaks
                        will be ignored. Any peak that is detected that has a maximum height
                        less than `height` will not be collected. NOTE: This value is highly
                        sensitive to baselining, so the Raman-noodles team recommends ensuring
                        a quality baseline before use.
        prominence (float): (Optional) The prominence of the peak. In short, it's a comparison
                            of the height of a peak relative to adjacent peaks that considers
                            both the height of the adjacent peaks, as well as their distance
                            from the peak being considered. More details can be found in the
                            `peak_prominences` module from scipy.
        distance (float): (Optional) The minimum distance between adjacent peaks.

    Returns:
        peaks (list): A list of the x and y-values (in a tuple) where peaks were detected.
        peak_list (list): An list of the indices of the fed-in data that correspond to the
                          detected peaks as well as other attributes such as the prominence
                          and height.
    """
    # handling errors in inputs
    if not isinstance(x_data, (list, np.ndarray)):
        raise TypeError('Passed value of `x_data` is not a list or numpy.ndarray! Instead, it is: '
                        + str(type(x_data)))
    if not isinstance(y_data, (list, np.ndarray)):
        raise TypeError('Passed value of `y_data` is not a list or numpy.ndarray! Instead, it is: '
                        + str(type(y_data)))
    if not isinstance(height, (int, float, type(None))):
        raise TypeError('Passed value of `height` is not a int or a float! Instead, it is: '
                        + str(type(height)))
    if not isinstance(prominence, (int, float, type(None))):
        raise TypeError('Passed value of `prominence` is not a int or a float! Instead, it is: '
                        + str(type(prominence)))
    if not isinstance(distance, (int, float, type(None))):
        raise TypeError('Passed value of `distance` is not a int or a float! Instead, it is: '
                        + str(type(distance)))
    # parse inputs
    if height == None:
        height = (0.02*max(y_data))
    else:
        pass
    if prominence == None:
        prominence = (0.02*max(y_data))
    else:
        pass
    if distance == None:
        distance = 10
    else:
        pass
    # find peaks
    peak_list = find_peaks(y_data, height=height, prominence=prominence, distance=distance)
    # convert peak indexes to data values
    peaks = []
    for i in peak_list[0]:
        peak = (x_data[i], y_data[i])
        peaks.append(peak)
    return peaks, peak_list


def set_params(peaks):
    """
    This module takes in the list of peaks from the peak detection modules, and then uses
    that to initialize parameters for a set of Pseudo-Voigt models that are not yet fit.
    There is a single model for every peak.

    Args:
        peaks (list): A list containing the x and y-values (in tuples) of the peaks.

    Returns:
        mod (lmfit.models.PseudoVoigtModel or lmfit.model.CompositeModel): This is an array of
                        the initialized Pseudo-Voigt models. The array contains all of the values
                        that are found in `pars` that are fed to an lmfit lorentzian model class.
        pars (lmfit.parameter.Parameters): An array containing the parameters for each peak
                        that were generated through the use of a Lorentzian fit. The pars
                        array contains a center value, a height, a sigma, and an amplitude
                        value. The center value is allowed to vary +- 10 wavenumber from
                        the peak max that was detected in scipy. Some wiggle room was allowed
                        to help mitigate problems from slight issues in the peakdetect
                        algorithm for peaks that might have relatively flat maxima. The height
                        value was allowed to vary between 0 and 1, as it is assumed the y-values
                        are normalized. Sigma is set to a maximum of 500, as we found that
                        giving it an unbound maximum led to a number of peaks that were
                        unrealistic for Raman spectra (ie, they were far too broad, and shallow,
                        to correspond to real data. Finally, the amplitude for the peak was set
                        to a minimum of 0, to prevent negatives.
    """
    # handling errors in inputs
    if not isinstance(peaks, list):
        raise TypeError('Passed value of `peaks` is not a list! Instead, it is: '
                        + str(type(peaks)))
    for i, _ in enumerate(peaks):
        if not isinstance(peaks[i], tuple):
            raise TypeError("""Passed value of `peaks[{}]` is not a tuple.
             Instead, it is: """.format(i) + str(type(peaks[i])))
    peak_list = []
    for i, _ in enumerate(peaks):
        prefix = 'p{}_'.format(i+1)
        peak = PseudoVoigtModel(prefix=prefix)
        if i == 0:
            pars = peak.make_params()
        else:
            pars.update(peak.make_params())
        pars[prefix+'center'].set(peaks[i][0], vary=False)
        pars[prefix+'height'].set(peaks[i][1], vary=False)
        pars[prefix+'sigma'].set(100, min=0, max=350)
        pars[prefix+'amplitude'].set(min=0)
        peak_list.append(peak)
        if i == 0:
            mod = peak_list[i]
        else:
            mod = mod + peak_list[i]
    return mod, pars


def model_fit(x_data, y_data, mod, pars, report=False):
    """
    This function takes in the x and y data for the spectrum being analyzed, as well as the model
    parameters that were generated in `lorentz_params` for a single peak, and uses it to generate
    a fit for the model at that one single peak position, then returns that fit.

    Args:
        x_data (list like): The x-values for the spectrum that is being fit.
        y_data (list like): The y-values for the spectrum that is being fit.
        mod (lmfit.model.CompositeModel): This is an array of the initialized Lorentzian models
                        from the `lorentz_params` function. This array contains all of the values
                        that are found in pars, that are fed to an lmfit Lorentzian model class.
        pars (lmfit.parameter.Parameters): An array containing the parameters for each peak that
                        were generated through the use of a Lorentzian fit. The pars array contains
                        a center value, a height, a sigma, and an amplitude value. The center value
                        is allowed to vary +- 10 wavenumber from the peak max that was detected in
                        scipy. Some wiggle room was allowed to help mitigate problems from slight
                        issues in the peakdetect algorithm for peaks that might have relatively
                        flat maxima. The height value was allowed to vary between 0 and 1, as it is
                        assumed the y-values are normalized. Sigma is set to a maximum of 500, as we
                        found that giving it an unbound maximum led to a number of peaks that were
                        unrealistic for Raman spectra (ie, they were far too broad, and shallow, to
                        correspond to real data. Finally, the amplitude for the peak was set to a
                        minimum of 0, to prevent negatives.
        report (boolean): (Optional) This value details whether or not the users wants to receive
                        a report of the fit values. If True, the function will print a report of
                        the fit.
    Returns:
        out (lmfit.model.ModelResult): An lmfit model class that contains all of the fitted values
                        for the input model.
    """
    # handling errors in inputs
    if not isinstance(x_data, (list, np.ndarray)):
        raise TypeError('Passed value of `x_data` is not a list or numpy.ndarray! Instead, it is: '
                        + str(type(x_data)))
    if not isinstance(y_data, (list, np.ndarray)):
        raise TypeError('Passed value of `y_data` is not a list or numpy.ndarray! Instead, it is: '
                        + str(type(y_data)))
    if not isinstance(mod, (lmfit.models.PseudoVoigtModel, lmfit.model.CompositeModel)):
        raise TypeError("""Passed value of `mod` is not a lmfit.models.PseudoVoigtModel or a 
        lmfit.model.CompositeModel! Instead, it is: """ + str(type(mod)))
    if not isinstance(pars, lmfit.parameter.Parameters):
        raise TypeError("""Passed value of `pars` is not a lmfit.parameter.Parameters!
         Instead, it is: """ + str(type(pars)))
    if not isinstance(report, bool):
        raise TypeError('Passed value of `report` is not a boolean! Instead, it is: '
                        + str(type(report)))
    # fit model
    out = mod.fit(y_data, pars, x=x_data)
    if report:
        print(out.fit_report())
    else:
        pass
    return out


def plot_fit(x_data, y_data, fit_result, plot_components=False):
    """
    This function plots the fit, each individual Lorentzian, and the orginal data for
    visual examination.

    Args:
        x_data (list like): The x-values of the spectrum to be fitted.
        y_data (list like): The y-values of the spectrum to be fitted.
        fit_result (lmfit.model.ModelResult): An lmfit model class that contains all
                        of the fitted values for the single input model.
        plot_components (boolean): (Optional) A Boolean that dictates whether or not
                        curves for individual fit components are shown in addition to the
                        concatenated fit that shows all of the function fits. Defaults to
                        False, but True will enable component plotting.

    Returns:
        None
    """
    # handling errors in inputs
    if not isinstance(x_data, (list, np.ndarray)):
        raise TypeError('Passed value of `x_data` is not a list or numpy.ndarray! Instead, it is: '
                        + str(type(x_data)))
    if not isinstance(y_data, (list, np.ndarray)):
        raise TypeError('Passed value of `y_data` is not a list or numpy.ndarray! Instead, it is: '
                        + str(type(y_data)))
    if not isinstance(fit_result, lmfit.model.ModelResult):
        raise TypeError("""Passed value of `fit_result` is not a lmfit.model.ModelResult!
         Instead, it is: """ + str(type(fit_result)))
    if not isinstance(plot_components, bool):
        raise TypeError('Passed value of `plot_components` is not a boolean! Instead, it is: '
                        + str(type(plot_components)))
    plt.figure(figsize=(15, 6))
    plt.ylabel('Counts', fontsize=14)
    plt.xlabel('Wavenumber (cm$^{-1}$)', fontsize=14)
    plt.xlim(min(x_data), max(x_data))
    plt.plot(x_data, y_data, 'r', alpha=1, linewidth=2, label='data')
    plt.plot(x_data, fit_result.best_fit, 'c-', alpha=0.5, linewidth=3, label='fit')
    if plot_components:
        comps = fit_result.eval_components(x=x_data)
        prefix = 'p{}_'.format(1)
        plt.plot(x_data, comps[prefix], 'b--', linewidth=1, label='peak pseudo-Voigt profile')
        for i in range(1, int(len(fit_result.values)/6)):
            prefix = 'p{}_'.format(i+1)
            plt.plot(x_data, comps[prefix], 'b--', linewidth=1)
    plt.legend(fontsize=12)
    plt.show()


def export_fit_data(x_data, out):
    """
    This function returns fit information for an input lmfit model set.

    Args:
        out (lmfit.model.ModelResult): An lmfit model class that contains all of the
                        fitted values for the input model class.

    Returns:
        fit_peak_data (numpy array): An array containing both the peak number, as well as the
                        fraction Lorentzian character, sigma, center, amplitude, full-width,
                        half-max, and the height of the peaks. The data can be accessed by the
                        array positions shown here:
                            fit_peak_data[i][0] = p[i]_fraction
                            fit_peak_data[i][1] = p[i]_simga
                            fit_peak_data[i][2] = p[i]_center
                            fit_peak_data[i][3] = p[i]_amplitude
                            fit_peak_data[i][4] = p[i]_fwhm
                            fit_peak_data[i][5] = p[i]_height
    """
    # handling errors in inputs
    if not isinstance(out, lmfit.model.ModelResult):
        raise TypeError('Passed value of `out` is not a lmfit.model.ModelResult! Instead, it is: '
                        + str(type(out)))
    fit_peak_data = []
    for i in range(int(len(out.values)/6)):
        peak = np.zeros(7)
        prefix = 'p{}_'.format(i+1)
        peak[0] = out.values[prefix+'fraction']
        peak[1] = out.values[prefix+'sigma']
        peak[2] = out.values[prefix+'center']
        peak[3] = out.values[prefix+'amplitude']
        peak[4] = out.values[prefix+'fwhm']
        peak[5] = out.values[prefix+'height']
        peak[6] = auc(x_data, out.eval_components(x=x_data)[prefix])
        fit_peak_data.append(peak)
    return fit_peak_data


def fit_data(x_data, y_data):
    """
    small wrapper function used in dataprep.py
    can remove height/prominence values once the peak_detect
    function is updated to be proportional to the data
    """
    peaks = peak_detect(x_data, y_data)[0]
    mod, pars = set_params(peaks)
    out = model_fit(x_data, y_data, mod, pars)
    fit_result = export_fit_data(x_data, out)
    return fit_result
