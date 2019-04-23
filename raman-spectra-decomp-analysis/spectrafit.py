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
from lmfit.models import PseudoVoigtModel
from peakutils.baseline import baseline
from scipy.signal import find_peaks


def subtract_baseline(y_data, deg=3, plot=False, x_data=None):
    """
    Function that fits an n-degree polynomial (default: n = 3) baseline
    to the spectral data, and subtracts it from the input data.

    Args:
        y_data (list like): The intensity data of the spectra to be baselined.
        deg (integer): (Optional) Integer value for the degree of the polynomial
                       to be used to baseline data. The value defaults to 3 if
                       no value is specified.
        plot (boolean): (Optional) Boolean value that indicates whether or not to
                        plot the baselined and original data. If true, it plots
                        both spectra on the same plot. Note that if the user wants
                        plotting functionality, x_data must be provided.
        x_data (list like): (Optional) The x-values associated with the y_data that
                            is being baselined. These values are only needed for
                            the function if the user desires plotting.

    Returns:
        y_out (list like): The baselined values of the y-axis.
    """
    # handling errors in inputs
    if not isinstance(y_data, (list, np.ndarray)):
        raise TypeError('Passed value of `y_data` is not a list or numpy.ndarray! Instead, it is: '
                        + str(type(y_data)))
    y_base = baseline(y_data, deg=deg, max_it=200)
    # to avoid strange results,
    # change all negative values to zero
    yb_plus = [0 if i < 0 else i for i in y_base]
    y_out = y_data - yb_plus
    # plot that lets you see the baseline fitting
    if plot and x_data is None:
        raise ValueError('Please add x_data as input to plot')
    elif plot:
        plt.figure(figsize=(10, 4))
        plt.plot(x_data, y_data, 'b--', label='input')
        plt.plot(x_data, y_out, 'b', label='output')
        plt.plot(x_data, yb_plus, 'r', label='baseline')
        plt.plot(x_data, y_base, 'r--', label='negative baseline')
        plt.axhline(y=0, color='orange', alpha=0.5, label='zero')
        plt.legend()
    else:
        pass
    return y_out


def peak_detect(x_data, y_data, height=0.1, prominence=0.1, distance=10):
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
    if not isinstance(height, (int, float)):
        raise TypeError('Passed value of `height` is not a int or a float! Instead, it is: '
                        + str(type(height)))
    if not isinstance(prominence, (int, float)):
        raise TypeError('Passed value of `prominence` is not a int or a float! Instead, it is: '
                        + str(type(prominence)))
    if not isinstance(distance, (int, float)):
        raise TypeError('Passed value of `distance` is not a int or a float! Instead, it is: '
                        + str(type(distance)))
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
        pars[prefix+'sigma'].set(50, min=0, max=500)
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


def export_fit_data(out):
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
        peak = np.zeros(6)
        prefix = 'p{}_'.format(i+1)
        peak[0] = out.values[prefix+'fraction']
        peak[1] = out.values[prefix+'sigma']
        peak[2] = out.values[prefix+'center']
        peak[3] = out.values[prefix+'amplitude']
        peak[4] = out.values[prefix+'fwhm']
        peak[5] = out.values[prefix+'height']
        fit_peak_data.append(peak)
    return fit_peak_data


def fit_data(x_data, y_data):
    """
    small wrapper function used in dataprep.py
    can remove height/prominence values once the peak_detect
    function is updated to be proportional to the data
    """
    peaks = peak_detect(x_data, y_data, height=10, prominence=20)[0]
    mod, pars = set_params(peaks)
    out = model_fit(x_data, y_data, mod, pars)
    fit_result = export_fit_data(out)
    return fit_result


def compound_report(compound):
    """
    Wrapper fucntion that utilizes many of the functions
    within spectrafit to give the peak information of a compound
    in shoyu_data_dict.p

    Args:
        compound (dict): a single NIST compound dictionary from shoyu_data_dict.

    Returns:
        peak_centers (list): A list with a peak center value for each peak.
        peak_sigma (list): A list with a sigma value for each peak.
        peak_ampl (list): A list with amplitudes for each peak.
        xmin (float): The minimum wavenumber value in the compound data
        xmax (float): The maximum wavenumber value in the compound data
    """
    # handling errors in inputs
    if not isinstance(compound, dict):
        raise TypeError('Passed value of `compound` is not a dict! Instead, it is: '
                        + str(type(compound)))
    x_data = compound['x']
    y_data = compound['y']
    # subtract baseline
    y_data = subtract_baseline(y_data)
    # detect peaks
    peaks = peak_detect(x_data, y_data)[0]
    # assign parameters for least squares fit
    mod, pars = set_params(peaks)
    # fit the model to the data
    out = model_fit(x_data, y_data, mod, pars)
    # export data in logical structure (see docstring)
    fit_peak_data = export_fit_data(out)
    # peak_fraction = []
    peak_center = []
    peak_sigma = []
    peak_ampl = []
    # peak_height = []
    for i, _ in enumerate(fit_peak_data):
        # peak_fraction.append(fit_peak_data[i][0])
        # if we ever need lorentzian fraction we can add it
        # right now it may break other functions 
        peak_sigma.append(fit_peak_data[i][1])
        peak_center.append(fit_peak_data[i][2])
        peak_ampl.append(fit_peak_data[i][3])
        # peak_height.append(fit_peak_data[i][5])
    xmin = min(x_data)
    xmax = max(x_data)
    return peak_center, peak_sigma, peak_ampl, xmin, xmax


def data_report(x_data, y_data):
    """
    Wrapper fucntion that utilizes many of the functions
    within spectrafit to give the peak information of inputted x
    and y data that have been initialized beforehand
    in shoyu_data_dict.p

    Args:
        x_data (list like): the wavenumber data (y) for an experimental sample.
        y_data (list like): the count data (x) for an experimental sample

    Returns:
        peak_centers (list): A list with a peak center value for each peak.
        peak_sigma (list): A list with a sigma value for each peak.
        peak_ampl (list): A list with amplitudes for each peak.
        xmin (float): The minimum wavenumber value in the compound data
        xmax (float): The maximum wavenumber value in the compound data
    """
    # handling errors in inputs
    if not isinstance(x_data, (list, np.ndarray)):
        raise TypeError('Passed value of `x_data` is not a list or numpy.ndarray! Instead, it is: '
                        + str(type(x_data)))
    if not isinstance(y_data, (list, np.ndarray)):
        raise TypeError('Passed value of `y_data` is not a list or numpy.ndarray! Instead, it is: '
                        + str(type(y_data)))
    # subtract baseline
    y_data = subtract_baseline(y_data)
    # detect peaks
    peaks = peak_detect(x_data, y_data)[0]
    # assign parameters for least squares fit
    mod, pars = set_params(peaks)
    # fit the model to the data
    out = model_fit(x_data, y_data, mod, pars)
    # export data in logical structure (see docstring)
    fit_peak_data = export_fit_data(out)
    # peak_fractions = []
    peak_centers = []
    peak_sigma = []
    peak_ampl = []
    # peak_height = []
    for i, _ in enumerate(fit_peak_data):
        # peak_fractions.append(fit_peak_data[i][0])
        # if we ever need lorentzian fraction we can add it
        # right now it may break other functions 
        peak_sigma.append(fit_peak_data[i][1])
        peak_centers.append(fit_peak_data[i][2])
        peak_ampl.append(fit_peak_data[i][3])
        # peak_height.append(fit_peak_data[i][5])
    xmin = min(x_data)
    xmax = max(x_data)
    return peak_centers, peak_sigma, peak_ampl, xmin, xmax
