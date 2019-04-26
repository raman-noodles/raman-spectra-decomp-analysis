"""
This module allows for IR spectra to be downloaded from the NIST Chemistry WebBook.
The CAS Registry Number for the compound is used to download the data and then add
it to a dictionary called shoyu_data_dict
Functionality also allows for any .jdx file to be added to the shoyu_data_dict
pickle file.

Developed by Raman-Noodles team.
"""

import os
import pickle
import jcamp
import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from ramannoodles import spectrafit


def download_cas(cas_num):
    """
    Function that saves raman data downloaded from NIST webpage
    by calling the CAS registry number for the compound.

    Args:
        cas_num (str): The CAS number associated with the compound
                       that is intended to be downloaded.
                       It is insensitive to hyphens.

    Returns:
        This function has no returns.
    """
    # handling errors in inputs
    if not isinstance(cas_num, str):
        raise TypeError('Passed value of `cas_num` is not a string! Instead, it is: '
                        + str(type(cas_num)))
    # drop any '-' from cas_num
    cas_num = ''.join(cas_num.split('-'))
    if not os.path.exists('../raman_spectra/'+cas_num+'_NIST_IR.jdx'):
        os.makedirs('../raman_spectra', exist_ok=True)
        url = 'https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=C'+cas_num+'&Index=1&Type=IR'
        data = requests.get(url, allow_redirects=True)
        if data.text == '##TITLE=Spectrum not found.\n##END=\n':
            print("""Bad Request: CAS # not found. Please verify
             with NIST Chemistry WebBook.""")
        else:
            if data.status_code == 200:
                open('../raman_spectra/'+cas_num+'_NIST_IR.jdx', 'wb').write(data.content)
                print('file downloaded to raman_spectra folder')
            else:
                print('Request status: {}'.format(data.status_code))
    else:
        print('file already in raman_spectra folder')


def add_jdx(filename, label=None):
    """
    Function that reads and adds a .jdx file to the raman_data_dict pickle file.

    Args:
        filename (str): This filename is the exact file name that is affiliated with the
                        .jdx file that has been downloaded that the user wants to add to
                        their pickle file.

    Returns:
        shoyu_data_dict (dict): This is the dictionary that contains the data loaded from
                                the pickle file and is the common way in which the user
                                interacts with data in this software.
    """
    if not isinstance(filename, str):
        raise TypeError("Passed value of `filename` is not a string! Instead, it is: "
                        + str(type(filename)))
    shoyu_data_dict = pickle.load(open('raman_spectra/shoyu_data_dict.p', 'rb'))
    data = jcamp.JCAMP_reader(filename)
    y_abs = 1 - data['y']
    data['yunits'] = 'ABSORBANCE'
    data['y'] = y_abs
    if label is None:
        shoyu_data_dict.update({data['title'].upper(): data})
        print('{} loaded into the dictionary - shoyu_data_dict.p'.format(data['title']))
    else:
        shoyu_data_dict.update({label: data})
        print('{} loaded into the dictionary - shoyu_data_dict.p'.format(label))
    pickle.dump(shoyu_data_dict, open('../raman_spectra/shoyu_data_dict.p', 'wb'))
    return shoyu_data_dict


def initialize_standard_library():
    """
    Function that downloads a standard library of raman spectra from the NIST Chemistry
    WebBook. It generates a folder and a pickle file for storing data for future use.
    This function must be run BEFORE any other function in this package to generate the
    shoyu_data_dict.p file

    Args:
        This function does not take input parameters

    Returns:
        This function has no returns.
    """
    # dictionary of CAS registry numbers for standard library
    cas_lib = {'water':'7732-18-5',
               'carbon monoxide':'630-08-0',
               'carbon dioxide':'124-38-9',
               'formic acid':'64-18-6',
               'isopropyl alcohol':'67-63-0',
               'ethanol':'64-17-5',
               'acetone':'67-64-1',
               'pentane':'109-66-0'}
    # initialize empty shoyu_data_dict
    shoyu_data_dict = {}
    os.makedirs('../raman_spectra', exist_ok=True)
    pickle.dump(shoyu_data_dict, open('../raman_spectra/shoyu_data_dict.p', 'wb'))
    for item in cas_lib:
        cas_num = ''.join(cas_lib[item].split('-'))
        download_cas(cas_num)
        add_jdx('../raman_spectra/'+cas_num+'_NIST_IR.jdx', label=None)


def more_please(cas_num, label=None):
    """
    Function that downloads a spectra from the NIST
    database, adds it to shoyu_data_dict, pickles shoyu_data_dict
    and returns the updated shoyu_data_dict.

    Args:
        cas_num (str): The CAS number that is associated with the compound intended
                       to be downloaded. It must be in the format of a string, but it
                       is insensitive to hyphens.
        label (str): (Optional) By passing this label, instead of using the title found
                     on the NIST webbook, when the compound spectral data is added to the
                     shoyu_data_dict it will use the text of `label` as the dictionary
                     key for this spectral data.

    Returns:
        shoyu_data_dict (dict): This is the dictionary that contains the data loaded from
                                the pickle file, and is the common way in which the user
                                interacts with data in this software.
    """
    # handling errors in inputs
    if not isinstance(cas_num, str):
        raise TypeError("Passed value of `cas_num` is not a string! Instead, it is: "
                        + str(type(cas_num)))
    # Drop any '-' from cas_num
    cas_num = ''.join(cas_num.split('-'))
    download_cas(cas_num)
    shoyu_data_dict = add_jdx('../raman_spectra/'+cas_num+'_NIST_IR.jdx', label)
    return shoyu_data_dict


def clean_spectra(compound):
    """
    Function that cleans the data of any duplicate x-values that will cause
    errors for interpolation.

    Args:
        compound (str): shoyu_data_dict key for the desired compound

    Returns:
        comp_data_clean (list): list of tuples containing all the non-repeated
                                x and y values
    """
    # handling errors in inputs
    if not isinstance(compound, dict):
        raise TypeError("Passed value of `compound` is not a dictionary! Instead, it is: "
                        + str(type(compound)))
    # extract data from dictionary
    x_comp = compound['x']
    y_comp = compound['y']
    y_comp = spectrafit.subtract_baseline(y_comp)
    # zip x and y values
    comp_data = list(zip(x_comp, y_comp))
    #clean comp1
    comp_data_clean = []
    for i in range(1, len(comp_data)-1):
        if comp_data[i][0] == comp_data[i-1][0]:
            pass
        else:
            comp_data_clean.append(comp_data[i])
    return comp_data_clean


def interpolate_spectra(comp_data_clean):
    """
    Function that produces interpolated values for the spectra at integer values
    across the range of the data.

    Args:
        comp_data_clean (list): list of tuples containing all the non-repeated
                                x and y values

    Returns:
        comp_data_int (list): list of interpolated values for the spectra at integer
                              values across the range of the input data
    """
    # handling errors in inputs
    if not isinstance(comp_data_clean, list):
        raise TypeError('Passed value of `comp_data_clean` is not a list! Instead, it is: '
                        + str(type(comp_data_clean)))
    for i, _ in enumerate(comp_data_clean):
        if not isinstance(comp_data_clean[i], tuple):
            raise TypeError('Component of the passed value is not a tuple! Instead, it is: '
                            + str(type(comp_data_clean[i])))
    # unzip data
    x_comp, y_comp = zip(*comp_data_clean)
    # interpolate data
    comp_int = interpolate.interp1d(x_comp, y_comp, kind='cubic')
    # define ranges
    comp_range = np.arange(int(min(x_comp))+1, int(max(x_comp)), 1)
    # run interpolations
    y_comp_interp = comp_int(comp_range)
    # zip interpolated values
    comp_data_int = list(zip(comp_range, y_comp_interp))
    return comp_data_int


def sum_spectra(comp1_data_int, comp2_data_int):
    """
    Function that adds the interpolated values for two spectra together.

    Args:
        comp1_data_int (list): list of tuples containing all the non-repeated
                               x and y values for compound 1
        comp2_data_int (list): list of tuples containing all the non-repeated
                               x and y values for compound 2

    Returns:
        x_combined (list): list of summed x-values across the range of
                           of the two compounds
        y_combined (list): list of summed y-values across the range of
                           of the two compounds
    """
    # handling errors in inputs
    if not isinstance(comp1_data_int, list):
        raise TypeError('Passed value of `comp1_data_int` is not a list! Instead, it is: '
                        + str(type(comp1_data_int)))
    # add the two spectra
    combined = sorted(comp1_data_int + comp2_data_int)
    # add by like
    same_x = {x:0 for x, _ in combined}
    for name, num in combined:
        same_x[name] += num
    sum_combined = list(map(tuple, same_x.items()))
    # unzip
    x_combined, y_combined = zip(*sum_combined)
    # set as arrays
    x_combined = np.asarray(x_combined)
    y_combined = np.asarray(y_combined)
    return x_combined, y_combined


def combine_spectra(compound_1, compound_2, plot=False):
    """
    Wrapping function that sums two compounds from shoyu_data_dict.p
    together. There is an optional plotting function embedded.

    Args:
         compound_1 (str): dictionary key for the compound in shoyu_data_dict.p
         compound_2 (str): dictionary key for the compound in shoyu_data_dict.p
         plot (boolean): (Optional) This argument is used to dictate whether or not you
                         would like to output a plot which shows the combined spectra,
                         as well as the two original spectra, overlaid on the same plot.
                         Defaults to False.

     Returns:
         x_combined (numpy array): The x-values of the new spectra that contains the
                                   combined values of the two spectra that were input.
         y_combined (numpy array): The y-values of the new spectra that contains the
                                   combined values of the two spectra that were input.
    """
    # handling errors in inputs
    if not isinstance(compound_1, dict):
        raise TypeError("Passed value of `compound_1` is not a dictionary! Instead, it is: "
                        + str(type(compound_1)))
    if not isinstance(compound_2, dict):
        raise TypeError("Passed value of `compound_2` is not a dictionary! Instead, it is: "
                        + str(type(compound_1)))
    data1 = clean_spectra(compound_1)
    data2 = clean_spectra(compound_2)
    comp1_data_int = interpolate_spectra(data1)
    comp2_data_int = interpolate_spectra(data2)
    x_combined, y_combined = sum_spectra(comp1_data_int, comp2_data_int)
    if plot:
        # plot original data and combined plot
        plt.figure(figsize=(15, 5))
        plt.plot([i[0] for i in data1], [i[1] for i in data1], 'b--', label=compound_1['title'])
        plt.plot([i[0] for i in data2], [i[1] for i in data2], 'g--', label=compound_2['title'])
        plt.plot(x_combined, y_combined, 'r', label='Combination', linewidth=2, alpha=0.7)
        plt.legend()
        plt.xlabel('cm$^{-1}$', fontsize=14)
        plt.ylabel('Absoprtion', fontsize=14)
    return x_combined, y_combined
