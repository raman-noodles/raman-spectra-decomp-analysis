"""
Module used to unit test the functionality and outputs of the peakidentify.py module
"""
"""This function takes in compounds from a dictionary from shoyu, and, using spectrafit,
identifies peaks found in both the fed-in known spectra, as well as the unknown spectra
to be analyzed. From that identification, it then classifies the peaks in the unknown
spectra based on the fed-in known spectra.
 """
import math
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


def test_keyfinder():
    """
    This function adds Raman experimental data to an existing hdf5 file. It uses the
    spectrafit.fit_data function to fit the data before saving the fit result and
    the raw data to the hdf5 file. The data_filename must be in a standardized format
    to interact properly with this function. It must take the form anyname_temp_time.xlsx
    (or .csv) since this function will parse the the temp and time from the filename to
    label the data and fit result in the hdf5 file.

    Args:
        hdf5_filename (str): the filename and location of an existing hdf5 file to add the
                             experiment data too.
        data_filename (str): the filename and location of raw Raman spectroscopy data in
                             either the form of an .xlsx or a .csv with the wavenumber data
                             in the 1st column and the counts data in the 2nd column. These
                             files should contain only the wavenumber and counts data
                             (no column labels).

    Returns:
        None
    """
def test_combine_experiment():
    """
    This function adds Raman experimental data to an existing hdf5 file. It uses the
    spectrafit.fit_data function to fit the data before saving the fit result and
    the raw data to the hdf5 file. The data_filename must be in a standardized format
    to interact properly with this function. It must take the form anyname_temp_time.xlsx
    (or .csv) since this function will parse the the temp and time from the filename to
    label the data and fit result in the hdf5 file.

    Args:
        hdf5_filename (str): the filename and location of an existing hdf5 file to add the
                             experiment data too.
        data_filename (str): the filename and location of raw Raman spectroscopy data in
                             either the form of an .xlsx or a .csv with the wavenumber data
                             in the 1st column and the counts data in the 2nd column. These
                             files should contain only the wavenumber and counts data
                             (no column labels).

    Returns:
        None
    """


def test_interpolated_spectra():
    """
    This function adds Raman experimental data to an existing hdf5 file. It uses the
    spectrafit.fit_data function to fit the data before saving the fit result and
    the raw data to the hdf5 file. The data_filename must be in a standardized format
    to interact properly with this function. It must take the form anyname_temp_time.xlsx
    (or .csv) since this function will parse the the temp and time from the filename to
    label the data and fit result in the hdf5 file.

    Args:
        hdf5_filename (str): the filename and location of an existing hdf5 file to add the
                             experiment data too.
        data_filename (str): the filename and location of raw Raman spectroscopy data in
                             either the form of an .xlsx or a .csv with the wavenumber data
                             in the 1st column and the counts data in the 2nd column. These
                             files should contain only the wavenumber and counts data
                             (no column labels).

    Returns:
        None
    """