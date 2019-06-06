"""
This model allows a user who has all their experimental data saved in a directory folder to mass import the files and have an hdf5 file created that has the data organized and fit. This module interacts closely with the dataprep.py module also
included with this package. 

The advantage of this model is that it is an automated (looped) version of the `add_experiment` function in dataprep.py thus allowing a user not to have to manually import data files one at a time.

The user needs to have had an organized filename structure as the organized hdf5 file relies on it. For this use case the compound name 'FA_' (Formic Acid) is the first part of the files in the directory and then the 

Developed by the Raman-Noodles team (2019 DIRECT Cohort, University of Washington)
"""


#initial imports
import os
import h5py
import matplotlib.pyplot as plt
from ramandecompy import dataprep


def data_import(hdf5_filename, directory):
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
        directory (str): the folder location of raw Raman spectroscopy data in
                             either the form of an .xlsx or a .csv with the wavenumber data
                             in the 1st column and the counts data in the 2nd column. These
                             files should contain only the wavenumber and counts data
                             (no column labels).

    Returns:
        None
    """
    # open hdf5 file as read/write
    dataprep.new_hdf5(hdf5_filename)
    exp_file = h5py.File(hdf5_filename+'.hdf5', 'r+')
    for filename in os.listdir(directory):
        if filename.startswith('FA_') and filename.endswith('.csv'):
            locationandfile = directory + filename
            dataprep.add_experiment(str(hdf5_filename)+'.hdf5', locationandfile)
            print('Data from {} fit with compound pseudo-Voigt model. Results saved to {}.'.format(filename, hdf5_filename))
            # printing out to user the status of the import (because it can take a long time if importing a lot of data,
            # about minute/data set for test files
            exp_file.close()
            continue
        else:
            print('Data from {} fit with compound pseudo-Voigt model. Results saved to {}.'.format(filename, hdf5_filename))
            exp_file.close()
            continue
    return