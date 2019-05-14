"""
Module used to unit test the functionality and outputs of the peakidentify.py module
"""
# IMPORTING MODULES
import os
import h5py
import numpy as np
from ramandecompy import spectrafit
from ramandecompy import peakidentify
from ramandecompy import dataprep

def test_peak_assignment():
    """This function tests the operation of the peak_assignment function in peakidentify.py"""
    #First, generate a testing dataset.
    hdf5_filename = 'dataprep_calibration_test.hdf5'
    key = 'Methane'
    hdf5_expfilename = 'dataprep_experiment_test.hdf5'
    expkey = '300C/25s'
    hdf5 = h5py.File(hdf5_filename, 'r')
    exphdf5 = h5py.File(hdf5_expfilename, 'r')
    unknown_x = list(exphdf5['{}/wavenumber'.format(expkey)])
    unknown_y = list(exphdf5['{}/counts'.format(expkey)])
    unknown_x = np.asarray(unknown_x)
    unknown_y = np.asarray(unknown_y)
    known_compound_list = list(hdf5.keys())
    precision = 0.08

    #Various try statements to make sure that bad inputs are handled correctly.

    try:
        peak_assignment(hdf5_expfilename, expkey,
                                     hdf5_filename, 'String', precision, False)
    except TypeError:
        print("An invalid known_compound_list was passed to the function, "
              "and it was handled well with a TypeError.")

    try:
        peak_assignment(hdf5_expfilename, expkey,
                                     hdf5_filename, [1, 3, 6], precision, False)
    except TypeError:
        print("An invalid element inside known_compound_list was passed to "
              "the function, and it was handled well with a TypeError.")

    try:
        peak_assignment(hdf5_expfilename, expkey,
                                     hdf5_filename, known_compound_list, 'precision', False)
    except TypeError:
        print("An invalid precision value was passed to the function, and "
              "it was handled well with a TypeError.")

    try:
        peak_assignment(hdf5_expfilename, expkey, hdf5_filename, known_compound_list, precision, 'False')
        
    except TypeError:
        print("An invalid plot value was passed to the function, and it "
              "was handled well with a TypeError.")

def test_compare_unknown_to_known():
    """This function tests the operation of the compare_unknown_to_known
    function in peakidentify.py"""
    #Build our test dataset.
    hdf5_filename = 'dataprep_calibration_test.hdf5'
    key = 'Methane'
    hdf5_expfilename = 'dataprep_experiment_test.hdf5'
    expkey = '300C/25s'
    hdf5 = h5py.File(hdf5_filename, 'r')
    exphdf5 = h5py.File(hdf5_expfilename, 'r')
    unknown_x = list(exphdf5['{}/wavenumber'.format(expkey)])
    unknown_y = list(exphdf5['{}/counts'.format(expkey)])
    unknown_x = np.asarray(unknown_x)
    unknown_y = np.asarray(unknown_y)
    known_compound_list = list(hdf5.keys())
    precision = 0.08
    known_peaks = []
    known_peaks_list = []
    for i, _ in enumerate(known_compound_list):
        for j,peak in enumerate(list(hdf5[known_compound_list[i]])[:-2]):
            known_peaks_list.append(list(hdf5['{}/{}'.format(known_compound_list[i], peak)])[2])
            known_peaks.append(known_peaks_list[i])
    unknown_peaks = []
    for i,_ in enumerate(list(exphdf5[expkey])[:-2]):
        if i < 9:
            unknown_peaks.append(list(exphdf5['{}/Peak_0{}'.format(expkey, i+1)])[2])
        else:
            unknown_peaks.append(list(exphdf5['{}/Peak_{}'.format(expkey, i+1)])[2])

    try:
        compare_unknown_to_known(1, known_peaks, precision, hdf5_filename, key)
    except TypeError:
        print("An invalid unknown_peaks value was passed to the function, "
              "and was handled correctly.")

    try:
        compare_unknown_to_known(unknown_peaks, 'known_peaks', precision, hdf5_filename, key)
    except TypeError:
        print("An invalid known_peaks value was passed to the function, "
              "and was handled correctly.")

    try:
        compare_unknown_to_known(unknown_peaks, known_peaks, 'precision', hdf5_filename, key)
    except TypeError:
        print("An invalid precision value was passed to the function, and "
              "was handled correctly.")

    #After testing for resilience to unexpected inputs, now ensure outputs are performing correctly

    #First, make sure function is returning the list.
    assert isinstance(compare_unknown_to_known(
        unknown_peaks, known_peaks, precision, hdf5_filename, key), np.ndarray), (""
                                                                 "Function is not returning list")

    #Compare one set of peaks to itself. The full association matrix should have all values = 1.
    self_comp = np.mean(compare_unknown_to_known(known_peaks,
                                                              known_peaks, precision,hdf5_filename, key))
    assert self_comp == 1, ("Peak Assignment Error. Comparison of compound "
                            "against itself should find all peaks.")

    dif_comp = np.mean(compare_unknown_to_known([1, 3, 6],
                                                             [1000, 2000, 5000], precision, hdf5_filename, key))
    assert dif_comp == 0, ("Peak Assignment Error. Passed values should "
                           "have no matching assignments.")

def test_peak_position_comparisons():
    """This function tests the operation of the peak_position_comparisons
    function in peakidentify. Said function returns a list of strings that
    contain text assignments of each peak in the unknown spectrum."""

    #First, generate good data.
    hdf5_filename = 'dataprep_calibration_test.hdf5'
    key2 = 'Methane'
    key = 'Hydrogen'
    hdf5_expfilename = 'dataprep_experiment_test.hdf5'
    expkey = '300C/25s'
    hdf5 = h5py.File(hdf5_filename, 'r')
    exphdf5 = h5py.File(hdf5_expfilename, 'r')
    unknown_x = list(exphdf5['{}/wavenumber'.format(expkey)])
    unknown_y = list(exphdf5['{}/counts'.format(expkey)])
    unknown_x = np.asarray(unknown_x)
    unknown_y = np.asarray(unknown_y)
    known_compound_list = list(hdf5.keys())
    unknown_peaks = []
    precision = 0.08
    for i,_ in enumerate(list(exphdf5[expkey])[:-2]):
        if i < 9:
            unknown_peaks.append(list(exphdf5['{}/Peak_0{}'.format(expkey, i+1)])[2])
        else:
            unknown_peaks.append(list(exphdf5['{}/Peak_{}'.format(expkey, i+1)])[2])
    known_peaks = []
    known_peaks_list = [] 
    association_matrix = []
    for i, _ in enumerate(known_compound_list):
        for j,peak in enumerate(list(hdf5[known_compound_list[i]])[:-2]):
            known_peaks_list.append(list(hdf5['{}/{}'.format(known_compound_list[i], peak)])[2])
        known_peaks.append(known_peaks_list)  
        association_matrix.append(compare_unknown_to_known(
            unknown_peaks, known_peaks[i], precision,
            hdf5_expfilename, expkey))
    #Then, test error handling of bad inputs for the function.
    try:
        peak_position_comparisons(1, known_peaks,
                                  known_compound_list,
                                  association_matrix,
                                  hdf5_filename,
                                  key)
    except TypeError:
        print("An invalid unknown_peaks value was passed to the function, "
              "and was handled correctly.")

    try:
        peak_position_comparisons(unknown_peaks,
                                  'known_peaks',
                                  known_compound_list,
                                  association_matrix, 
                                  hdf5_filename,
                                  key)
    except TypeError:
        print("An invalid known_peaks value was passed to the function, "
              "and was handled correctly.")

    try:
        peak_position_comparisons(unknown_peaks,
                                  known_peaks,
                                  'known_compound_list',
                                  association_matrix,
                                  hdf5_filename,
                                  key)
    except TypeError:
        print("An invalid known_compound_list value was passed to the function,"
              "and was handled correctly.")

    try:
        peak_position_comparisons(unknown_peaks,
                                  known_peaks,
                                  known_compound_list,
                                  'association_matrix',
                                  hdf5_filename,
                                  key)
    except TypeError:
        print("An invalid association_matrix value was passed to the function,"
              "and was handled correctly.")

    #Check to make sure the function is returning a list.
    assert isinstance(peak_position_comparisons(
        unknown_peaks, known_peaks, known_compound_list,
        association_matrix, hdf5_filename,
        key), list), "The function is not returning a list."

    #Test a call that says that no peaks have associations
    association_matrix_0 = []
    association_matrix_0.append(compare_unknown_to_known(known_peaks[0],
                                                         known_peaks[1],
                                                         0.08,
                                                         hdf5_filename,
                                                         key))
    zero_output = peak_position_comparisons(known_peaks[0],
                                            [known_peaks[1]],
                                            [key],
                                            association_matrix_0,
                                            hdf5_filename,
                                            key)[0]
    assert zero_output[0] == 'Hydrogen', """The function is not properly
    handling unassigned peaks."""

    #Test the function to make sure that it has the right functionality
    association_matrix = []
    #Generate a matrix with all associations equal to 1
    association_matrix.append(compare_unknown_to_known(known_peaks[0],
                                                                    known_peaks[0],
                                                                    0.08,
                                                                    hdf5_filename,
                                                                    key))

    #change the middle index to 0
    association_matrix[0][1] = 0
    test_peak_labels = peak_position_comparisons(known_peaks[0],
                                                              [known_peaks[0]],
                                                              [key],
                                                              association_matrix,
                                                              hdf5_filename,
                                                              key)
    assert test_peak_labels[0][0] == 'Hydrogen', """The funciton is
    not correctly assigning peaks when association matrix = 1"""
    assert test_peak_labels[1][0] == 'Unassigned', """The function is
    not correctly handling a lack of peak assignments"""
    assert test_peak_labels[2][0] == 'Hydrogen', """The funciton is
    not correctly assigning peaks when association matrix = 1"""

def test_percentage_of_peaks_found():
    """This function tests the operation of the
    percentage_of_peaks_found function in peakidentify.py"""
    #First, generate good data.
    hdf5_filename = 'dataprep_calibration_test.hdf5'
    key = 'Hydrogen'
    hdf5_expfilename = 'dataprep_experiment_test.hdf5'
    expkey = '300C/25s'
    hdf5 = h5py.File(hdf5_filename, 'r')
    exphdf5 = h5py.File(hdf5_expfilename, 'r')
    # extract spectra data
    known_x = list(hdf5['{}/wavenumber'.format(key)])
    known_y = list(hdf5['{}/counts'.format(key)])
    unknown_x = list(exphdf5['{}/wavenumber'.format(expkey)])
    unknown_y = list(exphdf5['{}/counts'.format(expkey)])
    known_x = np.asarray(known_x)
    known_y = np.asarray(known_y)
    unknown_x = np.asarray(unknown_x)
    unknown_y = np.asarray(unknown_y)
    known_compound_list = list(hdf5.keys())
    unknown_peaks = []
    precision = 0.08
    for i,_ in enumerate(list(exphdf5[expkey])[:-2]):
        if i < 9:
            unknown_peaks.append(list(exphdf5['{}/Peak_0{}'.format(expkey, i+1)])[2])
        else:
            unknown_peaks.append(list(exphdf5['{}/Peak_{}'.format(expkey, i+1)])[2])
    known_peaks = []
    known_peaks_list =[]
    association_matrix = []
    for i, _ in enumerate(known_compound_list):
        for j,peak in enumerate(list(hdf5[known_compound_list[i]])[:-2]):
            known_peaks_list.append(list(hdf5['{}/{}'.format(known_compound_list[i], peak)])[2])
        known_peaks.append(known_peaks_list)    
        association_matrix.append(compare_unknown_to_known(
            unknown_peaks, known_peaks[i], precision,
            hdf5_expfilename, expkey))

    #Test for input error handling.
    
    try:
        percentage_of_peaks_found([[0], [1], [2], [3]], 
                              association_matrix, 
                              known_compound_list, hdf5_filename)
    except TypeError:
        print("""The function correctly handled the error when a list of ints
        was input instead of the known_peaks list""")
        
    try:
        percentage_of_peaks_found(1, association_matrix,
                                  known_compound_list,
                                  hdf5_filename)
        
    except TypeError:
        print("""The function correctly handled the error when an int
        was input instead of the known_peaks list""")

    try:
        percentage_of_peaks_found(known_peaks, 1,
                                  known_compound_list,
                                  hdf5_filename)
        
    except TypeError:
        print("""The function correctly handled the error when an int
        was input instead of the association matrix""")

    try:
        percentage_of_peaks_found(known_peaks,
                                  association_matrix,
                                  'known_compound_list',
                                  hdf5_filename)
    except TypeError:
        print("""The function correctly handled the error when a string
        was input instead of the known_compound_list""")

    try:
        percentage_of_peaks_found(known_peaks,
                                  association_matrix,
                                  ['expkey'],
                                  hdf5_filename)
    except TypeError:
        print("""The function correctly handled the case where the compound
        list contains something that is not a compound""")

    #Test to make sure function returns a dictionary.
    assert isinstance(percentage_of_peaks_found(
        known_peaks,
        association_matrix,
        known_compound_list,
        hdf5_filename), dict), """The function is not
        returning a dictionary."""

#     #Test for function output.
    H_peaks = []
    for _,peak in enumerate(list(hdf5[key])[:-2]):
        H_peaks.append(list(hdf5['{}/{}'.format(key, peak)])[2])
    H_dict_0 = percentage_of_peaks_found([H_peaks],
                                         [[0, 0, 0,0]],
                                         [key],
                                         hdf5_filename)
    assert H_dict_0[key] == 0, """The function is not correctly
    calculating percentages when no peaks are found"""

    H_dict_1 = percentage_of_peaks_found([H_peaks],
                                         [[1, 1, 1,1]],
                                         [key],
                                         hdf5_filename)
    assert H_dict_1[key] == 100, """The function is not correctly
    calculating percentages when all peaks are found"""


def test_plotting_peak_assignments():
    """This function tests the operation of the peak_assignment
    function in peakidentify.py"""
    #First, generate good data.
    hdf5_filename = 'dataprep_calibration_test.hdf5'
    key = 'Methane'
    hdf5_expfilename = 'dataprep_experiment_test.hdf5'
    expkey = '300C/25s'
    hdf5 = h5py.File(hdf5_filename, 'r')
    exphdf5 = h5py.File(hdf5_expfilename, 'r')
    # extract spectra data
    unknown_x = list(exphdf5['{}/wavenumber'.format(expkey)])
    unknown_y = list(exphdf5['{}/counts'.format(expkey)])
    # extract fitted peak center values
    peak_centers = []
    for _,peak in enumerate(list(hdf5[key])[:-2]):
        peak_centers.append(list(hdf5['{}/{}'.format(key, peak)])[2])
    unknown_x = np.asarray(unknown_x)
    unknown_y = np.asarray(unknown_y)
    precision = 0.08
    unknown_peaks = []
    #Lets identify the peaks in the unknown spectrum.
    for i,_ in enumerate(list(exphdf5[expkey])[:-2]):
        if i < 9:
            unknown_peaks.append(list(exphdf5['{}/Peak_0{}'.format(expkey, i+1)])[2])
        else:
            unknown_peaks.append(list(exphdf5['{}/Peak_{}'.format(expkey, i+1)])[2])

    #OK, next identify all of the peaks present in the known compound set.
    #For efficiency, we'll also compare them against the unknown in the same for loop.
    known_peaks = []
    known_peaks_list = []
    known_compound_list = list(hdf5.keys())
    assignment_matrix = []
    for i, _ in enumerate(known_compound_list):
        for j,peak in enumerate(list(hdf5[known_compound_list[i]])[:-2]):
            known_peaks_list.append(list(hdf5['{}/{}'.format(known_compound_list[i], peak)])[2])
        known_peaks.append(known_peaks_list)    
        assignment_matrix.append(compare_unknown_to_known(
            unknown_peaks, known_peaks[i], precision,
            hdf5_expfilename, expkey))
    
    #Ok, so that generates a full association matrix that contains everything
    #we need to assign peaks.
    #Now, let's go through and actually assign text to peaks.
    unknown_peak_assignments = peak_position_comparisons(unknown_peaks,
                                                        known_peaks,
                                                        known_compound_list,
                                                        assignment_matrix,
                                                        hdf5_expfilename,
                                                        expkey)
    #Test for input error handling.
    try:
        plotting_peak_assignments(1,
                                      unknown_y,
                                      unknown_peaks,
                                      unknown_peak_assignments, 
                                      hdf5_expfilename,
                                      expkey)
    except TypeError:
        print("""The function correctly handled the error
        when an int was input instead of the unknown_x list""")

    try:
        plotting_peak_assignments(unknown_x,
                                      3,
                                      unknown_peaks,
                                      unknown_peak_assignments, 
                                      hdf5_expfilename,
                                      expkey)
    except TypeError:
        print("""The function correctly handled the error when an int
        was input instead of the unknown_y list""")

    try:
        plotting_peak_assignments(unknown_x,
                                  unknown_y,
                                  'unknown_peaks',
                                  unknown_peak_assignments,
                                  hdf5_expfilename,
                                  expkey)
    except TypeError:
        print("""The function correctly handled the error when a string
        was input instead of the unknown_peaks list""")

    try:
        plotting_peak_assignments(unknown_x,
                                  unknown_y,
                                  unknown_peaks,
                                  3,
                                  hdf5_expfilename,
                                  expkey)
    except TypeError:
        print("""The function correctly handled the error when an int
        was input instead of the unknown_peak_assignments""")

    try:
        plotting_peak_assignments(unknown_x,
                                  unknown_y,
                                  unknown_peaks,
                                  ['WATER', 23, 'CO'],
                                  hdf5_expfilename,
                                  expkey)
                                              
    except TypeError:
        print("""The function correctly handled the case when an int
        was passed in the unknown_peak_assignment list""")
        
    try:
        plotting_peak_assignments(unknown_x,
                                  unknown_y,
                                  unknown_peaks,
                                  ['WATER', 23, 'CO'],
                                  hdf5_expfilename,
                                  expkey)
                                              
    except TypeError:
        print("""The function correctly handled the case when an int
        was passed in the unknown_peak_assignment list""")
        
    try:
        plotting_peak_assignments(unknown_x,
                                  unknown_y,
                                  unknown_peaks,
                                  unknown_peak_assignments,
                                  3,
                                  expkey)
                                              
    except TypeError:
        print("""The function correctly handled the case when an int
        was passed in the hdf5_filename""")
        
    try:
        plotting_peak_assignments(unknown_x,
                                  unknown_y,
                                  unknown_peaks,
                                  unknown_peak_assignments,
                                  hdf5_expfilename,
                                  3)
                                              
    except TypeError:
        print("""The function correctly handled the case when an int
        was passed in the key""")

def test_peak_1d_score():
    """Evaluates the functionality of the peak_1D_score function"""
    # Initialize the test arguments
    row_i = [0, 1]
    row_j = [2, 1]
    rowcat = row_i + row_j
    arraya = np.array([[0, 1], [2, 1], [0, 3]])
    arraycat = np.concatenate((arraya[0], arraya[2]))

    # Run Bad Function for lists
    try:
        testscore = peakidentify.peak_1d_score(row_i, row_j, -1)
    except ValueError:
        print("An invalid scoremax value was passed to the function, "
              "and was handled correctly.")

    # Run Bad Function for arrays
    try:
        arrayscore = peakidentify.peak_1d_score(arraya[0], arraya[2], -1)

    except ValueError:
        print("An invalid scoremax value was passed to the function, "
              "and was handled correctly.")

    # Running a good example
    testscore = peakidentify.peak_1d_score(row_i, row_j, 1)
    arrayscore = peakidentify.peak_1d_score(arraya[0], arraya[2], 1)

    # make assertions
    assert len(row_i) == len(row_j), 'Input lengths do not match'
    assert len(arrayscore[0][:]) == len(arraycat), """Output list length
    different than concatenated lists length"""
    for i in range(len(rowcat)):
        assert 0 <= testscore[0][i] <= 1, 'Output value outside acceptable range'
        assert 0 <= arrayscore[0][i] <= 1, 'Output value outside acceptable range'


def test_score_max():
    """Evaluates the functionality of the score_max function"""
    # Initialize the test arguments
    k = 2
    row_i = [0, 3]
    row_j = [2, 1]
    rowcat = row_i + row_j
    arraya = np.array([[0, 1], [2, 1], [0, 3]])

    arraycat = np.concatenate((arraya[0], arraya[1]))

    # Run Function for lists
    try:

        maxscores = peakidentify.score_max(row_i, row_j, -1)

    except ValueError:

        print("An invalid k value was passed to the function, "
              "and was handled correctly.")

     # Run Function for arrays
    try:

        arrmaxscores = peakidentify.score_max(arraya[0], arraya[1], -1)

    except ValueError:

        print("An invalid k value was passed to the function, "
              "and was handled correctly.")

    # Run good examples
    maxscores = peakidentify.score_max(row_i, row_j, k)
    arrmaxscores = peakidentify.score_max(arraya[0], arraya[1], k)

    # make assertions
    assert len(arrmaxscores[0]) == len(arraycat), """Output list length different
    than concatenated lists length"""
    for i, _ in enumerate(rowcat):
        assert 0 <= arrmaxscores[0][i] <= 2, 'Output value outside acceptable range'
        assert 0 <= maxscores[0][i] <= 2, 'Output value outside acceptable range'
    for i, _ in enumerate(maxscores, 1):
        assert maxscores[0][i-1] >= maxscores[0][-1], 'Output values are less than the max value'


def test_score_sort():
    """Evaluates the functionality of the score_sort function"""
    # Initialize the test arguments
    row_i = [0, 1]
    row_j = [2, 1]
    rowcat = row_i + row_j
    arraya = np.array([[0, 1], [2, 1], [0, 3]])
    k = 2
    arraycat = np.concatenate((arraya[0], arraya[1]))
    # Run Previous Function to get max score normalization
    maxscores = peakidentify.score_max(row_i, row_j, k)

    # Run Function for lists

    try:
        sortedscores = peakidentify.score_sort(row_i, row_j, max(maxscores[0]))

    except TypeError:

        print("An invalid maxscores from score_max was passed to the function, "
              "and was handled correctly.")

    # Run Function for arrays

    try:

        arrsortedscores = peakidentify.score_sort(arraya[0], arraya[1], max(maxscores[0]))

    except TypeError:

        print("An invalid maxscores from score_max was passed to the function, "
              "and was handled correctly.")

    # Run good examples
    sortedscores = peakidentify.score_sort(row_i, row_j, int(max(maxscores[0])))
    arrsortedscores = peakidentify.score_sort(arraya[0],
                                              arraya[1],
                                              int(max(maxscores[0])))
    # make assertions
    assert len(arraycat) == len(arrsortedscores[0][0]), """Output list length
    different than concatenated lists length"""
    assert len(rowcat) == len(sortedscores[0][0]), """Output list length
    different than concatenated lists length"""
    for i, _ in enumerate(sortedscores):
        assert sortedscores[0][0][i] <= sortedscores[0][0][i+1], """Output values
        is sorted from smallest to largest"""
        assert arrsortedscores[0][0][i] <= arrsortedscores[0][0][i+1], """Output
        values is sorted from smallest to largest"""
