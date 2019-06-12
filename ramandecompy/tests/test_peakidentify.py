"""
Module used to unit test the functionality and outputs of the peakidentify.py module
"""
# IMPORTING MODULES
import os
import h5py
import numpy as np
from ramandecompy import peakidentify
from ramandecompy import dataprep


def test_peak_assignment():
    """This function tests the operation of the peak_assignment
    function in peakidentify.py"""
    #First, generate a testing dataset.
    dataprep.new_hdf5('peakidentify_calibration_test')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5',
                             'ramandecompy/tests/test_files/Hydrogen_Baseline_Calibration.xlsx',
                             label='Hydrogen')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5',
                             'ramandecompy/tests/test_files/CarbonMonoxide_Baseline_Calibration.xlsx',
                             label='CarbonMonoxide')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5',
                             'ramandecompy/tests/test_files/CO2_100wt%.csv',
                             label='CO2')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5','ramandecompy/tests/test_files/water.xlsx',label='H2O')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5','ramandecompy/tests/test_files/sapphire.xlsx',label='sapphire')
    dataprep.new_hdf5('peakidentify_experiment_test')
    dataprep.add_experiment('peakidentify_experiment_test.hdf5',
                            'ramandecompy/tests/test_files/FA_3.6wt%_300C_25s.csv')
    hdf5_calfilename = 'peakidentify_calibration_test.hdf5'
    hdf5_expfilename = 'peakidentify_experiment_test.hdf5'
    key = '300C/25s'
    calhdf5 = h5py.File(hdf5_calfilename, 'r+')
    exphdf5 = h5py.File(hdf5_expfilename, 'r+')
    unknown_x = list(exphdf5['{}/wavenumber'.format(key)])
    unknown_y = list(exphdf5['{}/counts'.format(key)])
    unknown_x = np.asarray(unknown_x)
    unknown_y = np.asarray(unknown_y)
    precision = 50
    #Various try statements to make sure that bad inputs are handled correctly.

    try:
        peakidentify.peak_assignment(hdf5_expfilename,
                                     key, hdf5_calfilename,
                                     precision, False, False)

    except TypeError:
        print("An invalid known_compound_list was passed to the function, "
              "and it was handled well with a TypeError.")

    try:
        peakidentify.peak_assignment(hdf5_expfilename,
                                     key, hdf5_calfilename,
                                     'precision', False, False)

    except TypeError:
        print("An invalid precision value was passed to the function, and "
              "it was handled well with a TypeError.")

    try:
        peakidentify.peak_assignment(hdf5_expfilename,
                                     key, hdf5_calfilename,
                                     precision, 'False', False)

    except TypeError:
        print("An invalid export label value was passed to the function, and it "
              "was handled well with a TypeError.")

    try:
        peakidentify.peak_assignment(hdf5_expfilename,
                                     key, hdf5_calfilename,
                                     precision, False, 'False')

    except TypeError:
        print("An invalid plot value was passed to the function, and it "
              "was handled well with a TypeError.")

    exphdf5.close()
    calhdf5.close()
    os.remove('peakidentify_calibration_test.hdf5')
    os.remove('peakidentify_experiment_test.hdf5')

def test_compare_unknown_to_known():
    """This function tests the operation of the compare_unknown_to_known
    function in peakidentify.py"""
    #First, generate a testing dataset.
    dataprep.new_hdf5('peakidentify_calibration_test')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5',
                             'ramandecompy/tests/test_files/Hydrogen_Baseline_Calibration.xlsx',
                             label='Hydrogen')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5',
                             'ramandecompy/tests/test_files/CarbonMonoxide_Baseline_Calibration.xlsx',
                             label='CarbonMonoxide')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5',
                             'ramandecompy/tests/test_files/CO2_100wt%.csv',
                             label='CO2')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5','ramandecompy/tests/test_files/water.xlsx',label='H2O')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5','ramandecompy/tests/test_files/sapphire.xlsx',label='sapphire')
    dataprep.new_hdf5('peakidentify_experiment_test')
    dataprep.add_experiment('peakidentify_experiment_test.hdf5',
                            'ramandecompy/tests/test_files/FA_3.6wt%_300C_25s.csv')
    hdf5_calfilename = 'peakidentify_calibration_test.hdf5'
    hdf5_expfilename = 'peakidentify_experiment_test.hdf5'
    key = '300C/25s'
    calhdf5 = h5py.File(hdf5_calfilename, 'r+')
    exphdf5 = h5py.File(hdf5_expfilename, 'r+')
    unknown_x = list(exphdf5['{}/wavenumber'.format(key)])
    unknown_y = list(exphdf5['{}/counts'.format(key)])
    unknown_x = np.asarray(unknown_x)
    unknown_y = np.asarray(unknown_y)
    known_compound_list = list(calhdf5.keys())
    precision = 50
    known_peaks = []
    known_peaks_list = []
    unknown_peaks = []
    for i, _ in enumerate(list(exphdf5['{}'.format(key)])[:-3]):
        if i < 9:
            unknown_peaks.append(list(exphdf5['{}/Peak_0{}'.format(key, i+1)])[0][2])
        else:
            unknown_peaks.append(list(exphdf5['{}/Peak_{}'.format(key, i+1)])[0][2])
    for i, _ in enumerate(known_compound_list):
        for _, peak in enumerate(list(calhdf5[known_compound_list[i]])[:-3]):
            known_peaks_list.append(list(calhdf5['{}/{}'.format(known_compound_list[i],
                                                                peak)])[0][2])
        known_peaks.append(known_peaks_list[i])

    try:
        peakidentify.compare_unknown_to_known(1, known_peaks, precision)
    except TypeError:
        print("An invalid unknown_peaks value was passed to the function, "
              "and was handled correctly.")

    try:
        peakidentify.compare_unknown_to_known(unknown_peaks, 'known_peaks', precision)
    except TypeError:
        print("An invalid known_peaks value was passed to the function, "
              "and was handled correctly.")

    try:
        peakidentify.compare_unknown_to_known(unknown_peaks, known_peaks, 'precision')
    except TypeError:
        print("An invalid precision value was passed to the function, and "
              "was handled correctly.")

    #After testing for resilience to unexpected inputs, now ensure
    #outputs are performing correctly

    #First, make sure function is returning the list.
    assert isinstance(peakidentify.compare_unknown_to_known(
        unknown_peaks, known_peaks, precision), np.ndarray), ("""Function
        is not returning list""")

    #Compare one set of peaks to itself. The full association matrix
    #should have all values = 1.
    self_comp = np.mean(peakidentify.compare_unknown_to_known(known_peaks,
                                                              known_peaks,
                                                              precision))
    assert self_comp == 1, ("Peak Assignment Error. Comparison of compound "
                            "against itself should find all peaks.")

    dif_comp = np.mean(peakidentify.compare_unknown_to_known([1, 3, 6],
                                                             [1000, 2000, 5000],
                                                             precision))
    assert dif_comp == 0, ("Peak Assignment Error. Passed values should "
                           "have no matching assignments.")

    exphdf5.close()
    calhdf5.close()
    os.remove('peakidentify_calibration_test.hdf5')
    os.remove('peakidentify_experiment_test.hdf5')

def test_peak_position_comparisons():
    """This function tests the operation of the peak_position_comparisons
    function in peakidentify. Said function returns a list of strings that
    contain text assignments of each peak in the unknown spectrum."""

    #First, generate a testing dataset.
    dataprep.new_hdf5('peakidentify_calibration_test')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5',
                             'ramandecompy/tests/test_files/Hydrogen_Baseline_Calibration.xlsx',
                             label='Hydrogen')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5',
                             'ramandecompy/tests/test_files/CarbonMonoxide_Baseline_Calibration.xlsx',
                             label='CarbonMonoxide')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5',
                             'ramandecompy/tests/test_files/CO2_100wt%.csv',
                             label='CO2')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5','ramandecompy/tests/test_files/water.xlsx',label='H2O')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5','ramandecompy/tests/test_files/sapphire.xlsx',label='sapphire')
    dataprep.new_hdf5('peakidentify_experiment_test')
    dataprep.add_experiment('peakidentify_experiment_test.hdf5',
                            'ramandecompy/tests/test_files/FA_3.6wt%_300C_25s.csv')
    hdf5_calfilename = 'peakidentify_calibration_test.hdf5'
    hdf5_expfilename = 'peakidentify_experiment_test.hdf5'
    key = '300C/25s'
    calhdf5 = h5py.File(hdf5_calfilename, 'r+')
    exphdf5 = h5py.File(hdf5_expfilename, 'r+')
    unknown_x = list(exphdf5['{}/wavenumber'.format(key)])
    unknown_y = list(exphdf5['{}/counts'.format(key)])
    unknown_x = np.asarray(unknown_x)
    unknown_y = np.asarray(unknown_y)
    known_compound_list = list(calhdf5.keys())
    precision = 50
    known_peaks = []
    known_peaks_list = []
    association_matrix = []
    unknown_peaks = []
    for i, _ in enumerate(list(exphdf5['{}'.format(key)])[:-3]):
        if i < 9:
            unknown_peaks.append(list(exphdf5['{}/Peak_0{}'.format(key, i+1)])[0][2])
        else:
            unknown_peaks.append(list(exphdf5['{}/Peak_{}'.format(key, i+1)])[0][2])
    known_peaks = []
    known_peaks_list = []
    num_peaks_list = []
    association_matrix = []
    split__index_list = []
    for i, _ in enumerate(known_compound_list):
        num_peaks_list.append(len(list(calhdf5[known_compound_list[i]])[:-3]))
        split__index_list.append(sum(num_peaks_list))
        for j, peak in enumerate(list(calhdf5[known_compound_list[i]])[:-3]):
            # Need to separate known peaks to make a list of two separate lists
            # to perform custom list split using list comprehension + zip()
            # and split_index_list
            known_peaks_list.append(list(calhdf5['{}/{}'.format(known_compound_list[i],
                                                                peak)])[0][2])
            result = [known_peaks_list[i : j] for i, j in zip([0] + split__index_list,
                                                              split__index_list +
                                                              [None])]
        known_peaks.append(result)
        association_matrix.append(peakidentify.compare_unknown_to_known(
            unknown_peaks, known_peaks[i][i], precision))

    #Then, test error handling of bad inputs for the function.
    try:
        peakidentify.peak_position_comparisons(1, known_peaks,
                                               association_matrix,
                                               hdf5_calfilename)

    except TypeError:
        print("An invalid unknown_peaks value was passed to the function, "
              "and was handled correctly.")

    try:
        peakidentify.peak_position_comparisons(unknown_peaks,
                                               'known_peaks',
                                               association_matrix,
                                               hdf5_calfilename)

    except TypeError:
        print("An invalid known_peaks value was passed to the function, "
              "and was handled correctly.")

    try:
        peakidentify.peak_position_comparisons(unknown_peaks,
                                               known_peaks,
                                               'association_matrix',
                                               hdf5_calfilename)

    except TypeError:
        print("An invalid association_matrix value was passed to the function,"
              "and was handled correctly.")

    #Check to make sure the function is returning a list.
    assert isinstance(peakidentify.peak_position_comparisons(
        unknown_peaks, known_peaks,
        association_matrix, hdf5_calfilename), list), """The function is
        not returning a list."""

    #Test a call that says that no peaks have associations
    association_matrix_0 = []

    association_matrix_0.append(peakidentify.compare_unknown_to_known(
        known_peaks[0][0],
        known_peaks[1][0],
        precision))

    zero_output = peakidentify.peak_position_comparisons(known_peaks[0][0],
                                                         [known_peaks[1][0]],
                                                         association_matrix_0,
                                                         hdf5_calfilename)[0]

    assert zero_output[0] == 'Acetaldehyde', """The function is not properly
    handling unassigned peaks."""

    #Test the function to make sure that it has the right functionality
    association_matrix = []
    #Generate a matrix with all associations equal to 1
    association_matrix.append(peakidentify.compare_unknown_to_known(
        known_peaks[0][0],
        known_peaks[0][0],
        precision))

    #change the middle index to 0
    association_matrix[0][1] = 0
    test_peak_labels = peakidentify.peak_position_comparisons(known_peaks[0][0],
                                                              [known_peaks[0][0]],
                                                              association_matrix,
                                                              hdf5_calfilename)
    print(test_peak_labels[0][0])
    print(test_peak_labels[1][0])

    assert test_peak_labels[0][0] == 'Acetaldehyde', """The funciton is
    not correctly assigning peaks when association matrix = 1"""
    assert test_peak_labels[1][0] == 'Unassigned', """The function is
    not correctly handling a lack of peak assignments"""

    exphdf5.close()
    calhdf5.close()
    os.remove('peakidentify_calibration_test.hdf5')
    os.remove('peakidentify_experiment_test.hdf5')

def test_percentage_of_peaks_found():
    """This function tests the operation of the
    percentage_of_peaks_found function in peakidentify.py"""

    #First, generate a testing dataset.
    dataprep.new_hdf5('peakidentify_calibration_test')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5',
                             'ramandecompy/tests/test_files/Hydrogen_Baseline_Calibration.xlsx',
                             label='Hydrogen')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5',
                             'ramandecompy/tests/test_files/CarbonMonoxide_Baseline_Calibration.xlsx',
                             label='CarbonMonoxide')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5',
                             'ramandecompy/tests/test_files/CO2_100wt%.csv',
                             label='CO2')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5','ramandecompy/tests/test_files/water.xlsx',label='H2O')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5','ramandecompy/tests/test_files/sapphire.xlsx',label='sapphire')
    dataprep.new_hdf5('peakidentify_experiment_test')
    dataprep.add_experiment('peakidentify_experiment_test.hdf5',
                            'ramandecompy/tests/test_files/FA_3.6wt%_300C_25s.csv')
    hdf5_calfilename = 'peakidentify_calibration_test.hdf5'
    hdf5_expfilename = 'peakidentify_experiment_test.hdf5'
    key = '300C/25s'
    calhdf5 = h5py.File(hdf5_calfilename, 'r+')
    exphdf5 = h5py.File(hdf5_expfilename, 'r+')
    unknown_x = list(exphdf5['{}/wavenumber'.format(key)])
    unknown_y = list(exphdf5['{}/counts'.format(key)])
    unknown_x = np.asarray(unknown_x)
    unknown_y = np.asarray(unknown_y)
    known_compound_list = list(calhdf5.keys())
    precision = 50

    unknown_peaks = []
    for i, _ in enumerate(list(exphdf5['{}'.format(key)])[:-3]):
        if i < 9:
            unknown_peaks.append(list(exphdf5['{}/Peak_0{}'.format(key, i+1)])[0][2])
        else:
            unknown_peaks.append(list(exphdf5['{}/Peak_{}'.format(key, i+1)])[0][2])
    known_peaks = []
    known_peaks_list = []
    num_peaks_list = []
    association_matrix = []
    split__index_list = []
    for i, _ in enumerate(known_compound_list):
        num_peaks_list.append(len(list(calhdf5[known_compound_list[i]])[:-3]))
        split__index_list.append(sum(num_peaks_list))
        for j, peak in enumerate(list(calhdf5[known_compound_list[i]])[:-3]):
            # Need to separate known peaks to make a list of two separate lists
            # to perform custom list split using list comprehension + zip()
            # and split_index_list
            known_peaks_list.append(list(calhdf5['{}/{}'.format(known_compound_list[i],
                                                                peak)])[0][2])
            result = [known_peaks_list[i : j] for i, j in zip([0] + split__index_list,
                                                              split__index_list +
                                                              [None])]
        known_peaks.append(result)
        association_matrix.append(peakidentify.compare_unknown_to_known(
            unknown_peaks, known_peaks[i][i], precision))

    #Test for input error handling.

    try:
        peakidentify.percentage_of_peaks_found([[0], [1], [2], [3], [4],
                                                [5],[6],[7],[8]],
                                                association_matrix,
                                                hdf5_calfilename)
    except TypeError:
        print("""The function correctly handled the error when a list of ints
        was input instead of the known_peaks list""")

    try:
        peakidentify.percentage_of_peaks_found(1, association_matrix,
                                               hdf5_calfilename)

    except TypeError:
        print("""The function correctly handled the error when an int
        was input instead of the known_peaks list""")

    try:
        peakidentify.percentage_of_peaks_found(known_peaks, 1,
                                               known_compound_list,
                                               hdf5_calfilename)

    except TypeError:
        print("""The function correctly handled the error when an int
        was input instead of the association matrix""")


    #Test to make sure function returns a dictionary.
    assert isinstance(peakidentify.percentage_of_peaks_found(
        known_peaks,
        association_matrix,
        hdf5_calfilename), dict), """The function is not
        returning a dictionary."""

#     #Test for function output.
    co2_peaks = []
    key = 'CO2'
    for _, peak in enumerate(list(calhdf5[key])[:-3]):
        co2_peaks.append(list(calhdf5['{}/{}'.format(key, peak)])[0][2])
    print(co2_peaks)
    co2_dict_0 = peakidentify.percentage_of_peaks_found([co2_peaks, [0], [0], [0],
                                                         [0],[0], [0], [0]],
                                                        [[0, 0], [0], [0],[0],
                                                         [0],[0], [0], [0]],
                                                        hdf5_calfilename)
    assert co2_dict_0[key] == 0, """The function is not correctly
    calculating percentages when no peaks are found"""

    co2_dict_1 = peakidentify.percentage_of_peaks_found([co2_peaks, [1], [1],[1],
                                                         [1],[1],[1], [1]],
                                                        [[1, 1], [1], [1],[1],
                                                         [1],[1],[1], [1]],
                                                        hdf5_calfilename)
    assert co2_dict_1[key] == 100, """The function is not correctly
    calculating percentages when all peaks are found"""

    exphdf5.close()
    calhdf5.close()
    os.remove('peakidentify_calibration_test.hdf5')
    os.remove('peakidentify_experiment_test.hdf5')

def test_plotting_peak_assignments():
    """This function tests the operation of the peak_assignment
    function in peakidentify.py"""
    #First, generate a testing dataset.
    dataprep.new_hdf5('peakidentify_calibration_test')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5',
                             'ramandecompy/tests/test_files/Hydrogen_Baseline_Calibration.xlsx',
                             label='Hydrogen')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5',
                             'ramandecompy/tests/test_files/CarbonMonoxide_Baseline_Calibration.xlsx',
                             label='CarbonMonoxide')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5',
                             'ramandecompy/tests/test_files/CO2_100wt%.csv',
                             label='CO2')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5','ramandecompy/tests/test_files/water.xlsx',label='H2O')
    dataprep.add_calibration('peakidentify_calibration_test.hdf5','ramandecompy/tests/test_files/sapphire.xlsx',label='sapphire')
    dataprep.new_hdf5('peakidentify_experiment_test')
    dataprep.add_experiment('peakidentify_experiment_test.hdf5',
                            'ramandecompy/tests/test_files/FA_3.6wt%_300C_25s.csv')
    hdf5_calfilename = 'peakidentify_calibration_test.hdf5'
    hdf5_expfilename = 'peakidentify_experiment_test.hdf5'
    key = '300C/25s'
    calhdf5 = h5py.File(hdf5_calfilename, 'r+')
    exphdf5 = h5py.File(hdf5_expfilename, 'r+')
    unknown_x = list(exphdf5['{}/wavenumber'.format(key)])
    unknown_y = list(exphdf5['{}/counts'.format(key)])
    unknown_x = np.asarray(unknown_x)
    unknown_y = np.asarray(unknown_y)
    known_compound_list = list(calhdf5.keys())
    precision = 50

    unknown_peaks = []
    for i, _ in enumerate(list(exphdf5['{}'.format(key)])[:-3]):
        if i < 9:
            unknown_peaks.append(list(exphdf5['{}/Peak_0{}'.format(key, i+1)])[0][2])
        else:
            unknown_peaks.append(list(exphdf5['{}/Peak_{}'.format(key, i+1)])[0][2])
    known_peaks = []
    known_peaks_list = []
    num_peaks_list = []
    association_matrix = []
    split__index_list = []
    for i, _ in enumerate(known_compound_list):
        num_peaks_list.append(len(list(calhdf5[known_compound_list[i]])[:-3]))
        split__index_list.append(sum(num_peaks_list))
        for j, peak in enumerate(list(calhdf5[known_compound_list[i]])[:-3]):
            # Need to separate known peaks to make a list of two separate lists
            # to perform custom list split using list comprehension + zip()
            # and split_index_list
            known_peaks_list.append(list(calhdf5['{}/{}'.format(known_compound_list[i],
                                                                peak)])[0][2])
            result = [known_peaks_list[i : j] for i, j in zip([0] + split__index_list,
                                                              split__index_list +
                                                              [None])]
        known_peaks.append(result)
        association_matrix.append(peakidentify.compare_unknown_to_known(
            unknown_peaks, known_peaks[i][i], precision))
    #Ok, so that generates a full association matrix that contains everything
    #we need to assign peaks.
    #Now, let's go through and actually assign text to peaks.
    unknown_peak_assignments = peakidentify.peak_position_comparisons(
        unknown_peaks, known_peaks, association_matrix, hdf5_calfilename)
    peak_labels = []
    for i, _ in enumerate(unknown_peak_assignments):
        peak_labels.append(str(unknown_peak_assignments[i]))
    #Test for input error handling.
    try:
        peakidentify.plotting_peak_assignments(1, unknown_y, unknown_peaks,
                                               unknown_peak_assignments,
                                               hdf5_expfilename,
                                               hdf5_calfilename,
                                               key, peak_labels)
    except TypeError:
        print("""The function correctly handled the error
        when an int was input instead of the unknown_x list""")

    try:
        peakidentify.plotting_peak_assignments(unknown_x, 3, unknown_peaks,
                                               unknown_peak_assignments,
                                               hdf5_expfilename,
                                               hdf5_calfilename,
                                               key, peak_labels)
    except TypeError:
        print("""The function correctly handled the error when an int
        was input instead of the unknown_y list""")

    try:
        peakidentify.plotting_peak_assignments(unknown_x,
                                               unknown_y,
                                               'unknown_peaks',
                                               unknown_peak_assignments,
                                               hdf5_expfilename,
                                               hdf5_calfilename,
                                               key, peak_labels)
    except TypeError:
        print("""The function correctly handled the error when a string
        was input instead of the unknown_peaks list""")

    try:
        peakidentify.plotting_peak_assignments(unknown_x,
                                               unknown_y,
                                               unknown_peaks,
                                               3,
                                               hdf5_expfilename,
                                               hdf5_calfilename,
                                               key, peak_labels)
    except TypeError:
        print("""The function correctly handled the error when an int
        was input instead of the unknown_peak_assignments""")

    try:
        peakidentify.plotting_peak_assignments(unknown_x,
                                               unknown_y,
                                               unknown_peaks,
                                               ['WATER', 23, 'CO'],
                                               hdf5_expfilename,
                                               hdf5_calfilename,
                                               key, peak_labels)

    except TypeError:
        print("""The function correctly handled the case when an int
        was passed in the unknown_peak_assignment list""")

    try:
        peakidentify.plotting_peak_assignments(unknown_x,
                                               unknown_y,
                                               unknown_peaks,
                                               ['H', 23, 'CO2'],
                                               hdf5_expfilename,
                                               hdf5_calfilename,
                                               key, peak_labels)

    except TypeError:
        print("""The function correctly handled the case when an int
        was passed in the unknown_peak_assignment list""")

    try:
        peakidentify.plotting_peak_assignments(unknown_x,
                                               unknown_y,
                                               unknown_peaks,
                                               unknown_peak_assignments,
                                               3,
                                               hdf5_calfilename,
                                               key, peak_labels)

    except TypeError:
        print("""The function correctly handled the case when an int
        was passed in the hdf5_filename""")

    try:
        peakidentify.plotting_peak_assignments(unknown_x,
                                               unknown_y,
                                               unknown_peaks,
                                               unknown_peak_assignments,
                                               hdf5_expfilename,
                                               3,
                                               key, peak_labels)

    except TypeError:
        print("""The function correctly handled the case when an int
        was passed in the hdf5_calfilename""")

    exphdf5.close()
    calhdf5.close()
    os.remove('peakidentify_calibration_test.hdf5')
    os.remove('peakidentify_experiment_test.hdf5')

def test_add_label():
    """
    Function that adds a label to a peak dataset in the hdf5 file
    """
    dataprep.new_hdf5('peakidentify_add_label_test')
    dataprep.add_experiment('peakidentify_add_label_test.hdf5',
                            'ramandecompy/tests/test_files/FA_3.6wt%_300C_25s.csv')
    hdf5_filename = 'peakidentify_add_label_test.hdf5'
    key = '300C/25s'
    peak = 'Peak_01'
    label = '[Hydrogen]'
    # open hdf5 file as read/write
    hdf5 = h5py.File(hdf5_filename, 'r+')
    try:
        peakidentify.add_label('hdf5_filename', key, peak, label)
    except TypeError:
        print("An invalid hdf5_filename was passed to the function, "
              "and it was handled well with a TypeError.")
    try:
        peakidentify.add_label(hdf5_filename, 'key', peak, label)
    except TypeError:
        print("An invalid temp was passed to the function, "
              "and it was handled well with a TypeError.")
    hdf5.close()
    del hdf5
    os.remove('peakidentify_add_label_test.hdf5')

def test_peak_1d_score():
    """Evaluates the functionality of the peak_1D_score function"""
    # Initialize the test arguments
    row_i = [0, 1]
    row_j = [2, 1]
    rowcat = row_i + row_j
    arraya = np.array([[0, 1], [2, 1], [0, 3]])
    arraycat = np.concatenate((arraya[0], arraya[2]))
    precision = 50

    # Run Bad Function for lists
    try:
        testscore = peakidentify.peak_1d_score(row_i, row_j, -1, precision)
    except ValueError:
        print("An invalid scoremax value was passed to the function, "
              "and was handled correctly.")

    # Run Bad Function for arrays
    try:
        arrayscore = peakidentify.peak_1d_score(arraya[0], arraya[2], -1, precision)

    except ValueError:
        print("An invalid scoremax value was passed to the function, "
              "and was handled correctly.")

    # Running a good example
    testscore = peakidentify.peak_1d_score(row_i, row_j, 1., precision)
    arrayscore = peakidentify.peak_1d_score(arraya[0], arraya[2], 1, precision)

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
    precision = 50

    arraycat = np.concatenate((arraya[0], arraya[1]))

    # Run Function for lists
    try:

        maxscores = peakidentify.score_max(row_i, row_j, -1, precision)

    except ValueError:

        print("An invalid k value was passed to the function, "
              "and was handled correctly.")

     # Run Function for arrays
    try:

        arrmaxscores = peakidentify.score_max(arraya[0], arraya[1], -1, precision)

    except ValueError:

        print("An invalid k value was passed to the function, "
              "and was handled correctly.")

    # Run good examples
    maxscores = peakidentify.score_max(row_i, row_j, k, precision)
    arrmaxscores = peakidentify.score_max(arraya[0], arraya[1], k, precision)

    # make assertions
    assert len(arrmaxscores[0]) == len(arraycat), """Output list length different
    than concatenated lists length"""
    for i, _ in enumerate(rowcat):
        assert 0 <= arrmaxscores[0][i] <= 2, 'Output value outside acceptable range'
        assert 0 <= maxscores[0][i] <= 2, 'Output value outside acceptable range'
    for i, _ in enumerate(maxscores, 1):
        assert maxscores[0][i-1] >= maxscores[0][-1], """Output values are
        less than the max value"""


def test_score_sort():
    """Evaluates the functionality of the score_sort function"""
    # Initialize the test arguments
    row_i = [0, 1]
    row_j = [2, 1]
    rowcat = row_i + row_j
    arraya = np.array([[0, 1], [2, 1], [0, 3]])
    k = 2
    precision = 50
    arraycat = np.concatenate((arraya[0], arraya[1]))
    # Run Previous Function to get max score normalization
    maxscores = peakidentify.score_max(row_i, row_j, k, precision)

    # Run Function for lists

    try:
        sortedscores = peakidentify.score_sort(row_i, row_j, max(maxscores[0]), precision)

    except TypeError:

        print("An invalid maxscores from score_max was passed to the function, "
              "and was handled correctly.")

    # Run Function for arrays

    try:

        arrsortedscores = peakidentify.score_sort(arraya[0], arraya[1],
                                                  max(maxscores[0]),
                                                  precision)

    except TypeError:

        print("An invalid maxscores from score_max was passed to the function, "
              "and was handled correctly.")

    # Run good examples
    sortedscores = peakidentify.score_sort(row_i, row_j,
                                           int(max(maxscores[0])),
                                           precision)
    arrsortedscores = peakidentify.score_sort(arraya[0], arraya[1],
                                              int(max(maxscores[0])),
                                              precision)
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