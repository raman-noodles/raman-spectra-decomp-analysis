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


# Will probably need to create an additional function

def peak_assignment(unknownhdf5_filename, key, knownhdf5_filename,
                    precision=50, exportlabelinput=True, plot=True):
    """This function is a wrapper function from which all classification
    of peaks occurs."""

    #Handling errors in inputs.
    if not isinstance(knownhdf5_filename, str):
        raise TypeError("""Passed value of `knownhdf5_filename` is not a string!
        Instead, it is: """ + str(type(knownhdf5_filename)))
    if not knownhdf5_filename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError("""`knownhdf5_filename` is not type = .hdf5!
        Instead, it is: """ + knownhdf5_filename.split('/')[-1].split('.')[-1])
    if not isinstance(unknownhdf5_filename, str):
        raise TypeError("""Passed value of `unknownhdf5_filename` is not a string!
        Instead, it is: """ + str(type(unknownhdf5_filename)))
    if not unknownhdf5_filename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError("""`unknownhdf5_filename` is not type = .hdf5!
        Instead, it is: """ + unknownhdf5_filename.split('/')[-1].split('.')[-1])
    if not isinstance(key, str):
        raise TypeError("""Passed value of `key` is not a int!
        Instead, it is: """ + str(type(key)))
    if not isinstance(precision, (float, int)):
        raise TypeError("""Passed value of `precision` is not a float or int!
        Instead, it is: """ + str(type(precision)))
    if not isinstance(exportlabelinput, bool):
        raise TypeError("""Passed value of `exportlabelinput` is not a Boolean!
        Instead, it is: """ + str(type(exportlabelinput)))
    if not isinstance(plot, bool):
        raise TypeError("""Passed value of `plot` is not a Boolean!
        Instead, it is: """ + str(type(plot)))
    # open .hdf5
    unhdf5 = h5py.File(unknownhdf5_filename, 'r+')
    knhdf5 = h5py.File(knownhdf5_filename, 'r+')

    #Extract keys from files
    known_compound_list = list(knhdf5.keys())

    if not isinstance(known_compound_list, list):
        raise TypeError("""Passed value of `known_compound_list` is not a list!
        Instead, it is: """ + str(type(known_compound_list)))
    #Now we need to check the elements within the known_compound_list
    #to make sure they are correct.
    for i, _ in enumerate(known_compound_list):
        if not isinstance(known_compound_list[i], str):
            raise TypeError("""Passed value within `known_compound_list` is not
            a string! Instead, it is: """ + str(type(known_compound_list[i])))

    # extract spectra data
    unknown_x = list(unhdf5['{}/wavenumber'.format(key)])
    unknown_y = list(unhdf5['{}/counts'.format(key)])
    unknown_x = np.asarray(unknown_x)
    unknown_y = np.asarray(unknown_y)
    #Lets identify the peaks in the unknown spectrum.
    unknown_peaks = []
    for i, peak in enumerate(list(unhdf5['{}'.format(key)])[:-3]):
        try:
            if i < 9:
                unknown_peaks.append(list(unhdf5['{}/Peak_0{}'.format(key,
                                                                      i+1)])[0][2])
            else:
                unknown_peaks.append(list(unhdf5['{}/Peak_{}'.format(key,
                                                                     i+1)])[0][2])
        except Exception as e:
            #Normal peakassignment
            print("""Function did not receive normal peak.
            The function continued to look for an adjusted peak.""")
            if i < 9:
                print(peak)
                unknown_peaks.append(list(unhdf5['{}/Peak_0{}*'.format(key,
                                                                       i+1)])[0][2])
            else:
                unknown_peaks.append(list(unhdf5['{}/Peak_{}*'.format(key,
                                                                      i+1)])[0][2])
            print('Peak_{}*'.format(i+1))
        else:
            pass


    #OK, next identify all of the peaks present in the known compound set.
    #For efficiency, we'll also compare them against the unknown in the same for loop.
    known_peaks = []
    known_peaks_list = []
    num_peaks_list = []
    assignment_matrix = []
    split__index_list = []
    for i, _ in enumerate(known_compound_list):
        print("The peaks that we found for "
              + str(known_compound_list[i]) + " are: ")
        num_peaks_list.append(len(list(knhdf5[known_compound_list[i]])[:-3]))
        split__index_list.append(sum(num_peaks_list))
        for j, peak in enumerate(list(knhdf5[known_compound_list[i]])[:-3]):
            print(list(knhdf5['{}/{}'.format(known_compound_list[i], peak)])[0][2])
            # Need to separate known peaks to make a list of two separate lists
            # to perform custom list split using list comprehension + zip() and split_index_list
            known_peaks_list.append(list(knhdf5['{}/{}'.format(known_compound_list[i],
                                                               peak)])[0][2])
            result = [known_peaks_list[i : j] for i, j in zip([0] + split__index_list,
                                                              split__index_list +
                                                              [None])]
        known_peaks.append(result)
        assignment_matrix.append(compare_unknown_to_known(
            unknown_peaks, known_peaks[i][i], precision))
    #Ok, so that generates a full association matrix that contains everything
    #we need to assign peaks.
    #Now, let's go through and actually assign text to peaks.
    unknown_peak_assignments = peak_position_comparisons(unknown_peaks,
                                                         known_peaks,
                                                         assignment_matrix,
                                                         knownhdf5_filename)
    print(unknown_peak_assignments)
    peak_labels = []
    for i, _ in enumerate(unknown_peak_assignments):
        peak_labels.append(str(unknown_peak_assignments[i]))
    frames = []
    for j, peak in enumerate(list(unhdf5['{}'.format(key)])[:-3]):
        frames.append(pd.DataFrame(add_label(unknownhdf5_filename,
                                             key, peak, peak_labels[j])))
         
    df = pd.concat(frames,axis=1, join='outer', join_axes=None, ignore_index=False,
              keys=None, levels=None, names=None, verify_integrity=False,
              copy=True,sort=True)
    df =df.T
    if plot:
        plotting_peak_assignments(unknown_x,
                                  unknown_y,
                                  unknown_peaks,
                                  unknown_peak_assignments,
                                  unknownhdf5_filename,
                                  knownhdf5_filename,
                                  key,
                                  peak_labels,
                                  exportlabelinput)

    percentages = percentage_of_peaks_found(known_peaks[len(known_compound_list)-1],
                                            assignment_matrix,
                                            knownhdf5_filename)
    print(percentages)
    knhdf5.close()
    unhdf5.close()
    return df

def compare_unknown_to_known(unknown_peaks, known_peaks, precision):
    """This function takes in peak positions for the spectrum to be
    analyzed and a single known compound and determines if the peaks
    found in the known compound are present in the unknown spectrum."""

    #Handling errors in inputs.
    if not isinstance(unknown_peaks, list):
        raise TypeError("""Passed value of `combined_peaks` is not a list!
        Instead, it is: """ + str(type(unknown_peaks)))

    if not isinstance(known_peaks, list):
        raise TypeError("""Passed value of `known_peaks` is not a list!
        Instead, it is: """ + str(type(known_peaks)))
# Need to check values within known_peaks

    if not isinstance(precision, (float, int)):
        raise TypeError("""Passed value of `precision` is not a float or int!
        Instead, it is: """ + str(type(precision)))

    assignment_matrix = np.zeros(len(unknown_peaks))
    peaks_found = 0
    for i, _ in enumerate(unknown_peaks):
        for j, _ in enumerate(known_peaks):
            # instead of If, call peak_1D_score
            if math.isclose(unknown_peaks[i], known_peaks[j],
                            abs_tol=precision, rel_tol=1e-9):
                # Instead of using a 1, just input the score
                # from the score calculator.
                # Bigger is better.
                # Storing only the second component in the list.
                assignment_matrix[i] = 1
                peaks_found += 1
                continue
            else:
                pass
        if peaks_found == len(known_peaks):
            continue
        else:
            pass
    print(assignment_matrix)

    return assignment_matrix

def peak_position_comparisons(unknown_peaks, known_compound_peaks,
                              association_matrix,
                              knownhdf5_filename):
    """This function takes in an association matrix and turns the numbers
    given by said matrix into a text label."""

    #Handling errors in inputs.
    if not isinstance(unknown_peaks, list):
        raise TypeError("""Passed value of `unknown_peaks` is not a list!
        Instead, it is: """ + str(type(unknown_peaks)))
    if not isinstance(known_compound_peaks, list):
        raise TypeError("""Passed value of `known_compound_peaks` is not a list!
        Instead, it is: """ + str(type(known_compound_peaks)))
    if not isinstance(knownhdf5_filename, str):
        raise TypeError("""Passed value of `knownhdf5_filename` is not a string!
        Instead, it is: """+ str(type(knownhdf5_filename)))
    if not knownhdf5_filename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError("""`knownhdf5_filename` is not type = .hdf5!
        Instead, it is: """+ knownhdf5_filename.split('/')[-1].split('.')[-1])
    if not isinstance(association_matrix, list):
        raise TypeError("""Passed value of `association_matrix` is not a float
        or int! Instead, it is: """ + str(type(association_matrix)))

    # open .hdf5
    knhdf5 = h5py.File(knownhdf5_filename, 'r+')

    #Extract keys from files
    known_compound_list = list(knhdf5.keys())

    if not isinstance(known_compound_list, list):
        raise TypeError("""Passed value of `known_compound_list` is not a list!
        Instead, it is: """+ str(type(known_compound_list)))
    # Now we need to check the elements within the known_compound_list
    # to make sure they are correct.
    for i, _ in enumerate(known_compound_list):
        if not isinstance(known_compound_list[i], str):
            raise TypeError("""Passed value within `known_compound_list` is
            not a string! Instead, it is: """ + str(type(known_compound_list[i])))

    unknown_peak_assignment = []
    #Step through the unknown peaks to make an assignment for each unknown peak.

    for i, _ in enumerate(unknown_peaks):
        # We might be able to make a small performance
        # improvement if we were to somehow
        # not search the peaks we already had searched
        # but that seems to not be trivial.
        position_assignment = []
        # We'll need an outer loop that walks through
        # all the different compound positions
        for j, _ in enumerate(known_compound_peaks):
            if association_matrix[j][i] == 1:
                position_assignment.append(known_compound_list[j])
            else:
                pass
        if position_assignment == []:
            position_assignment.append("Unassigned")
        else:
            pass
        unknown_peak_assignment.append(position_assignment)
    knhdf5.close()
    return unknown_peak_assignment


def percentage_of_peaks_found(known_peaks, association_matrix, knownhdf5_filename):
    """This function takes in a list of classified peaks, and returns a percentage of
    how many of the material's peaks are found in the unknown spectrum.
    This can be used as a metric of confidence."""

    #Handle bad inputs
    if not isinstance(known_peaks, list):
        raise TypeError("""Passed value of `known_peaks` is not a list!
        Instead, it is: """ + str(type(known_peaks)))
    if not isinstance(association_matrix, list):
        raise TypeError("""Passed value of `association_matrix` is not a float or int!
        Instead, it is: """ + str(type(association_matrix)))
    if not isinstance(knownhdf5_filename, str):
        raise TypeError("""Passed value of `knownhdf5_filename` is not a string!
        Instead, it is: """+ str(type(knownhdf5_filename)))
    if not knownhdf5_filename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError("""`knownhdf5_filename` is not type = .hdf5!
        Instead, it is: """ + knownhdf5_filename.split('/')[-1].split('.')[-1])

    # open .hdf5
    knhdf5 = h5py.File(knownhdf5_filename, 'r+')

    # Extract keys from files
    known_compound_list = list(knhdf5.keys())

    if not isinstance(known_compound_list, list):
        raise TypeError("""Passed value of `known_compound_list` is not a list!
        Instead, it is: """+ str(type(known_compound_list)))
    # Now we need to check the elements within the known_compound_list
    # to make sure they are correct.
    for i, _ in enumerate(known_compound_list):
        if not isinstance(known_compound_list[i], str):
            raise TypeError("""Passed value within `known_compound_list`
            is not a string!
            Instead, it is: """ + str(type(known_compound_list[i])))
    percentage_dict = {}
    for j, _ in enumerate(known_compound_list):
#         print(association_matrix)
#         print(known_peaks)
        count_number = sum(association_matrix[j])
        percentage_dict[known_compound_list[j]] = float(count_number /
                                                        (len(known_peaks[j]))) * 100
    knhdf5.close()
    return percentage_dict


def plotting_peak_assignments(unknown_x, unknown_y, unknown_peaks,
                              unknown_peak_assignments, unknownhdf5_filename,
                              knownhdf5_filename, key, peak_labels,
                              exportlabelinput=True):
    """This function plots a set of unknown peaks, and plots the assigned
    classification given by the functions within peakassignment"""

    #Handling errors in inputs.
    if not isinstance(unknown_peaks, list):
        raise TypeError("""Passed value of `unknown_peaks` is not a list!
        Instead, it is: """ + str(type(unknown_peaks)))

    if not isinstance(unknown_x, (list, np.ndarray)):
        raise TypeError("""Passed value of `unknown_x` is not a list or ndarray!
        Instead, it is: """ + str(type(unknown_x)))

    if not isinstance(unknown_y, (list, np.ndarray)):
        raise TypeError(""" Passed value of `unknown_y` is not a list or ndarray!
        Instead, it is: """ + str(type(unknown_y)))
    # handling input errors
    if not isinstance(unknownhdf5_filename, str):
        raise TypeError("""Passed value of `unknownhdf5_filename` is not a string!
        Instead, it is: """ + str(type(unknownhdf5_filename)))
    if not unknownhdf5_filename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError("""`unknownhdf5_filename` is not type = .hdf5!
        Instead, it is: """ + unknownhdf5_filename.split('/')[-1].split('.')[-1])
    if not isinstance(knownhdf5_filename, str):
        raise TypeError("""Passed value of `knownhdf5_filename` is not a string!
        Instead, it is: """ + str(type(knownhdf5_filename)))
    if not knownhdf5_filename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError("""`knownhdf5_filename` is not type = .hdf5!
        Instead, it is: """ + knownhdf5_filename.split('/')[-1].split('.')[-1])
    if not isinstance(key, str):
        raise TypeError("""Passed value of `key` is not a int!
        Instead, it is: """ + str(type(key)))
    if not isinstance(peak_labels, list):
        raise TypeError("""Passed value of `peak_labels` is not a list!
        Instead, it is: """ + str(type(peak_labels)))
    # Now we need to check the elements within the known_compound_list
    # to make sure they are correct.
    for i, _ in enumerate(peak_labels):
        if not isinstance(peak_labels[i], str):
            raise TypeError("""Passed value within `peak_labels` is not a string!
            Instead, it is: """ + str(type(peak_labels[i])))
    #Now we need to check the elements within the unknown_peak_assignment
    #to make sure they are correct.
    for i, _ in enumerate(unknown_peak_assignments):
        if not isinstance(unknown_peak_assignments[i], list):
            raise TypeError("""Passed value within `unknown_peak_assignment`
            is not a list!
            Instead, it is: """ + str(type(unknown_peak_assignments[i])))
            if not isinstance(unknown_peak_assignments[i][i], str):
                raise TypeError("""Passed value within `unknown_peak_assignment`
                is not a string! 
                Instead, it is: """ + str(type(unknown_peak_assignments[i][i])))
    # open .hdf5
    knhdf5 = h5py.File(knownhdf5_filename, 'r')
    unhdf5 = h5py.File(unknownhdf5_filename, 'r')
    residuals = np.asarray(list(unhdf5['{}/residuals'.format(key)]))
    #Extract keys from files
    known_compound_list = list(knhdf5.keys())

    if not isinstance(known_compound_list, list):
        raise TypeError("""Passed value of `known_compound_list` is not a list!
        Instead, it is: """ + str(type(known_compound_list)))
    # Now we need to check the elements within the known_compound_list
    # to make sure they are correct.
    for i, _ in enumerate(known_compound_list):
        if not isinstance(known_compound_list[i], str):
            raise TypeError("""Passed value within `known_compound_list` is
            not a string! Instead, it is: """ + str(type(known_compound_list[i])))
    # extract spectra data
    x_data = list(unhdf5['{}/wavenumber'.format(key)])
    y_data = list(unhdf5['{}/counts'.format(key)])
#     plt.plot(unknown_x, unknown_y, color='black', label='Unknown Spectrum')
    if exportlabelinput:
        print('export labelling only')
    else:
        peak_labels = []
        for i, _ in enumerate(unknown_peak_assignments):
            peak_labels.append(str(unknown_peak_assignments[i]))
    print(peak_labels)
    # plot spectra and peak labels
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   figsize=(15, 6), dpi=300)
    # plot data
    ax1.plot(x_data, y_data, color='blue')
    ax2.plot(x_data, residuals, color='teal')
    lineid_plot.plot_line_ids(x_data, y_data, unknown_peaks,
                              peak_labels, box_axes_space=0.30,
                              plot_kwargs={'linewidth':1},
                              max_iter=75, ax=ax1)
#     fig.set_size_inches(15,5)
    # lock the scale so that additional plots do not warp the labels
    ax1.set_autoscale_on(False)
    # Titles and labels
    ax2.set_xlabel('Wavenumber ($cm^{-1}$)', fontsize=14)
    ax1.set_xlim(min(x_data), max(x_data))
    ax1.set_ylabel('Counts', fontsize=14, labelpad=20)
    ax2.set_ylabel('Residuals', fontsize=14, labelpad=12)
    # scale residuals plot symmetrically about zero
    ylim = max(abs(min(residuals)), abs(max(residuals)))
    ax2.set_ylim(-ylim, ylim)
    # add grid lines to residual plot
    ax2.grid(which='major', axis='y', linestyle='-')
    # force tick labels for top plot
    ax1.tick_params(axis='both', which='both', labelsize=10, labelbottom=True)
    # add title
    ax1.set_title('{} spectra from {}'.format(key, unknownhdf5_filename),
                  fontsize=18, pad=350)
    plt.show()
    knhdf5.close()
    unhdf5.close()

def add_label(hdf5_filename, key, peak, label):
    """Function that adds a label to a peak dataset in the hdf5 file
    """
    #Handling errors in inputs.
    if not isinstance(hdf5_filename, str):
        raise TypeError("""Passed value of `hdf5_filename` is not a string!
        Instead, it is: """ + str(type(hdf5_filename)))
    if not hdf5_filename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError("""`hdf5_filename` is not type = .hdf5!
        Instead, it is: """ + hdf5_filename.split('/')[-1].split('.')[-1])
    if not isinstance(key, str):
        raise TypeError("""Passed value of `key` is not a int!
        Instead, it is: """ + str(type(key)))
    if not isinstance(key, str):
        raise TypeError("""Passed value of `key` is not a int!
        Instead, it is: """ + str(type(key)))
    if not isinstance(peak, str):
        raise TypeError("""Passed value of `peak` is not a string!
        Instead, it is: """ + str(type(peak)))
    if not isinstance(label, str):
        raise TypeError("""Passed value of `label` is not a string!
        Instead, it is: """ + str(type(label)))
    # open hdf5 file as read/write
    hdf5 = h5py.File(hdf5_filename, 'r+')
    # extract existing data from peak dataset
    peak_data = list(hdf5['{}/{}'.format(key, peak)])[0]
#     print(peak_data)
    # make a new tuple that contains the orginal data as well as the label
    label_tuple = (label,)
    data = tuple(peak_data) +label_tuple
    # delete the old dataset so the new one can be saved
    del hdf5['{}/{}'.format(key, peak)]
    # define a custom datatype that allows for a string as the the last tuple element
    my_datatype = np.dtype([('fraction', np.float),
                            ('center', np.float),
                            ('sigma', np.float),
                            ('amplitude', np.float),
                            ('fwhm', np.float),
                            ('height', np.float),
                            ('area under the curve', np.float),
                            ('label', h5py.special_dtype(vlen=str))])
    # recreate the old dataset in the hdf5 file
    dataset = hdf5.create_dataset('{}/{}'.format(key, peak),
                                  (1,), dtype=my_datatype)
    # apply custom dtype to data tuple
#     print(dataset)
    print(data)
#     print(my_datatype)
    data_array = np.array(data, dtype=my_datatype)
    # write new values to the blank dataset
    dataset[...] = data_array
#     print(dataset)
    hdf5.close()
    return data

def peak_1d_score(row_i, row_j, scoremax, precision):
    """
    Returns scores with respect to the repricoal of the
    calculated Euclidean distance between peaks
    #√((x1-x2)^2) in 1D
    #√((x1-x2)^2 + (y1-y2)^2) in 2D

    Parameters:
        row_i (list like):  input list
        row_j (list like): input list
        scoremax (float): Euclidean reciprocal score divided by max score;
        default is 1

    Returns:
        scores (list): Euclidean reciprocal scores
        peaks (tuple): peaks associated with scores
    """
    # Handling errors at the input
    if not isinstance(row_i, (list, np.ndarray)):
        raise TypeError("""Passed value of `row_i` is not a list or ndarray!
        Instead, it is: """ + str(type(row_i)))
    if not isinstance(row_j, (list, np.ndarray)):
        raise TypeError("""Passed value of `row_j` is not a list or ndarray!
        Instead, it is: """ + str(type(row_j)))
    if not isinstance(scoremax, (float, int)):
        raise TypeError("""Passed value of `scoremax` is not a float or int!
        Instead, it is: """ + str(type(scoremax)))
    if scoremax < 0:
        raise ValueError("""Passed value of `scoremax` is not within bounds!""")

    # Initializing the variables
    scores = []
    peaks = []

    for i, _ in enumerate(row_i):
        for j, _ in enumerate(row_j):
            # Calculating distances between peaks
            distance = np.where((row_i[i] - row_j[j] > precision), np.nan,
                                math.sqrt(sum([math.pow(row_i[i] - row_j[j], 2)])))
            # Score for peaks less than 50 units apart
            if 1 / (distance + 1) > (1/precision):
                # Dividing over the given max score
                scores.append(((1 / (distance + 1)) / scoremax))
                # Appends a tuple of the compared peaks
                peaks.append((row_i[i], row_j[j]))
            else:
                pass
    return scores, peaks


def score_max(row_i, row_j, k, precision):
    """
    Returns list of scores sorted with respect to the peaks
    related to its output max score

    Parameters:
        row_i (list like):  input list
        row_j (list like): input list
        k (int): input integer used to sort the scores / kth highest score

    Returns:
        maxscores (list): Euclidean reciprocal score divided by max score
        maxpeaks (tuple): peaks associated with max scores
    """

    # Handling errors at the input
    if not isinstance(row_i, (list, np.ndarray)):
        raise TypeError("""Passed value of `row_i` is not a list or ndarray!
        Instead, it is: """ + str(type(row_i)))
    if not isinstance(row_j, (list, np.ndarray)):
        raise TypeError("""Passed value of `row_j` is not a list or ndarray!
        Instead, it is: """ + str(type(row_j)))
    if not isinstance(k, int):
        raise TypeError("""Passed value of `k` is not an int!
        Instead, it is: """ + str(type(k)))
    if k < 0:
        raise ValueError("""Passed value of `k` is not within bounds!""")
    try:
        scoremax = sorted(set(peak_1d_score(row_i, row_j, 1, precision)[0][:]))[-k]
        maxscores, maxpeaks = peak_1d_score(row_i, row_j, scoremax, precision)

    except Exception as e:
        print("""Function did not receive a scoremax variable. The variable
        scoremax has been reset back to 1. This is equivalent to 
        your unnormalized score.""")

        maxscores, maxpeaks = peak_1d_score(row_i, row_j, 1, precision)

    return maxscores, maxpeaks


def score_sort(row_i, row_j, k, precision):
    """
    Returns list of scores sorted

    Parameters:
        list_input (list like):  input list
        row (list like): input list
        k (int): input integer used to sort the scores / kth highest score

    Returns:
        sortedscores (list): sorted Euclidean distances
    """
    # Handling errors at the input
    if not isinstance(row_i, (list, np.ndarray)):
        raise TypeError("""Passed value of `row_i` is not a list or ndarray!
        Instead, it is: """ + str(type(row_i)))
    if not isinstance(row_j, (list, np.ndarray)):
        raise TypeError("""Passed value of `row_j` is not a list or ndarray!
        Instead, it is: """ + str(type(row_j)))
    if not isinstance(k, int):
        raise TypeError("""Passed value of `k` is not an int!
        Instead, it is: """ + str(type(k)))
    if k < 0:
        raise ValueError("""Passed value of `k` is not within bounds!""")

    sortedscores = []
    sortedscores.append(score_max(row_i, row_j, k, precision))
    sortedscores.sort()

    return sortedscores
