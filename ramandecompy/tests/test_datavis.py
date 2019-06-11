"""docstring"""
import h5py
from ramandecompy import datavis


def test_plot_fit():
    """
    docstring
    """
    hdf5_filename = 'ramandecompy/tests/test_files/test_experiment.hdf5'
    key = '300C/25s'
    datavis.plot_fit(hdf5_filename, key)
    #test inputs
    try:
        datavis.plot_fit(4.2, '300C/25s')
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    try:
        datavis.plot_fit('test.txt', '300C/25s')
    except TypeError:
        print('A .txt was passed to the function, and it was handled well with a TypeError.')
    try:
        datavis.plot_fit('ramandecompy/tests/test_files/dataprep_experiment.hdf5', 4.2)
    except TypeError:
        print('A float was passed to the function, and it was handled well with a TypeError.')
    hdf5 = h5py.File(hdf5_filename, 'r')
    assert key in hdf5, 'input key does not exist within the specified .hdf5 file'
    hdf5.close()
