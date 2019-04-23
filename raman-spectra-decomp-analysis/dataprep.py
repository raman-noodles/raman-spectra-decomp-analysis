"""docstring"""
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from ramannoodles import spectrafit


def new_hdf5(new_filename):
    """docstring"""
    # handling input errors
    if not isinstance(new_filename, str):
        raise TypeError('Passed value of `filename` is not a string! Instead, it is: '
                        + str(type(new_filename)))
    # w- mode will create a file and fail if the file already exists
    hdf5 = h5py.File('{}.hdf5'.format(new_filename), 'w-')
    hdf5.close()


def add_calibration(hdf5_filename, data_filename, label=None):
    """docstring"""
    # handling input errors
    if not isinstance(hdf5_filename, str):
        raise TypeError('Passed value of `cal_filename` is not a string! Instead, it is: '
                        + str(type(hdf5_filename)))
    if not hdf5_filename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError('`cal_filename` is not type = .hdf5! Instead, it is: '
                        + hdf5_filename.split('/')[-1].split('.')[-1])
    if not isinstance(data_filename, str):
        raise TypeError('Passed value of `data_filename` is not a string! Instead, it is: '
                        + str(type(data_filename)))
    # r+ is read/write mode and will fail if the file does not exist
    cal_file = h5py.File(hdf5_filename, 'r+')
    data = pd.read_excel(data_filename, header=None, names=('x', 'y'))
    if data_filename.split('.')[-1] == 'xlsx':
        data = pd.read_excel(data_filename, header=None, names=('x', 'y'))
    elif data_filename.split('.')[-1] == 'csv':
        data = pd.read_csv(data_filename, header=None, names=('x', 'y'))
    else:
        print('data file type not recognized')
    # ensure that the data is listed from smallest wavenumber first
    if data['x'][:1].values > data['x'][-1:].values:
        data = data.iloc[::-1]
        data.reset_index(inplace=True, drop=True)
    else:
        pass
    # peak detection and data fitting
    fit_result = spectrafit.fit_data(data['x'].values, data['y'].values)
    # write data to .hdf5 using custom label if provided
    if label is not None:
        cal_file['{}/wavenumber'.format(label)] = data['x']
        cal_file['{}/counts'.format(label)] = data['y']
        for i, _ in enumerate(fit_result):
            cal_file['{}/Peak_{}'.format(label, i+1)] = fit_result[i]
    else:
        label = (data_filename.split('/')[-1]).split('.')[0]
        cal_file['{}/wavenumber'.format(label)] = data['x']
        cal_file['{}/counts'.format(label)] = data['y']
        for i, _ in enumerate(fit_result):
            cal_file['{}/Peak_{}'.format(label, i+1)] = fit_result[i]
    cal_file.close()


def add_experiment(hdf5_filename, exp_filename):
    """docstring"""
    # handling input errors
    if not isinstance(hdf5_filename, str):
        raise TypeError('Passed value of `hdf5_filename` is not a string! Instead, it is: '
                        + str(type(hdf5_filename)))
    if not hdf5_filename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError('`hdf5_filename` is not type = .hdf5! Instead, it is: '
                        + hdf5_filename.split('/')[-1].split('.')[-1])
    if not isinstance(exp_filename, str):
        raise TypeError('Passed value of `data_filename` is not a string! Instead, it is: '
                        + str(type(exp_filename)))
    # r+ is read/write mode and will fail if the file does not exist
    exp_file = h5py.File(hdf5_filename, 'r+')
    if exp_filename.split('.')[-1] == 'xlsx':
        data = pd.read_excel(exp_filename, header=None, names=('x', 'y'))
    elif exp_filename.split('.')[-1] == 'csv':
        data = pd.read_csv(exp_filename, header=None, names=('x', 'y'))
    else:
        print('data file type not recognized')
    # ensure that the data is listed from smallest wavenumber first
    if data['x'][:1].values > data['x'][-1:].values:
        data = data.iloc[::-1]
        data.reset_index(inplace=True, drop=True)
    else:
        pass
    # peak detection and data fitting
    fit_result = spectrafit.fit_data(data['x'].values, data['y'].values)
    # extract experimental parameters from filename
    specs = exp_filename.split('/')[-1].split('.')[:-1]
    if len(specs) > 1:
        spec = ''
        for _,element in enumerate(specs):
            spec = str(spec+element)
        specs = spec
    specs = specs.split('_')
    specs
    time = specs[-1]
    temp = specs[-2]
    # write data to .hdf5
    exp_file['{}/{}/wavenumber'.format(temp, time)] = data['x']
    exp_file['{}/{}/counts'.format(temp, time)] = data['y']
    for i, _ in enumerate(fit_result):
        if i < 9:
            exp_file['{}/{}/Peak_0{}'.format(temp, time, i+1)] = fit_result[i]
        else:
            exp_file['{}/{}/Peak_{}'.format(temp, time, i+1)] = fit_result[i]
    exp_file.close()


def view_hdf5(filename):
    """docstring"""
        # handling input errors
    if not isinstance(filename, str):
        raise TypeError('Passed value of `filename` is not a string! Instead, it is: '
                        + str(type(filename)))
    if not filename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError('`filename` is not type = .hdf5! Instead, it is: '
                        + filename.split('/')[-1].split('.')[-1])
    # pring groups and datasets in first three layers
    print('**** {} ****'.format(filename))
    hdf5 = h5py.File(filename, 'r')
    for _,layer_1 in enumerate(list(hdf5.keys())):
        if isinstance(hdf5[layer_1], h5py.Group):
            print('\033[1m{}\033[0m'.format(layer_1))
            for _,layer_2 in enumerate(list(hdf5[layer_1].keys())):
                if isinstance(hdf5['{}/{}'.format(layer_1, layer_2)], h5py.Group):
                    print('|    \033[1m{}\033[0m'.format(layer_2))
                    for _,layer_3 in enumerate(list(hdf5['{}/{}'.format(layer_1, layer_2)])):
                        if isinstance(hdf5['{}/{}/{}'.format(layer_1, layer_2, layer_3)], h5py.Group):
                            print('|    |    \033[1m{}\033[0m/...'.format(layer_3))
                        else:
                            print('|    |    {}'.format(layer_3))
                else:
                    print('|    {}'.format(layer_2))
        else:
            print('{}'.format(layer_1))


def plot_fit(hdf5_filename, key):
    """docstring"""
    # handling input errors
    if not isinstance(hdf5_filename, str):
        raise TypeError('Passed value of `hdf5_filename` is not a string! Instead, it is: '
                        + str(type(hdf5_filename)))
    if not hdf5_filename.split('/')[-1].split('.')[-1] == 'hdf5':
        raise TypeError('`hdf5_filename` is not type = .hdf5! Instead, it is: '
                        + hdf5_filename.split('/')[-1].split('.')[-1])
    if not isinstance(key, str):
        raise TypeError('Passed value of `key` is not a string! Instead, it is: '
                        + str(type(key)))
    # open .hdf5
    hdf5 = h5py.File(hdf5_filename, 'r')
    # extract spectra data
    x_data = list(hdf5['{}/wavenumber'.format(key)])
    y_data = list(hdf5['{}/counts'.format(key)])
    # extract fitted peak center values
    peak_centers = []
    for _,peak in enumerate(list(hdf5[key])[:-2]):
        peak_centers.append(list(hdf5['{}/{}'.format(key, peak)])[2])
    # plot spectra and peak center values
    plt.figure(figsize=(16,5))
    plt.plot(x_data, y_data, label = 'spectra data')
    for i,peak in enumerate(peak_centers):
        if i == 0:
            plt.axvline(x=peak, color='orange', alpha=0.6, label='detected peak')
        else:
            plt.axvline(x=peak, color='orange', alpha=0.6)
    plt.xlabel('wavenumber ($cm^{-1}$)', fontsize=14)
    plt.xlim(min(x_data), max(x_data))
    plt.ylabel('counts', fontsize=14)
    plt.title('{} spectra from {}'.format(key, hdf5_filename), fontsize=16)
    plt.legend(fontsize=12)