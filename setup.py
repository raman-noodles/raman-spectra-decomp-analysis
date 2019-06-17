from setuptools import setup

setup(name='ramandecompy',
      version='1.0.beta',
      description='A python package for analyzing raman spectroscopy decomposition data.',
      url='https://github.com/raman-noodles/raman-spectra-decomp-analysis',
      author='Raman Noodles Group, University of Washington (2019)',
      license='MIT',
      packages=['ramandecompy'],
      install_requires=['numpy', 'matplotlib', 'scipy', 'lmfit', 'peakutils', 'h5py', 'pandas', 'xlrd', 'lineid_plot', 'sklearn'])