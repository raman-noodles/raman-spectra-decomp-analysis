from setuptools import setup

setup(name='raman-spectra-decomp-analysis',
      version='1.0.beta',
      description='A python package for analyzing raman spectroscopy decomposition data.',
      url='https://github.com/raman-noodles/raman-specrta-decomp-analysis',
      author='Raman Noodles Group, University of Washington (2019)',
      license='MIT',
      packages=['raman-spectra-decomp-analysis'],
      install_requires=['numpy', 'jcamp', 'requests', 'matplotlib', 'scipy', 'lmfit', 'peakutils', 'h5py', 'pandas', 'xlrd'])
