#!/usr/bin/env python
#
# Copyright (C) 2022 Felix Michaud <felixmichaudlnhrdt@gmail.com>
#                    Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
# License: BSD 3 clause

import os
import textwrap
from setuptools import setup, find_packages, Command
from importlib.machinery import SourceFileLoader

version = SourceFileLoader('bambird.version',
                           'bambird/version.py').load_module()

with open('README.md', 'r') as fdesc:
    long_description = fdesc.read()

class CleanCommand(Command):
    """Custom clean command to tidy up the project root.
    Deletes directories ./build, ./dist and ./*.egg-info
    From the terminal type:
        > python setup.py clean
    """
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.egg-info')

setup(
      name = 'bambird',
      version = version.__version__,  # Specified at bambird/version.py file
      packages = find_packages(),
      author = 'Felix Michaud and Sylvain Haupert',
      author_email = 'felixmichaudlnhrdt@gmail.com, sylvain.haupert@mnhn.fr',
      maintainer = 'Felix Michaud and Sylvain Haupert',
      description = 'BAM, unsupervised labelling function to extract and cluster similar animal vocalizations together',
      long_description = long_description,
      long_description_content_type='text/markdown',
      license = 'BSD 3 Clause',
      keywords = ['ecoacoustics', 'bioacoustics', 'ecology', 'dataset', 'signal processing', 'segmentation', 'features', 'clustering', 'unsupervised  labelling', 'deep learning', 'machine learning'],
      url = 'https://github.com/ear-team/bambird',
      #project_urls={'Documentation': 'https://bambird.github.io'},
      platform = 'OS Independent',
      cmdclass={'clean': CleanCommand},
      license_file = 'LICENSE',                     
      python_requires='>=3.5',
      install_requires = ['scikit-image>=0.19.2',
                          'scikit-maad>=1.3.12',
                          'librosa>=0.8.0',
                          'scikit-learn>=1.0',
                          'hdbscan',
                          'matplotlib', 
                          'umap-learn',
                          'tqdm',
                          'kneed',
                          'pyyaml'],
      classifiers=textwrap.dedent("""
        Development Status :: 4 - Beta
        Intended Audience :: Science/Research
        License :: OSI Approved :: BSD License
        Operating System :: OS Independent
        Programming Language :: Python :: 3.5
        Programming Language :: Python :: 3.6
        Programming Language :: Python :: 3.7
        Programming Language :: Python :: 3.8
        Topic :: Scientific/Engineering :: Artificial Intelligence 
        """).strip().splitlines()
       )
