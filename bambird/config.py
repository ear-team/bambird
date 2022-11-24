#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Collection of functions to load the configuration file
"""
#
# Authors:  Felix MICHAUD   <felixmichaudlnhrdt@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
#
# License: New BSD License

#%%
# general packages
import sys

# basic packages
import yaml

#%%

RANDOM_SEED = 1979  # Fix the random seed to be able to repeat the results

DEFAULT_PARAMS_XC = {
    'PARAM_XC_LIST': ['len:"20-180"', 'q:">C"'],
    'NUM_FILES': 10
}

DEFAULT_PARAMS_EXTRACT = {
    # Extract Audio resampling
    "SAMPLE_RATE": 44100,  # Sampling frequency in Hz
    # Audio preprocess
    "LOW_FREQ": 250,  # Low frequency in Hz of the bandpass filter applied to the audio
    "HIGH_FREQ": 12000,  # High frequency in Hz of the bandpass filter applied to the audio
    # butterworth filter order to select the bandwidth corresponding to the ROI
    "BUTTER_ORDER": 5,
    # Max duration of the audio files that we will use to compute the features
    "AUDIO_DURATION": 20,
    # Split the audio signal of chunk with duration = SIGNAL LENGTH (in second)
    "SIGNAL_LENGTH": 20,
    "OVLP": 0,  # Define the overlap ratio between each chunk
    # Spectrogram
    # Mode to compute the remove_background ('mean', 'median')
    "MODE_RMBCKG": "median",
    # Number of points used to compute the running mean of the noise profil
    "N_RUNNING_MEAN": 10,
    "NFFT": 2048,  # Number of points of the spectrogram
    # Combination of parameters for ROIs extraction
    "MASK_PARAM1": 50,  # 30 37
    "MASK_PARAM2": 30,  # 20 33
    # Select and merge bbox parameters
    # values should be a multiple of the resolution/2 of the reducted spectro
    # df_reduc = 323Hz FACTOR_F = 15, NFFT = 2048 and SAMPLE_RATE = 44100
    # dt_reduc = 0.232s for FACTOR_T = 10, NFFT = 2048 and SAMPLE_RATE = 44100
    "MIN_DURATION": 0.1,  # minimum event duration in s
    "MARGIN_T_LEFT": 0.2,
    "MARGIN_T_RIGHT": 0.2,
    "MARGIN_F_TOP": 100,
    "MARGIN_F_BOTTOM": 100,
    # save parameters
    "MARGIN_T": 0.1,  # time margin in s around the ROI
    "MARGIN_F": 250,  # frequency margin in Hz around the ROI
    # butterworth filter order to select the bandwidth corresponding to the ROI
    "FILTER_ORDER": 5,
}

DEFAULT_PARAMS_FEATURES = {
    # Extract Audio resampling
    "SAMPLE_RATE": 24000,  # Sampling frequency in Hz
    # Audio preprocess
    "LOW_FREQ": 250,  # Low frequency in Hz of the bandpass filter applied to the audio
    "HIGH_FREQ": 11000,  # High frequency in Hz of the bandpass filter applied to the audio
    # butterworth filter order to select the bandwidth corresponding to the ROI
    "BUTTER_ORDER": 1,
    # Spectrogram
    # Number of points used to compute the running mean of the noise profil
    "N_RUNNING_MEAN": 35,
    "NFFT": 2048,  # Number of points of the spectrogram
    "SHAPE_RES": "med",
}

DEFAULT_PARAMS_CLUSTER = {
    "PERCENTAGE_PTS": 10,       # in %
    "METHOD": "HDBSCAN",        # HDBSCAN or DBSCAN
    "SCALER": "STANDARDSCALER", # STANDARDSCALER or ROBUSTSCALER or MINMAXSCALER
    "KEEP":   "BIGGEST"         # ALL or BIGGEST
}


#%%
""" ===========================================================================

                    Private functions 

============================================================================"""
def _fun_call_by_name(val):
    if '.' in val:
        module_name, fun_name = val.rsplit('.', 1)
        # Restrict which modules may be loaded here to avoid safety issue
        # Put the name of the module
        assert module_name.startswith('bambird')
    else:
        module_name = '__main__'
        fun_name = val
    try:
        __import__(module_name)
    except ImportError :
        raise ("Can''t import {} while constructing a Python object".format(val))
    module = sys.modules[module_name]
    fun = getattr(module, fun_name)
    return fun

def _fun_constructor(loader, node):
    val = loader.construct_scalar(node)
    print(val)
    return _fun_call_by_name(val)

""" ===========================================================================

                    Public function 

============================================================================"""

def get_loader():
  """Add constructors to PyYAML loader."""
  loader = yaml.SafeLoader
  loader.add_constructor("!fun", _fun_constructor)
  return loader
