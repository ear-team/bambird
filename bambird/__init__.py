# -*- coding: utf-8 -*-
""" 



"""
from .version import __version__

# -*- coding: utf-8 -*-
""" 
Labelling function
==================

The package ``bam`` is an ensemble of functions to build a labelling function to clean audio recordings from Xeno-Canto.

Config yaml
-----------
.. autosummary::
    :toctree: generated/
    get_loader

Dataset
-------
.. autosummary::
    :toctree: generated/
    query
    download
    grab_audio_to_df
    change_path
    
ROIs segmentation
-----------------
.. autosummary::
    :toctree: generated/
    
    intersection_bbox
    fusion_bbox
    merge_bbox
    extract_rois_core
    extract_rois_full_sig    
    single_file_extract_rois
    multicpu_extract_rois
    save_rois

ROIs features
-------------
.. autosummary::
    :toctree: generated/
    
    compute_features
    multicpu_compute_features
    
ROIs clustering
---------------
.. autosummary::
    :toctree: generated/
    
    prepare_features
    find_cluster
    cluster_eval
    overlay_rois
    mark_rois
    unmark_rois

"""

from .config import (
    get_loader
    )

from .dataset import(
    grab_audio_to_df,
    change_path,
    query_xc,
    download_xc,
    )

from .segmentation import(     
    extract_rois_core,
    extract_rois_full_sig,   
    single_file_extract_rois,
    multicpu_extract_rois,
    )

from .features import(
    compute_features,
    multicpu_compute_features,
    )
                 
from .cluster import (
    prepare_features,
    find_cluster,
    cluster_eval,
    overlay_rois,
    mark_rois,
    unmark_rois
    )


__all__ = [
        # config.py
        'get_loader',
        # dataset.py
        'grab_audio_to_df',
        'change_path',
        'query_xc',
        'download_xc',
        # segmentation.py                         
        'extract_rois_core',
        'extract_rois_full_sig',
        'single_file_extract_rois',
        'multicpu_extract_rois',
        # features.py
        'compute_features',
        'multicpu_compute_features',
        # cluster.py
        'prepare_features',
        'find_cluster',
        'cluster_eval',
        'overlay_rois',
        'mark_rois',
        'unmark_rois'
        ]


