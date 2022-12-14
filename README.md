# bambird package

## Unsupervised classification to improve the quality of a bird song recording dataset


Open audio databases such as [Xeno-Canto](https://xeno-canto.org/) are widely used to build datasets to explore bird song repertoire or to train models for automatic bird sound classification by deep learning algorithms. However, such databases suffer from the fact that bird sounds are weakly labelled: a species name is attributed to each audio recording without timestamps that provide the temporal localization of the bird song of interest. 
Manual annotations can solve this issue, but they are time consuming, expert-dependent, and cannot run on large datasets. Another solution consists in using a labelling function that automatically segments audio recordings before assigning a label to each segmented audio sample. Although labelling functions were introduced to expedite strong label assignment, their classification performance remains mostly unknown. 
To address this issue and reduce label noise (wrong label assignment) in large bird song datasets, we introduce a data-centric novel labelling function composed of three successive steps: 1) time-frequency sound unit segmentation, 2) feature computation for each sound unit, and 3) classification of each sound unit as bird song or noise with either an unsupervised DBSCAN algorithm or the supervised BirdNET neural network. 
The labelling function was optimized, validated, and tested on the songs of 44 West-Palearctic common bird species. We first showed that the segmentation of bird songs alone aggregated from 10% to 83% of label noise depending on the species. We also demonstrated that our labelling function was able to significantly reduce the initial label noise present in the dataset by up to a factor of three. Finally, we discuss different opportunities to design suitable labelling functions to build high-quality animal vocalizations with minimum expert annotation effort.

<br/>
<div align="center">
    <img src="./docs/figure_workflow_sans_alpha.png" alt="drawing"/>
</div>
<br/>

Based on this work, we propose **bambird**, an open source Python package that provides a complete workflow to create your own labelling function to build cleaner bird song recording dataset. **bambird** is mostly based on [scikit-maad](https://github.com/scikit-maad/scikit-maad) package

[![DOI](https://zenodo.org/badge/xxx.svg)](https://zenodo.org/badge/latestdoi/xxxxx)

## Installation
bambird dependencies:

- scikit-maad >= 1.3.12
- librosa
- scikit-learn
- kneed
- hdbscan
- tqdm
- umap-learn

**bambird** is hosted on PyPI. To install, run the following command in your Python environment:

```bash
$ pip install bambird
```

To install the latest version from source clone the master repository and from the top-level folder call:

```bash
$ git clone https://github.com/ear-team/bambird.git && cd bambird
$ pip install -e .
```
## Functions
The functions available in the package are:
from config.py
>- **load_config** : Load the configuration file to set all the parameters of bambird

from dataset.py
>- **query_xc** : Query metadata from Xeno-Canto website with audiofile depending on the search terms. The audio recordings metadata are grouped and stored in a dataframe.
>- **download_xc**: Download the audio files from Xeno-Canto based on the input dataframe.  It will create directories for each species if needed
>- **grab_audio_to_df**: create a dataframe with all recordings in the directory.  The first column name corresponds to full path to the filename. The second column name correspond to the filename alone without the extension
>- **change_path**:  change the path to the audio in the dataframe. This is usefull when the audio are moved from their original place

from segmentation.py                         
>- **extract_rois_core**: function called by single_file_extract_rois. Define a specific process to extract Rois. In this case, the function extract the most energetic part of songs/calls.
>- **extract_rois_full_sig**:f unction called by single_file_extract_rois. Define a specific process to extract Rois. In this case, the function extract the full songs/calls.
>- **single_file_extract_rois**: Extract all Rois in a single audio file
>- **multicpu_extract_rois**: Extract all Rois in the dataset (multiple audio files)

from features.py
>- **compute_features**: Compute features of a single Roi such as shape (wavelets), centroid and bandwidth
>- **multicpu_compute_features**: Compute features such as shape (wavelets), centroid and bandwidth of all Rois in the dataset (multiple audio files)

from cluster.py
>- **find_cluster**:  Clustering of ROIs.  Use DBSCAN or HDSCAN clustering method for several reasons :
        * DBSCAN does not need the number of clusters to do the clustering
        * DBSCAN is able to deal with noise and keep them outside any clusters.
        So, the goal of the clustering is to aggregate similar ROIs
    which might correspond to the main call or song of a species. If several 
    clusters are found, which means that we might have ROIs corresponding to 
    different calls and/or songs for the species, we can keep the cluster with 
    the highest number of ROIs or all the clusters.
>- **cluster_eval**:   Evaluation of the clustering (requires annotations or any other files to compare with the result of the clustering)
>- **overlay_rois**: Overlay Rois with colors and number depending on the cluster number or the label.
>- **mark_rois**: Add a marker to the audio filenames of each Roi depending on the result of the evaluation of the clustering (TN, FN, TP, FP)
>- **unmark_rois**: Remove the markers


## Examples and documentation

- See the directory "example" to find scripts to run the labelling function on a collection of birds species or on a single file
- All scripts to reproduce the results found the article as well as the examples are also on [colab](https://colab.research.google.com/)
  - [clustering_only_train](https://colab.research.google.com/drive/1RTRo3DQ1czDAb4QaOsE30eK2JKSZlbZQ)
  - [clustering_only_validation](https://colab.research.google.com/drive/1fTPK8LAt97jcXP0XuyY09J4ZbxvFHO4i)
  - [clustering_only_test](https://colab.research.google.com/drive/1K-Os_ZVivtk_-REsAYNmpzKiaxYa0AvO)
  - [birdnet_train](https://colab.research.google.com/drive/1Ev-Xjc4evEIhT3HlqeGKrEoTPsUPc4fF)
  - [birdnet_validations](https://colab.research.google.com/drive/19jvks3rv678ZJF4nAdNTD69C6lGc7V44)
  - [birdnet_test](https://colab.research.google.com/drive/1So-L8LE5duk7EavSb9kY_ecDlr4zOT8s)
  - [full_process_validation](https://colab.research.google.com/drive/1gR8ECKZBzf50y7A_JEj-eLTx7p_IzmIt)
  - [full_process_test](https://colab.research.google.com/drive/1oKYt548aroTuILoM5AACd2sNJ954X1G_)
  - [workflow_single_file](https://colab.research.google.com/drive/1DgK-LlovEv_0jh70dggqlV2G0jbJCfKi)
  - [workflow_multiple_species](https://colab.research.google.com/drive/18tglsE1JciyD1xpTryX3JIenHKGScLSq)  
- Full description of the package **scikit-maad**: https://doi.org/10.1111/2041-210X.13711
- Online reference manual and example gallery of **scikit-maad** [here](https://scikit-maad.github.io/).
- In depth information related to the Multiresolution Analysis of Acoustic Diversity implemented in scikit-maad was published in: Ulloa, J. S., Aubin, T., Llusia, D., Bouveyron, C., & Sueur, J. (2018). [Estimating animal acoustic diversity in tropical environments using unsupervised multiresolution analysis](https://doi.org/10.1016/j.ecolind.2018.03.026). Ecological Indicators, 90, 346–355

## Citing this work

If you find **bambird** usefull for your research, please consider citing it as:

- Michaud, F.,  Sueur, J., Le Cesne, M., & Haupert, S. (2022). [Unsupervised classification to improve the quality of a bird song recording dataset](https://doi.org/xxx). Ecological Informatics, xx, xxx–xxx

## Contributions and bug report

Improvements and new features are greatly appreciated. If you would like to contribute developing new features or making improvements to the available package, please refer to our [wiki](https://github.com/ear-team/bambird/wiki/How-to-contribute-to-bambird). Bug reports and especially tested patches may be submitted directly to the [bug tracker](https://github.com/ear-team/bambird/issues). 

