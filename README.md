# bamscape


## Installation

First install the environment `xprize-final`
```bash
$ conda env update -n my_env --file environment.yaml
```

Then install the latest version of the dev branch from source clone the master repository and from the top-level folder call:

```bash
$ git clone --branch dev https://github.com/ear-team/bambird.git && cd bambird
$ pip install -e .
```

Optional, install a wrapper of birdnet to get the embeddings. 
birdnetlib requires Python 3.9+ and prior installation of Tensorflow Lite and ffmpeg.

Prequisite
```bash
pip install tflite-runtime
```
```bash
sudo apt-get install ffmpeg
```
Then install birdnetlib
```bash
pip install birdnetlib
```

## Usage

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

