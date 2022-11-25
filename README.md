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

**bambird** is hosted on PyPI. To install, run the following command in your Python environment:

```bash
$ pip install bambird
```

To install the latest version from source clone the master repository and from the top-level folder call:

```bash
$ git clone https://github.com/ear-team/bambird.git && cd bambird
$ pip install -e .
```

## Examples and documentation

- See the directory "example" to find scripts to run the labelling function on a collection of birds species or on a single file
- More example scripts will be available soon on [colab](https://colab.research.google.com/)
  - [workflow single species](https://colab.research.google.com/drive/18tglsE1JciyD1xpTryX3JIenHKGScLSq#scrollTo=bzlosQhqt7vf)
- Full description of the package **scikit-maad**: https://doi.org/10.1111/2041-210X.13711
- Online reference manual and example gallery of **scikit-maad** [here](https://scikit-maad.github.io/).
- In depth information related to the Multiresolution Analysis of Acoustic Diversity implemented in scikit-maad was published in: Ulloa, J. S., Aubin, T., Llusia, D., Bouveyron, C., & Sueur, J. (2018). [Estimating animal acoustic diversity in tropical environments using unsupervised multiresolution analysis](https://doi.org/10.1016/j.ecolind.2018.03.026). Ecological Indicators, 90, 346–355

## Citing this work

If you find **bambird** usefull for your research, please consider citing it as:

- Michaud, F.,  Sueur, J., Le Cesne, M., & Haupert, S. (2022). [Unsupervised classification to improve the quality of a bird song recording dataset](https://doi.org/xxx). Ecological Informatics, xx, xxx–xxx

## Contributions and bug report

Improvements and new features are greatly appreciated. If you would like to contribute developing new features or making improvements to the available package, please refer to our [wiki](https://github.com/ear-team/bambird/wiki/How-to-contribute-to-bambird). Bug reports and especially tested patches may be submitted directly to the [bug tracker](https://github.com/ear-team/bambird/issues). 

