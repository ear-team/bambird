#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Collection of functions to cluster separate the sound of interests (or regions
of interest ROIs) from the noise, compute metrics and overlay the results 
"""
#
# Authors:  Felix MICHAUD   <felixmichaudlnhrdt@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
#
# License: New BSD License

#%%
# general packages
import os
import warnings
from pathlib import Path

# basic packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# audio package
import librosa

# Scikit-Maad (ecoacoustics functions) package
import maad

# scikit-learn (machine learning) package
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# Kneed package to find the knee of a curve
from kneed import KneeLocator

# HDBSCAN package (clustering)
import hdbscan

from bamxc import config as cfg

warnings.filterwarnings("ignore", module="librosa")
warnings.filterwarnings("ignore", module="maad")
warnings.filterwarnings(action="ignore")

plt.rcParams.update({"figure.max_open_warning": 0})
plt.style.use("default")


# %%
""" ===========================================================================

                                    Clustering

============================================================================"""

###############################################################################
def prepare_features(df_features,
                     scaler = "STANDARDSCALER"):

    # selec the scaler
    if scaler == "STANDARDSCALER":
        scaler = StandardScaler() #
    elif scaler == "ROBUSTSCALER":
        scaler = RobustScaler()
    elif scaler == "MINMAXSCALER" :
        scaler = MinMaxScaler()    
        
    # with shapes
    # create a vector with X in order to scale all features together
    X = pd.DataFrame()
    X = df_features.loc[:, df_features.columns.str.startswith("shp")]
    X_vect = X.to_numpy()
    X_shape = X_vect.shape
    X_vect = X_vect.reshape(X_vect.size, -1)
    X_vect = scaler.fit_transform(X_vect)
    X = X_vect.reshape(X_shape)

    # add other features like frequency centroid
    X2 = pd.DataFrame()
    X2 = df_features[
        [
            "centroid_f",
            # "peak_f",
            # "bandwidth_f",
            # "bandwidth_min_f",
            # "bandwidth_max_f",
            # "duration_t",
            # "min_f",
            # "max_f"
        ]
    ]
    
    # Preprocess data : data scaler
    X2 = scaler.fit_transform(X2)

    # create a matrix with all features after rescaling
    X = np.concatenate((X, X2), axis=1)
    
    return X
###############################################################################
def find_cluster(
        dataset,
        params=cfg.DEFAULT_PARAMS_CLUSTER,
        display=False,
        verbose=False):
    """

    Clustering of ROIs of each species

    We will use DBSCAN clustering method for several reasons :
        * DBSCAN does not need the number of clusters to do the clustering
        * DBSCAN is able to deal with noise and keep them outside any clusters.

    So, the goal of the clustering is to aggregate similar ROIs for a species 
    which might correspond to the main call or song of the bird. If several 
    clusters are found, which means that we might have ROIs corresponding to 
    different calls and/or songs for the species, we will keep the cluster with 
    the highest number of ROIs.

    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    params : TYPE, optional
        DESCRIPTION. The default is DEFAULT_PARAMS_CLUSTER.
    display : TYPE, optional
        DESCRIPTION. The default is False.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    df_cluster : TYPE
        DESCRIPTION.

    """

    if verbose :
        print('\n')
        print('====================== CLUSTER FEATURES ======================\n')
    
    # test if dataset not a dataframe
    if isinstance(dataset, pd.DataFrame) == False:
        # test if dataset_path is a valid csv file
        if os.path.isfile(dataset):
            # load the data from the csv
            df_features = pd.read_csv(dataset, sep=';')
                      
    elif isinstance(dataset, pd.DataFrame): 
        df_features = dataset.copy()
        
        # reset the index
        try:
            df_features.reset_index('filename_ts', inplace = True)
        except:
            pass     

    else:
        raise Exception(
            "WARNING: dataset must be a valid path to a csv file"
            + "or a dataframe"
        )

    # drop NaN rows
    df_features = df_features.dropna(axis=0)
    
    if display:
        # Prepare the plots
        fig, ax = plt.subplots(1, len(df_features.species.unique()))
        fig.set_size_inches(len(df_features.species.unique()) * 6, 5)

        fig2, ax2 = plt.subplots(1, len(df_features.species.unique()))
        fig2.set_size_inches(len(df_features.species.unique()) * 6, 5)

        fig3, ax3 = plt.subplots(1, len(df_features.species.unique()))
        fig3.set_size_inches(len(df_features.species.unique()) * 6, 5)
        
        # in case the number of species is 1, ax, ax2, ax3 must be converted
        # into list       
        try :
            len(ax)
        except:
            ax = [ax]

        try :
            len(ax2)
        except:
            ax2 = [ax2]
        
        try :
            len(ax3)
        except:
            ax3 = [ax3]

    # count the species
    count = 0

    # initialize df_cluster
    df_cluster = df_features[['filename_ts',
                              'fullfilename_ts',
                              # 'fullfilename', 
                              # 'filename',
                              'code',
                              'species',
                              # 'abs_min_t',
                              'min_f',
                              'min_t',
                              'max_f',
                              'max_t']]
    
    # add 3 others columns if they exist in df_features 
    # if the ROIs were extracted outside of this workflow, they might not exist
    # in the dataframe df_features
    if 'fullfilename' in df_features :
        df_cluster['fullfilename'] = df_features['fullfilename']
    if 'filename' in df_features :
        df_cluster['filename'] = df_features['filename']
    if 'abs_min_t' in df_features :
        df_cluster['abs_min_t'] = df_features['abs_min_t']
        
    # find the cluster for each species separately
    #------------------------------------------------------- 
    for species in np.sort(df_features.species.unique()):

        # select the ROIs of the current species
        df_single_species = df_features[df_features["species"] == species]
        
        # test if the number of ROIs is higher than 2.
        # If not, it is impossible to cluster ROIs. It requires at least 3 ROIS
        if len(df_single_species) <3 :
            df_cluster["cluster_number"] = -1 # noise
            df_cluster["auto_label"] = 0 # noise
            
            if verbose:
                print("Only {} ROIs. It requires at least 3 ROIs to perform clustering".format(
                        len(df_single_species)))
                
        else:

            # Prepare the features of that species
            #-------------------------------------------------------
            X = prepare_features(df_single_species, 
                                 scaler = params['SCALER'])
    
            if display:
                # Plot the features
                ax3[count].imshow(
                    X,
                    interpolation="None",
                    cmap="viridis",
                    vmin=np.percentile(X, 10),
                    vmax=np.percentile(X, 90),
                )
                ax3[count].set_xlabel("features")
                ax3[count].set_title("Shapes")
    
            # Select the minimum of points for a cluster
            #-------------------------------------------------------
            min_points = round(params["PERCENTAGE_PTS"] / 100 * len(df_single_species))
            if min_points < 2: min_points = 2  
    
            # automatic estimation of the maximum distance eps
            #-------------------------------------------------------
            if params["EPS"] == 'auto' :
                # Calculate the average distance between each point in the data set and
                # its N nearest neighbors (N corresponds to min_points).
                neighbors = NearestNeighbors(n_neighbors=min_points)
                neighbors_fit = neighbors.fit(X)
                distances, indices = neighbors_fit.kneighbors(X)
        
                # Sort distance values by ascending value and plot
                distances = np.sort(distances, axis=0)
                distances = distances[:, 1] 
        
                # find the knee (curvature inflexion point)
                kneedle = KneeLocator(
                    x=np.arange(0, len(distances), 1),
                    y=distances,
                    interp_method="polynomial",
                    # online = False,
                    # S=10,
                    curve="convex",
                    direction="increasing",
                )
        
                if display:
                    # plot the distance + the knee
                    ax2[count].set_xlabel("cumulative number of ROIs", fontsize=10)
                    ax2[count].set_ylabel("eps", fontsize=10)
                    ax2[count].axhline(y=kneedle.knee_y, xmin=0,xmax=len(distances), color="r")
                    ax2[count].set_title("sorted k-dist graph", fontsize=12)
                    ax2[count].plot(distances)
        
                # first find the maximum distance that corresponds to 95% of observations
                eps = kneedle.knee_y
                
                if eps == 0:
                    eps = distances.max()
    
            # set eps manually 
            #-------------------------------------------------------
            else :
                eps = params["EPS"]
    
            # find the number of clusters and the rois that belong to the cluster
            #--------------------------------------------------------------------
            if params["METHOD"] == "DBSCAN":
                cluster = DBSCAN(eps=eps, min_samples=min_points).fit(X)
                
                if verbose:
                    print("DBSCAN eps {} min_points {} Number of soundtypes found for {} : {}".format(eps, min_points,
                            species, np.unique(cluster.labels_).size))
                    
            elif params["METHOD"] == "HDBSCAN":
                cluster = hdbscan.HDBSCAN(
                    min_cluster_size=min_points,
                    min_samples=round(min_points / 2),
                    cluster_selection_epsilon=float(eps),
                    cluster_selection_method="eom",
                    allow_single_cluster=True,
                ).fit(X)
                
                if verbose:
                    print("HDBSCAN Number of soundtypes found for {} : {}".format(
                            species, np.unique(cluster.labels_).size))
    
                # metric DBCV
                """DBCV_score = DBCV.DBCV(X, cluster.labels_, dist_function=euclidean)
                
                print('Number of soundtypes found for {} : {} \
                      / DBCV score {:.2f} \
                      / DBCV relative {:.2f}'.format(species,
                                                   np.unique(cluster.labels_).size,
                                                   DBCV_score,
                                                   cluster.relative_validity_))"""

            # add the label found with the clustering
            #-------------------------------------------------------
            # add the cluster label into the label's column of the dataframe
            df_cluster.loc[df_cluster["species"] == species, "cluster_number"] = cluster.labels_.reshape(-1, 1)
    
            # add the automatic label (SIGNAL = 1 or NOISE = 0) into the auto_label's column of
            # the dataframe
            # Test if we want to consider only the biggest or all clusters 
            # that are not noise (-1) to be signal
            if params["KEEP"] == 'BIGGEST' :
                # set by default to 0 the auto_label of all
                df_cluster.loc[df_cluster["species"] == species, "auto_label"] = int(0)
                # find the cluster ID of the biggest cluster that is not noise
                try :
                    biggest_cluster_ID = df_cluster.loc[(df_cluster["species"] == species) & (
                                                     df_cluster["cluster_number"] >= 0)]["cluster_number"].value_counts().argmax()
                    # set by default to 1 the auto_label of the biggest cluster
                    df_cluster.loc[(df_cluster["species"] == species) & (
                                    df_cluster["cluster_number"] == biggest_cluster_ID), "auto_label"] = int(1)
                except:
                    # if there is only noise
                    pass
                
            elif params["KEEP"] == 'ALL' :
                # set by to 0 the auto_label of the noise (cluster ID = -1)
                df_cluster.loc[(df_cluster["species"] == species) & (
                                df_cluster["cluster_number"] < 0), "auto_label"] = int(0)
                # set by to 1 the auto_label of the signal (cluster ID >= 0)
                df_cluster.loc[(df_cluster["species"] == species) & (
                                df_cluster["cluster_number"] >= 0), "auto_label"] = int(1)
            
                        
            if display:
                # display the result in 2D (2D reduction of the dimension)
                pca = PCA(n_components=2)
                principalComponents = pca.fit_transform(X)
                df_PCA = pd.DataFrame(
                    data=principalComponents,
                    columns=["principal component 1", "principal component 2"],
                )
                ax[count].set_xlabel("PC 1", fontsize=10)
                ax[count].set_ylabel("PC 2", fontsize=10)
                ax[count].set_title(species, fontsize=12)
    
                ax[count].scatter(
                    df_PCA["principal component 1"],
                    df_PCA["principal component 2"],
                    c=cluster.labels_,
                    s=50,
                    alpha=0.8,
                )

        # increment
        count += 1

    return df_cluster

###############################################################################
def cluster_eval(df_cluster,
                 path_to_csv_with_gt,
                 colname_label    = 'auto_label' ,
                 colname_label_gt = 'manual_label',
                 verbose=False):

    fp_initial = []
    tp_initial = []
    precision_initial = []
    precision = []
    recall = []
    tp = []
    fp = []
    tn = []
    fn = []
    number_rois_initial = []
    number_rois_final = []
    
    df = df_cluster.copy()
    
    try : 
        # load all annotations
        df_labels = pd.read_csv(path_to_csv_with_gt, sep=';')
        try :
            df_labels.drop('species', axis=1, inplace=True)
        except:
            pass
        try : 
            df_labels.drop('species', axis=1, inplace=True)
        except:
            pass
        df_labels.set_index('filename_ts', inplace=True)
        df_labels.loc[df_labels[colname_label_gt] == '0', colname_label_gt] = 0
        df_labels.loc[df_labels[colname_label_gt] == '1', colname_label_gt] = 1
    
        # join df_label and df then drop rows with NaN
        if 'filename_ts' in df :
            df.set_index('filename_ts', inplace=True)    
        df = df.join(df_labels[colname_label_gt])
        df = df.dropna(axis=0)
        
    except :
        raise Exception("WARNING: path_to_csv_with_gt must be a valid path to a csv ")
        
    # Create a new column 'marker' with tp, tn, fp, fn
    df['marker'] = None
    #TP
    df.loc[(df[colname_label]==1) * (df[colname_label_gt]==1), 'marker'] = 'TP'
    #TN
    df.loc[(df[colname_label]==0) * (df[colname_label_gt]==0), 'marker'] = 'TN'
    #FP
    df.loc[(df[colname_label]==1) * (df[colname_label_gt]==0), 'marker'] = 'FP'
    #FN
    df.loc[(df[colname_label]==0) * (df[colname_label_gt]==1), 'marker'] = 'FN'

    # select Rois that belongs to the species depending on the clustering
    
    for species in np.sort(df.species.unique()):

        number_rois_initial += [len(df[df["species"] == species])]
        number_rois_final += [np.sum((df["species"] == species) & (df[colname_label] == 1))]

        fp_initial += [np.sum(df[df["species"] == species][colname_label_gt] == 0)]
        tp_initial += [np.sum( df[df["species"] == species][colname_label_gt] == 1)]

        precision_initial += [round(tp_initial[-1] / (tp_initial[-1] + fp_initial[-1]) * 100)]

        _tn, _fp, _fn, _tp = confusion_matrix(
            df.dropna()[df["species"] == species][colname_label_gt].to_list(),
            df.dropna()[df["species"] ==  species][colname_label].to_list()).ravel()
        
        tp += [_tp]
        fp += [_fp]
        tn += [_tn]
        fn += [_fn]

        if (_tp + _fp) > 0:
            precision += [round(_tp / (_tp + _fp) * 100)]
        else:
            precision += [0]
        if (_tp + _fn) > 0:
            recall += [round(_tp / (_tp + _fn) * 100)]
        else:
            recall += [0]

        if verbose:
            print(
                "Initial number of ROIs is {} / Final number of ROIs is {} => {}% reduction / noise {}% => {}%  / recall {}% ({})".format(
                    number_rois_initial[-1],
                    number_rois_final[-1],
                    round(100 - number_rois_final[-1] /
                          number_rois_initial[-1] * 100, 1),
                    100 - precision_initial[-1],
                    100 - precision[-1],
                    recall[-1],
                    species,
                )
            )

    # dataframe with scores
    df_scores = pd.DataFrame(list(zip(np.sort(df.species.unique()),
                                      number_rois_initial,
                                      number_rois_final,
                                      tp_initial,
                                      fp_initial,
                                      tp,
                                      fp,
                                      tn,
                                      fn,
                                      precision_initial,
                                      precision,
                                      recall)),
                             columns=['species',
                                      'number_rois_initial',
                                      'number_rois_final',
                                      'tp_initial',
                                      'fp_initial',
                                      'tp',
                                      'fp',
                                      'tn',
                                      'fn',
                                      'precision_initial',
                                      'precision',
                                      'recall'])
    # set species as index
    df_scores.set_index('species', inplace = True)
    
    if verbose:
        print("------------------------------------------------------")
        print("------->Median initial noise {:.1f}%".format(
            100-np.percentile(precision_initial, 50)))
        print("Lower outlier Initial noise  {:.1f}%".format(
            100-np.percentile(precision_initial, 95)))
        print("Higher outlier Initial noise {:.1f}%".format(
            100-np.percentile(precision_initial, 5)))
        print("------->  Median Final noise {:.1f}%".format(
            100-np.percentile(precision, 50)))
        print("Lower outlier Final noise    {:.1f}%".format(
            100-np.percentile(precision,95)))
        print("Higher outlier Final noise   {:.1f}%".format(
            100-np.percentile(precision,5)))
        print("------------------------------------------------------")
        # calculate the F1-SCORE (macro and micro)
        y_true = (df.species * df[colname_label_gt].apply(np.int64))
        y_pred = (df.species * df[colname_label].apply(np.int64))
        print("******************************************************")
        print("avg intial noise {:2.1f}% >>> avg final noise {:2.1f}%".format(100-np.mean(precision_initial),
                                                                    100-np.mean(precision)))
        p, r, f, _ = precision_recall_fscore_support( y_true, 
                                                      y_pred, 
                                                      average='macro')
        print("MACRO precision {:.2f} | recall {:.2f} | F {:.2f}".format(p,r,f))
        print("******************************************************")

    return df_scores, p, r, f, df.marker
      

###############################################################################
def overlay_rois (cluster,
                  markers = None,
                  column_labels ='auto_label',
                  unique_labels=[0,1],
                  color_labels=['tab:red', 'tab:green', 'tab:orange', 'tab:blue', 
                                 'tab:purple','tab:pink','tab:brown','tab:olive',
                                 'tab:cyan','tab:gray','yellow'],
                  textbox_label=True,
                  params=cfg.DEFAULT_PARAMS_EXTRACT,
                  filename=None,
                  random_seed=None,
                  verbose=False,
                  **kwargs):
    
    if verbose :
        print('\n')
        print('============== OVERLAY ROIS ON THE ORIGINAL FILE ==============\n')
    
    # test if dataset not a dataframe
    if isinstance(cluster, pd.DataFrame) == False:
        
        # test if dataset_path is a valid csv file
        if os.path.isfile(cluster):
            
            # load the data from the csv
            df_cluster = pd.read_csv(cluster, index_col=0)
            df_cluster.reset_index(inplace=True)
                      
    elif isinstance(cluster, pd.DataFrame): 
        df_cluster = cluster.copy()

    else:
        raise Exception(
            "***WARNING*** cluster must be a valid path to a csv file"
            + "or a dataframe"
        )
    
    # add the markers into the dataframe df_cluster if markers is not None
    if markers is not None :
        try:
            df_cluster.set_index('filename_ts', inplace = True)
        except:
            pass
        df_cluster = df_cluster.join(markers)
        df_cluster.reset_index(inplace=True)

    # select one xeno-canto file
    #---------------------------
    if "filename" in df_cluster :
        try:
            df_cluster.set_index(['filename'],inplace=True)
        except:
            pass
    else:
        # if there is no filename column but filename_ts, extract the filename
        if "filename_ts" in df_cluster :
            df_cluster['filename'] = df_cluster['filename_ts'].apply(lambda file : file.split('_',1)[0]+'.mp3')
            df_cluster.set_index(['filename'],inplace=True)
        else :
            raise Exception(
                "***WARNING*** cluster must have a column 'filename' or 'filename_ts' "
        )
    
    # if no filename given, pick a random one
    if filename is None :
        filename = df_cluster.sample(n=1, random_state=random_seed).index

    # extract the row corresponding to the filename
    df_single_file = df_cluster.loc[filename]

    # reset the index
    df_single_file.reset_index(inplace=True)
    df_cluster.reset_index(inplace=True)

    # Xeno-Canto filename
    fullfilename = df_single_file['fullfilename'].unique()[0]
    
    # add a label column
    df_single_file['label'] = df_single_file[column_labels]
    
    if verbose :
        print('Display ROIs found in the file {}'.format(fullfilename))
        print('labels : {}'.format(df_single_file.label.values))
    
    # Load the audio and add the bounding boxes
    #------------------------------------------
    try:
        # 1. load audio
        sig, sr = librosa.load(
            fullfilename, sr=params["SAMPLE_RATE"], duration=params["AUDIO_DURATION"]
        )
        
        fig = plt.figure(figsize=kwargs.pop("figsize", 
                                         (params["AUDIO_DURATION"]/60*15,7)))
        
        ax0 = plt.subplot2grid((5, 1), (0, 0), rowspan=1)
        ax1 = plt.subplot2grid((5, 1), (1, 0), rowspan=4)

        maad.util.plot_wave(s=sig, fs=sr, ax=ax0, now = False)
        
        # 2. compute the spectrogram
        Sxx, tn, fn, ext = maad.sound.spectrogram(
            sig,
            params["SAMPLE_RATE"],
            nperseg=params["NFFT"],
            noverlap=params["NFFT"] // 2,
        )
        
       
        df_single_file['min_t'] = df_single_file['min_t']+ df_single_file['abs_min_t']
        df_single_file['max_t'] = df_single_file['max_t']+ df_single_file['abs_min_t'] 
        df_single_file = maad.util.format_features(df_single_file, tn, fn)
        
        # 3.
        # Convert in dB
        X = maad.util.power2dB(Sxx, db_range=96) + 96

        kwargs.update({"vmax": np.max(X)})
        kwargs.update({"vmin": np.min(X)})
        kwargs.update({"extent": ext})
        maad.util.plot_spectrogram(X, 
                              log_scale=False, 
                              colorbar=False,
                              ax=ax1,
                              now = False,
                              **kwargs)
        
        # 4.
        if unique_labels is None :
            unique_labels = list(df_single_file[column_labels].unique())
        
        # overlay
        maad.util.overlay_rois(im_ref=X, 
                              rois = df_single_file,
                              ax=ax1,
                              fig=fig,
                              unique_labels=unique_labels,
                              edge_color=color_labels,
                              textbox_label=textbox_label)

        fig.suptitle((df_single_file.filename.unique()
                     + " " 
                     + df_single_file.species.unique()))
        fig.tight_layout()


    except Exception as e:
        if verbose:
            print("\nERROR : " + str(e))
    
    return filename

###############################################################################
def mark_rois (markers,
               dataset_csv,
               display = False,
               verbose = False):

    if verbose :
        print('\n')
        print('============= MARK ROIS with the prefix TP TN FP FN ============\n')

        
    # test if dataset_csv is a valid csv file
    if os.path.isfile(dataset_csv):
        # load the data from the csv
        df = pd.read_csv(dataset_csv, index_col=0, sep=';')
        df.reset_index(inplace=True)           
    else:
        raise Exception(
            "***WARNING*** dataset must be a valid path to a csv file")    
   
    # test if there is already a marker column.
    # If so, stop the process
    if 'marker' in df :
        marked = False
        if verbose:
            print('***WARNING*** ROIs are already marked.'
                  +' Unmarked ROIs before marking them again')
    
    else: 
        # join markers and dataset
        try :
            df.set_index('filename_ts', inplace=True)
            df = df.join(markers)
            df.reset_index(inplace=True)
        except:
            print('***WARNING*** No filename_ts column found in the dataset \n' +
                  '>>> None of the ROIs was marked')
            return
        
        # test if there is a column fullfilename_ts in the dataset
        if 'fullfilename_ts' and 'filename_ts' in df : 
            marked = True
            for idx, row in df.iterrows():
    
                f = Path(row['fullfilename_ts'])
                filename_ts = f.parts[-1]
                path = f.parent
                
                # select the marker to add to the file
                prefix = row['marker']
                # if the marker does not exit, add 'NA'
                if pd.isna(prefix) : prefix = 'NA'
    
                # rename the file with the marker
                f_new = path / Path(prefix + '_'+ filename_ts)
                os.rename(f, f_new)  
                
                # test if the file has been renamed
                if os.path.isfile(f_new) :
                    marked = marked and True
                    # add the new filename into the dataframe df
                    df.loc[idx,'fullfilename_ts'] = str(f_new)
                    df.loc[idx,'filename_ts'] = f_new.parts[-1]
                    
                else:
                    marked = False
                    if verbose :
                        print("***WARNING*** The ROIs with the filename_ts {} could not be renamed as {}".format(f, f_new))
                    
            try :
                df.set_index('filename_ts', inplace = True)
                df.to_csv(dataset_csv, sep=';')
                df.reset_index('filename_ts')
                if verbose :
                    print(" DONE \n" +
                          ">>> Save the dataframe with marked ROIs in {}".format(dataset_csv))
            except:
                if verbose :
                    print("***WARNING*** Dataframe with marked ROIs was not saved")
        else:
            marked = False
            if verbose :
                print('***WARNING*** Either fullfilename_ts or filename_ts do not exist in the dataframe. \n' +
                      '>>> None of the ROIs was marked')
    return df, marked

###############################################################################
def unmark_rois (dataset_csv,
                 display = False,
                 verbose = False):

    if verbose :
        print('\n')
        print('========================== UNMARK ROIS ========================\n')
    
    # test if dataset_csv is a valid csv file
    if os.path.isfile(dataset_csv):
        # load the data from the csv
        df = pd.read_csv(dataset_csv, index_col=0, sep=';')
        df.reset_index(inplace=True)         
    else:
        raise Exception(
            "***WARNING*** dataset must be a valid path to a csv file"
        )    
      
    # test if there is a columns fullfilename_ts filename_ts and marker in the dataset
    if ('fullfilename_ts' and 'filename_ts' and 'marker') in df : 
        unmarked = True
        for idx, row in df.iterrows():

            f = Path(row['fullfilename_ts'])
            filename_ts = f.parts[-1]
            path = f.parent
            
            # rename the file without the marker (-3 char)
            f_new = path / filename_ts[3:]
            os.rename(f, f_new)  
            
            # test if the file has been renamed
            if os.path.isfile(f_new) :
                unmarked = unmarked and True
                # add the new filename into the dataframe df
                df.loc[idx,'fullfilename_ts'] = str(f_new)
                df.loc[idx,'filename_ts'] = f_new.parts[-1]
                
            else:
                unmarked = False
                if verbose :
                    print("***WARNING*** The ROIs with the filename_ts {} could not be renamed as {}".format(f, f_new))
        
        # drop the column marker before saving
        df.drop('marker', axis=1, inplace=True)        
        
        try :
            df.set_index('filename_ts', inplace = True)
            df.to_csv(dataset_csv, sep=';')
            df.reset_index('filename_ts')
            if verbose :
                print(" DONE \n" +
                      ">>> Save the dataframe with unmarked ROIs in {}".format(dataset_csv))
        except:
            if verbose :
                print("***WARNING*** Dataframe with unmarked ROIs was not saved")
    else:
        unmarked = False
        if verbose :
            print('***WARNING*** Either fullfilename_ts or filename_ts or marker do not exist in the dataframe. \n' +
                  '>>> None of the ROIs was unmarked')
            
    return df, unmarked

    