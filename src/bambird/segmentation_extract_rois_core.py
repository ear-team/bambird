#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Collection of functions to segment the sound of interests (also called regions 
of interest (ROIs) in time and frequency.
"""
#
# Authors:  Felix MICHAUD   <felixmichaudlnhrdt@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
#
# License: New BSD License

#%%

# basic packages
import numpy as np
import matplotlib.pyplot as plt

# scikit-image (image processing) package
from skimage.transform import resize

# Scikit-Maad (ecoacoustics functions) package
import maad

# import bambird
from bambird import config as cfg

#%%
###############################################################################
def _intersection_bbox(bb1, bb2):
    """
    test if 2 bounding-boxes intersect.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    boolean

    """

    # test if values are consistant (min < max)
    assert bb1["min_x"] <= bb1["max_x"]
    assert bb1["min_y"] <= bb1["max_y"]
    assert bb2["min_x"] <= bb2["max_x"]
    assert bb2["min_y"] <= bb2["max_y"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["min_x"], bb2["min_x"])
    y_top = max(bb1["min_y"], bb2["min_y"])
    x_right = min(bb1["max_x"], bb2["max_x"])
    y_bottom = min(bb1["max_y"], bb2["max_y"])

    if (x_right < x_left) or (y_bottom < y_top):
        is_intersected = False
    else:
        is_intersected = True

    return is_intersected

###############################################################################
def _fusion_bbox(df_rois, idx, idx2):
    """[fusion beginning and ends of 2 events that are too close to each other. It is applied if
    the 2 beginnings or the 2 ends are too close, or if the end of the first one is too close to the beginning
    of the second event. This function test all the events that follow each other in the dataframe.]

    Args:
        df_rois : 
            dataframe with all bbox to test for merging
        idx1 : 
            position of the FIRST bbox in the dataframe
        idx2 : 
            position of the SECOND bbox in the dataframe

    Return       df_rois: 
            returns the original dataframe with merged bbox
    """
    bb1 = df_rois.loc[idx]
    bb2 = df_rois.loc[idx2]

    min_t = min(bb1.min_t, bb2.min_t)
    max_t = max(bb1.max_t, bb2.max_t)
    min_f = min(bb1.min_f, bb2.min_f)
    max_f = max(bb1.max_f, bb2.max_f)

    df_rois.loc[idx, ["min_f", "min_t", "max_f", "max_t"]] = [
        min_f,
        min_t,
        max_f,
        max_t,
    ]

    min_x = min(bb1.min_x, bb2.min_x)
    max_x = max(bb1.max_x, bb2.max_x)
    min_y = min(bb1.min_y, bb2.min_y)
    max_y = max(bb1.max_y, bb2.max_y)

    df_rois.loc[idx, ["min_y", "min_x", "max_y", "max_x"]] = [
        min_y,
        min_x,
        max_y,
        max_x,
    ]

    df_rois = df_rois.drop(index=idx2)

    return df_rois

###############################################################################
def _merge_bbox(df_rois, margins):
    """[Merge two bbox that are within the margin. As the process is iterative
    if the resulting bbox is also close to another bbox, then it is merged. 
    And so on]

    Args:
        df_rois : 
            dataframe with all bbox to test for merging
        margins : 
            array with margins before/after the bbox and upper/lower the bbox

    Returns:
        df_rois: 
            returns the original dataframe with merged bbox
    """

    # loop until all bbox are merged
    # test if the number of rois is stable. if not, then run again the merging
    # process.
    num_rois = len(df_rois) + 1
    while len(df_rois) < num_rois:
        num_rois = len(df_rois)
        for idx, row in df_rois.iterrows():
            # test if the index idx is still in the dataframe
            if idx in df_rois.index:
                # remove the current ROIs (to avoid auto-merge)
                df_rois_without_current_row = df_rois.drop(idx)
                # add the margins to the bbox
                bb1 = row[["min_x", "min_y", "max_x", "max_y"]] + margins
                # loop to all the other ROIs
                for idx2, row2 in df_rois_without_current_row.iterrows():
                    # add the margins to the bbox
                    bb2 = row2[["min_x", "min_y", "max_x", "max_y"]] + margins
                    # If intersection => merge
                    if _intersection_bbox(bb1, bb2):
                        df_rois = _fusion_bbox(df_rois, idx, idx2)
                    # test if length of df_rois >1
                    if len(df_rois) <= 1:
                        break

    return df_rois

#%%
###############################################################################
def extract_rois_core(
    sig,
    params=cfg.PARAMS_EXTRACT,
    display=False,
    verbose=False,
    **kwargs):
    
    """ Extract all Rois in the audio file
    Parameters
    ----------
    audio_path : TYPE
        DESCRIPTION. 
    params : dictionnary
        contains all the parameters to extract the rois 
    display : TYPE, optional
        DESCRIPTION. The default is False.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.
    Returns
    -------
    df_rois : TYPE
        DESCRIPTION.
    """
    
    # 1. compute the spectrogram
    Sxx, tn, fn, ext = maad.sound.spectrogram(
                                    sig,
                                    params["SAMPLE_RATE"],
                                    nperseg=params["NFFT"],
                                    noverlap=params["NFFT"] // 2)

    t_resolution = tn[1] - tn[0]
    f_resolution = fn[1] - fn[0]

    if verbose:
        print("time resolution {}s".format(t_resolution))
        print("frequency resolution {}s".format(f_resolution))

    if display:

        # creating grid for subplots
        fig = plt.figure()
        fig.set_figheight(5)
        fig.set_figwidth(13)
        ax0 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=1)
        ax1 = plt.subplot2grid(shape=(2, 4), loc=(1, 0), colspan=1)
        ax2 = plt.subplot2grid(shape=(2, 4), loc=(0, 1), colspan=1)
        ax3 = plt.subplot2grid(shape=(2, 4), loc=(1, 1), colspan=1)
        ax4 = plt.subplot2grid(shape=(2, 4), loc=(0, 2), rowspan=2, colspan=2)

        maad.util.plot_spectrogram(
                        Sxx,
                        extent=ext,
                        ax=ax0,
                        title="1. original spectrogram",
                        interpolation=None,
                        now=False)

    # 2. reduce the size of the spectrogram Sxx
    # Both parameters can be adapted to the audio signature that we want to select
    Sxx_reduced = resize(
                    Sxx, 
                    (Sxx.shape[0] // params["FACTOR_F"], 
                     Sxx.shape[1] // params["FACTOR_T"]),
                    anti_aliasing=True)
    
    if display:
        maad.util.plot_spectrogram(
                            Sxx_reduced,
                            log_scale=True,
                            extent=ext,
                            ax=ax1,
                            title="2. Resized spectrogram",
                            interpolation=None,
                            now=False)

    # 3. Clean spectrogram : remove background)
    Sxx_clean_reduced, _ = maad.sound.remove_background_along_axis(
                                                    Sxx_reduced,
                                                    mode=params["MODE_RMBCKG"],
                                                    N=params["N_RUNNING_MEAN"],
                                                    display=False)
    
    # 4. convert to dB
    Sxx_clean_dB_reduced = (maad.util.power2dB(Sxx_clean_reduced, db_range=96) +96)

    if display:
        maad.util.plot_spectrogram(
                        Sxx_clean_dB_reduced,
                        extent=ext,
                        log_scale=False,
                        ax=ax2,
                        title="3. cleaned spectrogram",
                        interpolation=None,
                        now=False)

    # reduce time and frequential vectors
    if params["FACTOR_T"] % 2 == 0:  # is even
        tn_reduced = (tn[np.arange(params["FACTOR_T"] // 2, 
                                   len(tn), 
                                   params["FACTOR_T"])]
                      + t_resolution/2)
    else:  # is odd
        tn_reduced = tn[np.arange(params["FACTOR_T"] // 2,
                                  len(tn), 
                                  params["FACTOR_T"])]

    if params["FACTOR_F"] % 2 == 0:  # is even
        fn_reduced = (fn[ np.arange(params["FACTOR_F"] // 2, 
                                    len(fn), 
                                    params["FACTOR_F"])]
                      + f_resolution/2)
    else:  # is odd
        fn_reduced = fn[np.arange(params["FACTOR_F"] // 2,
                                  len(fn), 
                                  params["FACTOR_F"])]

    # 7. binarization of the spectrogram to select part of the spectrogram with
    # acoustic activity
    # Both parameters can be adapted to the situation in order to take more
    # or less ROIs that are more or less large
    im_mask = maad.rois.create_mask(
                        Sxx_clean_dB_reduced,
                        mode_bin="absolute",
                        bin_h=params["MASK_PARAM1"],
                        bin_l=params["MASK_PARAM2"])

    if display:
        maad.util.plot_spectrogram(
                        im_mask,
                        extent=ext,
                        ax=ax3,
                        title="4. mask",
                        interpolation=None,
                        now=True)

    # 8. get the mask with rois (im_rois) and the bounding box for each rois (rois_bbox)
    # and an unique index for each rois => in the pandas dataframe rois
    _, df_rois = maad.rois.select_rois(im_mask, min_roi=None)
 
    # and format ROis to reduced tn and fn
    df_rois = maad.util.format_features(df_rois, tn_reduced, fn_reduced)
    # and format ROis to initial tn and fn
    df_rois = maad.util.format_features(df_rois, tn, fn)

    # Remove cut ROIs (begining and end of audio) and 1 line rois (vertical
    # and horizontal )
    # Drop ROIs with same min_x max_x and min_y max_y (=> 1 line)
    df_rois = df_rois[df_rois.min_x < df_rois.max_x]
    df_rois = df_rois[df_rois.min_y < df_rois.max_y]

    # Drop two columns
    df_rois = df_rois.drop(
        columns=["labelID", "label"])

    if verbose:
        print("\nBEFORE MERGING FOUND {} ROIS ".format(len(df_rois)))

    # Test if we found an ROI otherwise we pass to the next chunk
    if len(df_rois) > 0:

        if len(df_rois) > 1:
            # 9. Merge ROIs that are very close to each other.
            # Default, if marings = [0,0,0,0], bbox must overlap in order
            # to be merge. If margins is not null, all bbox are expanded
            # according to margins. If the expanded bbox overlapped, they
            # are merged.
            # The process is iterative which means than several ROIs can be
            # merged after several pass
            margins = [
                -round(params["MARGIN_T_LEFT"] / t_resolution),
                -round(params["MARGIN_F_BOTTOM"] / f_resolution),
                 round(params["MARGIN_T_RIGHT"] / t_resolution),
                 round(params["MARGIN_F_TOP"] / f_resolution)]
            df_rois = _merge_bbox(df_rois, margins)

        # Keep only events with duration longer than MIN_DURATION
        df_rois = df_rois[((df_rois["max_t"] - df_rois["min_t"]) > params["MIN_DURATION"])]

        if verbose:
            print("=> AFTER MERGING FOUND {} ROIS".format(len(df_rois)))

        if display:
            # Convert in dB
            X = maad.util.power2dB(Sxx, db_range=96) + 96
            kwargs = {"vmax": np.max(X)}
            kwargs.update({"vmin": np.min(X)})
            kwargs.update({"extent": ext})
            kwargs.update({"figsize": (1, 2.5)})
            maad.util.plot_spectrogram(X, ext, 
                                  log_scale=False, 
                                  ax=ax4, 
                                  title="5. Overlay ROIs")
            maad.util.overlay_rois(X, df_rois, 
                              ax=ax4, fig=fig,
                              edge_color='yellow',
                              **kwargs)
            kwargs.update({"ms": 4, "marker": "+", "fig": fig, "ax": ax4})
            # ax, fig = maad.util.overlay_centroid(X, df_rois, **kwargs)
            fig.suptitle(kwargs.pop("suptitle", ""))
            fig.tight_layout()

    return df_rois