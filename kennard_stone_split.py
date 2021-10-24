# -*- coding=utf-8 -*-

"""
train test split implementation of kennard-stone with distance metric option
adapted from https://hxhc.github.io/post/kennardstone-spxy/
"""

from __future__ import division
import numpy as np
from scipy.spatial.distance import cdist

#TODO make verbose, combine into class

def kennardstone(spectra, test_size=0.25, metric='euclidean', *args, **kwargs):
    """Kennard Stone Sample Split method
    Parameters
    ----------
    spectra: ndarray, shape of i x j
        i spectrums and j variables (wavelength/wavenumber/ramam shift and so on)
    test_size : float, int
        if float, then round(i x (1-test_size)) spectrums are selected as test data, by default 0.25
        if int, then test_size is directly used as test data size
    metric : str, optional
        The distance metric to use, by default 'euclidean'
        See scipy.spatial.distance.cdist for more infomation
    Returns
    -------
    select_pts: list
        index of selected spetrums as train data, index is zero based
    remaining_pts: list
        index of remaining spectrums as test data, index is zero based
    References
    --------
    Kennard, R. W., & Stone, L. A. (1969). Computer aided design of experiments.
    Technometrics, 11(1), 137-148. (https://www.jstor.org/stable/1266770)
    """
    print(f'sampling data of {len(spectra)} rows')
    if test_size < 1:
        train_size = round(spectra.shape[0] * (1 - test_size))
    else:
        train_size = spectra.shape[0] - round(test_size)

    if train_size > 2:
        print(f'calculating distance using {metric}')
        distance = cdist(spectra, spectra, metric=metric, *args, **kwargs)
        select_pts, remaining_pts = max_min_distance_split(distance, train_size)
    else:
        raise ValueError("train sample size should be at least 2")

    return select_pts, remaining_pts


def max_min_distance_split(distance, train_size):
    """sample set split method based on maximun minimun distance, which is the core of Kennard Stone
    method
    Parameters
    ----------
    distance : distance matrix
        semi-positive real symmetric matrix of a certain distance metric
    train_size : train data sample size
        should be greater than 2
    Returns
    -------
    select_pts: list
        index of selected spetrums as train data, index is zero-based
    remaining_pts: list
        index of remaining spectrums as test data, index is zero-based
    """

    select_pts = []
    remaining_pts = [x for x in range(distance.shape[0])]

    # first select 2 farthest points
    first_2pts = np.unravel_index(np.argmax(distance), distance.shape)
    select_pts.append(first_2pts[0])
    select_pts.append(first_2pts[1])
    # remove the first 2 points from the remaining list
    remaining_pts.remove(first_2pts[0])
    remaining_pts.remove(first_2pts[1])

    for i in range(train_size - 2):
        if i % 100 == 0:
            print(f'{i} samples selected')
        # find the maximum minimum distance
        select_distance = distance[select_pts, :]
        min_distance = select_distance[:, remaining_pts]
        min_distance = np.min(min_distance, axis=0)
        max_min_distance = np.max(min_distance)

        # select the first point (in case that several distances are the same, choose the first one)
        points = np.argwhere(select_distance == max_min_distance)[:, 1].tolist()
        for point in points:
            if point in select_pts:
                pass
            else:
                select_pts.append(point)
                remaining_pts.remove(point)
                break
    return select_pts, remaining_pts


def ks_split(df, test_size=0.25, metric='mahalanobis', spectra_colrange=None):
    """
    Compilation for kennard-stone sampling algorithm
    :param df: (dataframe) n x d -dimension dataframe of spectral data to sample
    :param test_size: (float) fraction of samples to allocate to testing set. Default = 0.25
    :param metric: (str) distance metric with which to measure distance between points (plese see kennardstone function
    for further detail)
    :param spectra_colrange: (list|None) a 1 x 2 -dimension list indicating the numerical index of  start and end columns
    for the spectral data. If None, spectra_colrange = [10, -1]. Default = None
    :return: 2 dataframes, split for training and testing
    """
    # determine the column range for spectra
    if spectra_colrange is None:
        spectra_colrange= [10, -1]
    # convert numerical spectra into np.array for processing by kennard stone
    df_matrix = np.array(df.iloc[:, spectra_colrange[0]: spectra_colrange[1]])
    # obtain indices of train/test subsets using kennard stone algorithm
    train_idx, test_idx = kennardstone(df_matrix, test_size=test_size, metric=metric)
    # subset train/test data by provided indices
    train = df.iloc[train_idx, :]
    test = df.iloc[test_idx, :]
    return train, test

