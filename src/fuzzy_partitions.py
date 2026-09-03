#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:05:29 2023

@author: asier.urio
"""
import numpy as np

def generate_fuzzy_partitions_lims(attribs_lims, n_labels=3):
    """
    Generates a list with the triangular fuzzy partitions, each represented
    as [left bound, center, right bound]

    Parameters
    ----------
    attribs_lims : 2D Array
        Values of the the lower and upper bounds of the variables
    n_labels : int, optional
        Number of fuzzy terms for the variable. The default is 3.

    Returns
    -------
    partitions : 3D Array
        An arraycontaining the triangular membership functions data.
        [[[low_var1,...,x_var1],
         ...,
         [y_var1,...,high_var1]],
         ...,
         [low_varn,...,x_varn],
          ...,
          [y_varn,...,high_varn]]]
        For the case of only one variable in data the format is the same:
        [[[low_var1,...,x_var1],
         ...,
         [y_var1,...,high_var1]]]

    """
    max_values = attribs_lims[:, 1].copy().astype("float")
    min_values = attribs_lims[:, 0].copy().astype("float")
    half_support = (max_values - min_values) / (n_labels - 1)
    partitions = np.zeros((3, n_labels, len(attribs_lims)))
    center = min_values
    for i in range(n_labels):
        partitions[:, i] = np.array([center - half_support,
                                    center,
                                    center + half_support])
        center += half_support
    partitions = np.swapaxes(partitions, 0, 2)
    return partitions


def normalize_fuzzy_partitions(fp):
    xy_min = fp[:, 0, 0]
    xy_max = fp[:, -1, -1]
    xy_max_min = np.vstack((xy_min, xy_max)).T
    f = []
    for fpi, x_range in zip(fp, xy_max_min):
        f.append((fpi - x_range[0])/(x_range[1]-x_range[0]))
    return np.array(f)


def label_for_data_array(data, partitions):
    """
    Return the indices of the partition with higher membership value
    (for triangular partitions)

    Parameters
    ----------
    data : Array of arrays Float
        Array with values to compute the membership.
        [[var1_value1,...,varn_value1],
         ...,
         [var1_valuem,...,varn_valuem]]
    partitions : List of lists
        List of lists with the information of the  triangular membership functions.
        for each variable
        All partitions has to be same size
        [[[low_var1,...,x_var1],
         ...,
         [y_var1,...,high_var1]],
         ...,
         [low_varn,...,x_varn],
          ...,
          [y_varn,...,high_varn]]]

    Returns
    -------
    max_idx: Array of list of int
        Array with the indices of the set producing the highest membership value.
        [[var1_value1_label,...,varn_value1_label],
         ...,
         [var1_valuem_label,...,varn_valuem_label]]

        For the case of only one variable in data the format is the same:
            [[var1_value1_label],
             ...,
             [var1_valuem_label]]

    """
    if data.ndim == 1:
        data = np.reshape(data, (-1, 1))
    max_idx = np.zeros(data.shape)
    max_membership_values = np.zeros(data.shape)
    membership_values = np.zeros(data.shape)
    for i, partition in enumerate(np.swapaxes(partitions,0,1)):
        # If not swaped partition is all the sets for one variable...
        for j in range(len(partitions)):
            membership_values[:, j] = np.interp(data[:, j],
                                                partition[j], [0, 1, 0])
        tf = membership_values > max_membership_values
        max_membership_values = membership_values * tf + max_membership_values * np.invert(tf)
        max_idx = np.where(tf, i, max_idx)
    return max_idx


def membership_for_data_array(data, labels, partitions):
    """
    Return the membership values of the data for the given labels
    (for triangular partitions)

    Parameters
    ----------
    data : Array of arrays Float
        Array with values to compute the membership.
        [[var1_value1,...,varn_value1],
         ...,
         [var1_valuem,...,varn_valuem]]
    labels : Array of int
        Array with values to compute the membership.
        [label_var1,...,label_var_n]
    partitions : List of lists
        List of lists with the information of the  triangular membership functions.
        for each variable
        All partitions has to be same size
        [[[low_var1,...,x_var1],
         ...,
         [y_var1,...,high_var1]],
         ...,
         [low_varn,...,x_varn],
          ...,
          [y_varn,...,high_varn]]]

    Returns
    -------
    membership_values: Array of list of float
        Array with the membership values of the data for the given labels
        [[var1_value1_memb,...,varn_value1_memb],
         ...,
         [var1_valuem_memb,...,varn_valuem_memb]]

        For the case of only one variable in data the format is the same:
            [[var1_value1_memb],
             ...,
             [var1_valuem_memb]]

    """
    if data.ndim == 1:
        data = np.reshape(data,(-1,1))
    membership_values = np.zeros(data.shape)
    for i, partition in enumerate(partitions):
        label_idx = labels[i]
        # If not swaped partition is all the sets for one variable...
        for j in range(len(partitions)):
            membership_values [:,i] = np.interp(data[:,i],partition[label_idx],[0,1,0])
    return membership_values


def get_memberships(x, fp):
    '''
    Converts the actual real input value into an array of membership for
    each of the posible different partitons

    Parameters
    ----------
    x : Array of arrays (2D)
        An array of inputs, each one with n values.
    fp : Arrays of arrays (3D)
        An array for each input-output variable. In each another array for
        each partition. And finally, an array with the 3 points that define
        the triangle (for y=0, y=1, y=0)

    Returns
    -------
    mx : Array of arrays (3D)
        An array for the inputs but each one of them is replaced with an array
        with values for the membership for each posible partition.

    '''
    mx = np.array(list(map(
        lambda x1: list(map(
            lambda xi, fpi:
                # membership of one x value for each label in fpi
                list(map(
                    lambda e: np.interp(xi, e, [0, 1, 0]),
                    fpi)),
                x1, fp)), x)
            ))
    return mx

def get_y_sets(fp):
    xy_min = fp[-1][0][0]
    xy_max = fp[-1][-1][-1]
    xy_range = np.arange(xy_min, xy_max, (xy_max-xy_min)/100)
    # FIXME: precision 100 or other
    y_sets = np.array(list(map(np.interp,
                               [xy_range]*len(fp[-1]),
                               fp[-1],
                               [[0, 1, 0]]*len(fp[-1]))))
    return xy_range, y_sets


def get_x_ranges(fp):
    xy_min = fp[:, 0, 0]
    xy_max = fp[:, -1, -1]
    return np.vstack((xy_min, xy_max)).T

