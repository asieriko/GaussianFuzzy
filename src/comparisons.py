#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:51:05 2023

@author: asier.urio
"""
import numpy as np

from src.fuzzy_partitions import (
    get_memberships,
    get_x_ranges,
    get_y_sets,
    normalize_fuzzy_partitions,
    )

from src.mamdani import (
    get_rules_activation_degrees,
    )


import numpy as np

def jaccard(set1, set2):
    # Jaccard index
    # TODO: Other FS, T2...
    universe = [min(set1[0],set2[0]), max(set1[-1],set2[-1])]
    s1 = discretize(universe,set1)
    s2 = discretize(universe,set2)
    intersection = sum(np.minimum(s1,s2))
    union = sum(np.maximum(s1,s2))
    if union == 0.0:
        return 0
    return intersection/union

def discretize(u, fset, n_points=100, shape=[0,1,0]):  # shape because the domain limits ...
    return np.interp(np.linspace(u[0],u[1],n_points), fset, shape)


def antecedent_comparison(r1, r2, fp, tnorm=min):
    """
    Compares two list of fuzzy sets:
        - If the antecedent does not appear in either rules, the comparison of these is not taken into account
        - If one of the antecedents is missing, the comparison of these is not taken into account
        - If the two antecedents are fuzzy sets, the jaccarad index is computed between them
        - If only conditions 1 and 2 are meet, then the comparison is 0
    """
    acum = []
    for a1,a2,fpi in zip(r1,r2,fp):
        # For cases whe ai is a fuzzy set or -1 if not used
        # if len(a1) == 1 or len(a2) == 1:
        #     continue
        # else: #if len(a1) != 1 and len(a2) != 1:
        acum.append(jaccard(fpi[a1], fpi[a2]))
    if acum:
        return tnorm(acum)
    else:
        return 0


def comparison(r1, r2, fp, tnorm=min):
    """
    r1 and r2: arrays of index of fuzzy partition
    fp: fuzzy partitions
    Compares two fuzzy rules as follows:
        - If the consequences are different, the comparison is 0
        - If the antecedent does not appear in either rules, the comparison of these is 1
        - If one of the antecedents is missing, the jaccard is made between the other and the
        set formed by 1, for all the universe
        - If the two antecedents are fuzzy sets, the jaccarad index is computed between them
        - Thre return value is the minimun of all the pairwise comparisons
    """
    # Classification:
    if r1[-1] != r2[-1]:
        return 0
    comp = antecedent_comparison(r1[0],r2[0], fp, tnorm)
    # I return this because I use min in the aggregation of the antecedents and
    # as the consequences are equal their comparison is 1, so min(comp,1)=comp
    return comp

def compare_rule_base(rb, fp, tnorm=min):
    comparison = np.ones([len(rb),len(rb)])
    for i, rule1 in enumerate(rb):
        for j, rule2 in enumerate(rb):
            if j >= i:
                continue
            comp = antecedent_comparison(rule1, rule2, fp, tnorm)
            comparison[i][j] = comparison[j][i] = comp
    return comparison
