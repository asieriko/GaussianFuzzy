#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:43:51 2023

@author: asier.urio
"""
import numpy as np
from src.fuzzy_partitions import (
    get_memberships,
    membership_for_data_array,
    label_for_data_array,
    )


def rule_str(rule):
    s = "IF "
    for i, ant in enumerate(rule[:-1]):
        if i == 0:
            s += f" x{i} is L_{ant} "
        else:
            s += f"AND x{i} is L_{ant} "
    s += f"THEN y is L_{rule[-1]}"
    return s


def mamdani(rules, x, fp, method='centroid'):
    rules = rules.astype(int)
    n = len(x)
    xy_min = fp[-1][0][0]
    xy_max = fp[-1][-1][-1]
    xy_range = np.arange(xy_min, xy_max, (xy_max-xy_min)/100)
    y = np.zeros(len(xy_range))
    yi = []
    ai = []
    for rule in rules:
        partitions = fp[np.arange(n+1), rule]
        m = []
        for xi, pi in zip(x, partitions[:-1]):
            m.append(np.interp(xi, pi, [0, 1, 0]))
            # TODO: Only one interp per consequent class, store and use
        m = np.min(m)
        # m = np.product(m)
        consequent = np.interp(xy_range, partitions[-1], [0, 1, 0])
        consequent = np.minimum(consequent, m)
        yi.append(np.sum(xy_range * consequent)/np.sum(consequent))
        ai.append(m) # compute area of consequent
        y = np.maximum(y, consequent)
    if np.sum(y) == 0:
        return 0
    return np.sum(xy_range * y)/np.sum(y)


# Rule Base generator

def wang_mendel(xy, fuzzy_partitions):
    rules = []
    imp_degree = []
    labels = label_for_data_array(xy, fuzzy_partitions).astype(int)
    importance_degree = np.array([np.prod(
        membership_for_data_array(np.array([xyi]), li, fuzzy_partitions))
        for xyi, li in zip(xy, labels)],
        dtype=object)
    input_spaces, is_reps = np.unique(labels[:, :-1], axis=0,
                                      return_counts=True)
    for ants in input_spaces:
        sel_idx = np.where(np.prod(labels[:, :-1]==ants, axis=1))
        selected_labels = labels[sel_idx]
        selected_importance_degree = importance_degree[sel_idx]
        max_importance_degree_idx = np.argmax(selected_importance_degree)
        selected_rule = selected_labels[max_importance_degree_idx]
        rules.append(selected_rule)
        imp_degree.append([selected_importance_degree[max_importance_degree_idx]])
    # rules = np.array(rules)
    return np.array(rules), np.array(importance_degree)


# Evaluation functions


def compute_RMSE(rules, xy, fp):
    mse = compute_MSE(rules, xy, fp)
    return (np.sqrt(mse)/np.mean(xy[:,-1]))*100

def compute_MSE(rules, xy, fp):
    MSE = 0
    for xyi in xy:
        MSE += (mamdani(rules, xyi[:-1], fp) - xyi[-1])**2
    MSE = MSE/len(xy)
    return MSE


# TODO: Next two functions in this file?
def get_rules_memberships(mx, rules):
    # Membership value of each input - subinput for each rule
    mx_rules = np.array(list(map(
        lambda r: mx[:, np.arange(len(r)), r],
        rules[:, :-1])))
    # Degree of each rule for each example
    mx_eachrule = np.prod(mx_rules, axis=2)
    # Degree of each example for each rule
    mx_eachexample = mx_eachrule.T
    return mx_eachexample


def get_rules_activation_degrees(mxy, rules):
    # Membership value of each input-output - subinput for each rule
    mx_rules = np.array(list(map(
        lambda r: mxy[:, np.arange(len(r)), r],
        rules)))
    # Degree of each rule for each example
    mx_eachrule = np.prod(mx_rules, axis=2)
    # FIXME: product or min ...
    # Degree of each example for each rule
    mx_eachexample = mx_eachrule.T
    return mx_eachexample
