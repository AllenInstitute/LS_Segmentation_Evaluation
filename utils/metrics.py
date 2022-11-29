#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:56:58 2022

@author: nicholas.lusk
"""
import warnings

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pycm import ConfusionMatrix
from utils import annot_pred_overlap

# Identify prediction and annotation overlap and build dataframe for validation metrics
# Key: 1: signal that is detected and considered to be a cell
#      2: signal that is detected but found to be noise and rejected
#      3: singal that is annotated but not predicted to be signal
def get_performance(data, method, trained, get_keys = []):
    #since using different algorithm more flexibility in distance may be needed
    perf_metrics = []

    for i in range(10):
        dist = i + 1
        flow_df = annot_pred_overlap(data, dist, method, trained, get_keys)
        cm_flow = ConfusionMatrix(flow_df['annotated'].values, flow_df['predicted'].values, classes = [1.0, 2.0, 3.0])
        perf_metrics.extend([[cm_flow.class_stat['PPV'][1.0], dist],
                            [cm_flow.class_stat['TPR'][1.0], dist],
                            [cm_flow.class_stat['F1'][1.0], dist]])

    # ignores numpy.where() warning
    warnings.simplefilter(action = 'ignore', category = FutureWarning)

    performance = np.asarray(perf_metrics)
    performance = np.where(performance == 'None', 0, performance)
    return performance

def get_error_loc():
    return

# plots the performance of the flow model as the distance from annotated centroid is varied
def plot_performance(performance, method = ''):
    colors = ['r', 'g', 'b']
    mets = ['Percision', 'Recall', 'F1 Score']

    opt_flow = np.argmax(performance[2::3, 0])

    fig, ax = plt.subplots(figsize = (4, 4))
    fig.suptitle('Distance between annotated and predicted Cell')

    for i in range(3):
        sns.lineplot(x = performance[i::3, 1], y = performance[i::3, 0], color = colors[i], label = mets[i], ax = ax, zorder = i+2)
        ax.set_xlabel('Max distance (px)')
        ax.set_ylabel('Metric score')
        ax.set_title(method + ' detection')
        ax.set_ylim(0.2, 1)
    
        ax.axvline(x = performance[2::3, 1][opt_flow], color = 'k', linestyle = '--', zorder = 1)

        
    sns.despine()
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    print('Top F1 Scores for flow detection: {0}'.format(performance[2::3, 0][opt_flow]))



def plot_error_loc():
    return