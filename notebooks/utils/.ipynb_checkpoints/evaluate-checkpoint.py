#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:23:13 2022

@author: nicholas.lusk
"""

import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from pycm import ConfusionMatrix

# prediction and annotation overlap and build dataframe for validation metrics
# max distance a prediction can be off and still count as cell. very conservative at 1 px
def annot_pred_overlap(blocks, max_dist, type_2D = None, trained = False, get_keys = []):
    
    '''
    Parameters:
        blocks (dict): annotation data
        max_dist (int): distance in px from centroid of annotated cell to predicted cell to be considered the same
        type_2D (str): the method used for cellpose 3D. leave as None if using cellfinder
        train (bool): if a self trained model was used
        get_keys(list): the particular annotation blocks to evaluate. Leave blank if all will be used
        
    returns:
        metric_df (df): a dataframe containing location, block, true value, predicted value, and mask_id
    '''
    
    metric_df = pd.DataFrame(columns = ['x', 'y', 'z', 'block', 'annotated', 'predicted', 'mask_id'])
    
    if trained:
        if isinstance(type_2D, type(None)):
            type_key = ['train_detect', 'train_reject']
        else:
            type_key = ['train_detect_' + type_2D, 'train_reject_' + type_2D]    
    else:
        if isinstance(type_2D, type(None)):
            type_key = ['pred_detect', 'pred_reject']
        else:
            type_key = ['pred_detect_' + type_2D, 'pred_reject_' + type_2D]
    
    if len(get_keys) == 0:
        key_list = list(blocks.keys())
    else:
        key_list = get_keys
            
    for key in key_list:
    
        # get annotated data (not always a reject file)
        annot_detect = np.asarray(blocks[key]['detect'])
            
        try:
            annot_reject = np.asarray(blocks[key]['reject'])
        
            annot_points = np.concatenate((annot_detect, annot_reject), axis = 0)
            annot_types = np.concatenate((np.ones((len(annot_detect), 1)),
                                          np.ones((len(annot_reject), 1)) * 2), axis = 0)
        except:
            annot_points = annot_detect.copy()
            annot_types = np.ones((len(annot_detect), 1))

        # get predicted data
        pred_detect = np.asarray(blocks[key][type_key[0]])
        pred_reject = np.asarray(blocks[key][type_key[1]])
        
        try:
            detect_mask = np.asarray(blocks[key][type_key[0] + '_mask'])
            reject_mask = np.asarray(blocks[key][type_key[1] + '_mask'])
        except:
            detect_mask = np.zeros(pred_detect.shape[0])
            reject_mask = np.zeros(pred_reject.shape[0])
    
        if len(pred_reject) > 0:
            pred_points = np.concatenate((pred_detect, pred_reject), axis = 0)
            pred_types = np.concatenate((np.ones((len(pred_detect), 1)),
                                         np.ones((len(pred_reject), 1)) * 2), axis = 0)
            masks = np.concatenate((detect_mask,
                                    reject_mask), axis = 0)
        else:
            pred_points = pred_detect.copy()
            pred_types = np.ones((len(pred_detect), 1))
            masks = detect_mask.copy()
        
        # create annotation and prediction tree for comparison
        annot_tree = KDTree(annot_points)
        pred_tree = KDTree(pred_points)
    
        # returns: for each element in annot_tree[i], indexes[i] is a list of indecies within distance r from pred_tree
        indexes = annot_tree.query_ball_tree(pred_tree, r = max_dist)
        pred_id = np.zeros((len(annot_types), 1))
        mask_id = np.zeros((len(annot_types), 1))
        pred_extra = np.zeros((len(pred_types), 1))
        prev_pred = []
    
        # get the index and type for all annotated cells and id predicted cells that were not annotated
        for c, idx in enumerate(indexes):
            if len(idx) > 0 and idx[0] not in prev_pred:
                pred_id[c] = pred_types[idx[0]]  
                pred_extra[idx[0]] = 1
                mask_id[c] = masks[idx[0]]
                prev_pred.append(idx[0])
    
        pred_id[pred_id == 0] = 3
        
        data_array = np.concatenate((annot_points,
                                    np.ones((len(annot_points), 1)) * int(key),
                                    annot_types,
                                    pred_id,
                                    mask_id), axis = 1)
    
        # get location and type of predicted cells that where not annotated
        pred_extra_loc, _ = np.where(pred_extra == 0)
        if len(pred_extra_loc) > 0:
            curr_points = pred_points[pred_extra_loc[:], :]
            curr_types = pred_types[pred_extra_loc[:], :]
            pred_array = np.concatenate((curr_points,
                                        np.ones((len(curr_points), 1)) * int(key),
                                        np.ones((len(curr_points), 1)) * 3,
                                        curr_types,
                                        np.zeros((len(curr_points), 1))), axis = 1)

            # add to annot array
            data_array = np.vstack((data_array, pred_array))

        # create dataframes
        curr_df = pd.DataFrame(data_array, columns = ['x', 'y', 'z', 'block', 'annotated', 'predicted', 'mask_id'])         
        metric_df = pd.concat((metric_df, curr_df))
    return metric_df

# Identify prediction and annotation overlap and build dataframe for validation metrics
# Key: 1: signal that is detected and considered to be a cell
#      2: signal that is detected but found to be noise and rejected
#      3: singal that is annotated but not predicted to be signal
def get_performance(blocks, type_2D, max_dist = 10, trained = False, get_keys = []):
    
    '''
    Parameters:
        blocks (dict): annotation data
        type_2D (list): the method(s) used for cellpose 3D
        max_dist (int): distance in px from centroid of annotated cell to predicted cell to be tested
        train (bool): If a self trained model was used
        get_keys(list): the particular annotation blocks to evaluate. Leave blank if all will be used
    '''
    
    #since using different algorithm more flexibility in distance may be needed
    perf_metrics = []

    for i in range(max_dist):
        dist = i + 1
        
        if len(type_2D) == 1:
            df = annot_pred_overlap(blocks, dist, type_2D[0], trained, get_keys)
            cm = ConfusionMatrix(df['annotated'].values, df['predicted'].values, classes = [1.0, 2.0, 3.0])
            perf_metrics.extend([[cm.class_stat['PPV'][1.0], dist],
                                [cm.class_stat['TPR'][1.0], dist],
                                [cm.class_stat['F1'][1.0], dist]])
        
        else:
            df_1 = annot_pred_overlap(blocks, dist, type_2D[0], trained, get_keys)
            df_2 = annot_pred_overlap(blocks, dist, type_2D[1], trained, get_keys)
            cm_1 = ConfusionMatrix(df_1['annotated'].values, df_1['predicted'].values, classes = [1.0, 2.0, 3.0])
            cm_2 = ConfusionMatrix(df_2['annotated'].values, df_2['predicted'].values, classes = [1.0, 2.0, 3.0])
            perf_metrics.extend([[cm_1.class_stat['PPV'][1.0], cm_2.class_stat['PPV'][1.0], dist],
                                [cm_1.class_stat['TPR'][1.0], cm_2.class_stat['TPR'][1.0], dist],
                                [cm_1.class_stat['F1'][1.0], cm_2.class_stat['F1'][1.0], dist]])

        # ignores numpy.where() warning
        warnings.simplefilter(action = 'ignore', category = FutureWarning)

        performance = np.asarray(perf_metrics)
        performance = np.where(performance == 'None', 0, performance)
            
    return performance

#========================================================
# Plot functions
#========================================================

# plots the performance of the flow model as the distance from annotated centroid is varied
def plot_performance(performance, methods):
     
    '''
    Parameters:
        performance (array): output from get_performance() function
        methods (list): the method(s) used for cellpose 3D
    '''
    
    colors = ['r', 'g', 'b']
    mets = ['Percision', 'Recall', 'F1 Score']
    
    if len(methods) == 1:
        opts = np.argmax(performance[2::3, 0])
        offset = 1
        
        if isinstance(methods[0], type(None)):
            methods[0] = 'cellfinder'
        
    else:     
        opts = [np.argmax(performance[2::3, 0]), np.argmax(performance[2::3, 1])]
        offset = 2

    fig, ax = plt.subplots(1, offset, figsize = (4 * offset, 4))
    fig.suptitle('Distance between annotated and predicted Cell')
    if offset == 1:
        for i in range(3):
            sns.lineplot(x = performance[i::3, 1], y = performance[i::3, 0], color = colors[i], label = mets[i], ax = ax, zorder = i+2)
            ax.set_xlabel('Max distance (px)')
            ax.set_ylabel('Metric score')
            ax.set_title(methods[0] + ' detection')
            ax.set_ylim(0.2, 1)
            ax.axvline(x = performance[2::3, 1][opts], color = 'k', linestyle = '--', zorder = 1)
    
    else:                 
        for c, method in enumerate(methods):
            for i in range(3):
                sns.lineplot(x = performance[i::3, 2], y = performance[i::3, c], color = colors[i], label = mets[i], ax = ax[c], zorder = i+2)
                ax[c].set_xlabel('Max distance (px)')
                ax[c].set_ylabel('Metric score')
                ax[c].set_title(method + ' detection')
                ax[c].set_ylim(0.2, 1)
                ax[c].axvline(x = performance[2::3, 2][opts[c]], color = 'k', linestyle = '--', zorder = 1)
    if len(methods) == 1:    
        print('Top F1 Scores for {0} detection: {1}'.format(methods[0], 
                                                            performance[2::3, 0][opts]))
    else:
        print('Top F1 Scores for {0} detection: {1} and {2} detection: {3}'.format(methods[0],
                                                                                   performance[2::3, 0][opts[0]], 
                                                                                   methods[1],
                                                                                   performance[2::3, 1][opts[1]]))
    return

# plot confusion matrix for cell detection
def plot_cm(metric_df):

    '''
    Parameters:
        metric_df (dataframe): output from annot_pred_overlap() function
    '''
    
    # ignores numpy.where() warning
    warnings.simplefilter(action = 'ignore', category = FutureWarning)

    # Plot cumulative, Row normalized, and Performance for each block as well as total
    for i in range(int(metric_df['block'].max()) + 1):
        fig, ax = plt.subplots(1, 3, figsize = (10, 3))
    
        if i < metric_df['block'].max():
            block_df = metric_df.loc[metric_df['block'] == i + 1, :]
            cm = ConfusionMatrix(block_df['annotated'].values, block_df['predicted'].values, classes = [1.0, 2.0, 3.0])
            fig.suptitle('Confusion matricies: Block ' + str(i + 1))
        else:
            cm = ConfusionMatrix(metric_df['annotated'].values, metric_df['predicted'].values, classes = [1.0, 2.0, 3.0])
            fig.suptitle('Confusion matricies: Total')
    
        for j in range(3):
            if j == 0:
                cm_df = pd.DataFrame(cm.to_array(normalized = False), columns = ['cell', 'noncell', 'not_pred'], 
                                     index = ['cell', 'noncell', 'not_annot'])
                ax[j].set_title('Cumulative')
            elif j == 1:
                cm_df = pd.DataFrame(cm.to_array(normalized = True), columns = ['cell', 'noncell', 'not_pred'], 
                                     index = ['cell', 'noncell', 'not_annot'])
                ax[j].set_title('Normalized (by row)')
            else:
                perf_array = np.asarray([[cm.class_stat['PPV'][k], cm.class_stat['TPR'][k], cm.class_stat['F1'][k]] for k in [1.0, 2.0]])
                perf_array = np.where(perf_array == 'None', 0, perf_array)
                cm_df = pd.DataFrame(perf_array.astype(float), columns = ['Precision', 'Recall', 'F1 Score'], index = ['cell', 'noncell'])
                ax[j].set_title('Performance')
            
            sns.heatmap(cm_df, annot = True, cmap = 'coolwarm', fmt = '.4g', annot_kws = {'fontweight': 'heavy'}, ax = ax[j])
        
            if j == 0:
                ax[j].set_ylabel('Annotated label')
        
            if j != 2:
                ax[j].set_xlabel('Predicted label')
            else:
                ax[j].set_xlabel('Performance metric')
            
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    return

def plot_error_loc():
    return

def plot_error_PSF():
    return