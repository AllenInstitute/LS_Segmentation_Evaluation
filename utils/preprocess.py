#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:23:13 2022

@author: nicholas.lusk
"""

import os
import re
import importlib

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from scipy.spatial import KDTree
from scipy import ndimage as ndi
from astropy.stats import SigmaClip
from photutils.background import Background2D

# function to turn .xml files into x, y, z coordinates
def get_locations(path):
    # read xml file
    with open(path, 'r') as f:
        data = f.read()
    # get xml data in iterable format
    markers = BeautifulSoup(data, features='xml').find_all('Marker')
    cell_loci = []
    # iterate and output marker locations
    for marker in markers:
        coords = re.findall('[0-9]+', marker.text)
        coords = [int(x) for x in coords]
        cell_loci.append(coords)
    
    return cell_loci

# prediction and annotation overlap and build dataframe for validation metrics
# max distance a prediction can be off and still count as cell. very conservative at 1 px
def annot_pred_overlap(blocks, max_dist, type_2D = None):

    metric_df = pd.DataFrame(columns = ['x', 'y', 'z', 'block', 'annotated', 'predicted'])
    
    if isinstance(type_2D, type(None)):
        pred_keys = ['pred_detect', 'pred_reject']
    else:
        pred_keys = ['pred_detect_' + type_2D, 'pred_reject_' + type_2D]
    
    for key in blocks.keys():
    
        # get annotated data (not alwats a reject file)
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
        pred_detect = np.asarray(blocks[key][pred_keys[0]])
        pred_reject = np.asarray(blocks[key][pred_keys[1]])
    
        if len(pred_reject) > 0:
            pred_points = np.concatenate((pred_detect, pred_reject), axis = 0)
            pred_types = np.concatenate((np.ones((len(pred_detect), 1)),
                                     np.ones((len(pred_reject), 1)) * 2), axis = 0)
        else:
            pred_points = pred_detect.copy()
            pred_types = np.ones((len(pred_detect), 1))
        
        # create annotation and prediction tree for comparison
        annot_tree = KDTree(annot_points)
        pred_tree = KDTree(pred_points)
    
        # returns: for each element in annot_tree[i], indexes[i] is a list of indecies within distance r from pred_tree
        indexes = annot_tree.query_ball_tree(pred_tree, r = max_dist)
        pred_id = np.zeros((len(annot_types), 1))
        pred_extra = np.zeros((len(pred_types), 1))
    
        # get the index and type for all annotated cells and id predicted cells that were not annotated
        for c, idx in enumerate(indexes):
            if len(idx) > 0:
                pred_id[c] = pred_types[idx[0]]
                pred_extra[idx[0]] = 1
    
        pred_id[pred_id == 0] = 3
        data_array = np.concatenate((annot_points,
                                     np.ones((len(annot_points), 1)) * int(key),
                                     annot_types,
                                     pred_id), axis = 1)
    
        # get location and type of predicted cells that where not annotated
        pred_extra_loc, _ = np.where(pred_extra == 0)
        if len(pred_extra_loc) > 0:
            curr_points = pred_points[pred_extra_loc[:], :]
            curr_types = pred_types[pred_extra_loc[:], :]
            pred_array = np.concatenate((curr_points,
                                        np.ones((len(curr_points), 1)) * int(key),
                                        np.ones((len(curr_points), 1)) * 3,
                                        curr_types), axis = 1)
        
            # add to annot array
            data_array = np.vstack((data_array, pred_array))

        # create dataframes                                         
        curr_df = pd.DataFrame(data_array, columns = ['x', 'y', 'z', 'block', 'annotated', 'predicted'])
        metric_df = pd.concat((metric_df, curr_df))
    return metric_df

# preprocessing function to standardize the image stack for training networks
# mainly taken from https://photutils.readthedocs.io/en/stable/background.html
# modified to deal with 3D signal and more complex background
def bkg_transforms(img, bkg, sigma):
    curr_norm = img- bkg.background
    curr_norm = ndi.gaussian_filter(curr_norm, sigma = 1.0, mode = 'constant', cval = 0, truncate = 2)
    thresh = np.mean(curr_norm[curr_norm > 0]) + sigma * np.std(curr_norm[curr_norm > 0])
    curr_thresh = np.where(curr_norm > thresh, curr_norm, 0)
    curr_scaled = np.where(curr_thresh > 0, img, 0)
    return curr_scaled

def astro_preprocess(img, estimator, save_dir, box = (50, 50), filt = (3, 3), sigma = 3, sig_clip = 3, pad = 0, smooth = True):
    
    bkg_sub_array = np.zeros(img.shape)
    est = getattr(importlib.import_module('photutils.background'), estimator)
    
    for depth in range(img.shape[0]):
        
        curr_img = img[depth, :, :]
        sigma_clip = SigmaClip(sigma = sig_clip, maxiters = 10)
        # check if padding has been added and mask regions accordingly
        if pad > 0:
            if depth >= pad or depth < (img.shape[0] - pad):

                mask = np.full(curr_img.shape, True)
                mask[pad:-pad, pad:-pad] = False
                
                bkg = Background2D(curr_img, box_size = box, filter_size = filt, 
                                   bkg_estimator = est(), fill_value=0.0, 
                                   sigma_clip=sigma_clip, coverage_mask = mask, 
                                   exclude_percentile = 50)
                
                bkg_sub_array[depth, :, :] = bkg_transforms(curr_img, bkg, sigma)
        else:
            
            bkg = Background2D(curr_img, box_size = box, filter_size = filt, 
                               bkg_estimator = est(), fill_value=0.0)
            
            bkg_sub_array[depth, :, :] = bkg_transforms(curr_img, bkg, sigma)
            
    
    if smooth:
        bkg_sub_array = ndi.gaussian_filter(bkg_sub_array, sigma = 1.0, mode = 'constant', cval = 0, truncate = 2)
        
    return bkg_sub_array


