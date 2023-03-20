#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:23:13 2022

@author: nicholas.lusk
"""

import os
import re
import warnings
import importlib

import numpy as np
import pandas as pd
import utils.pymusica as pymusica

from bs4 import BeautifulSoup

from scipy import ndimage as ndi
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.morphology import convex_hull_image
from astropy.stats import SigmaClip
from astropy.visualization import SqrtStretch
from photutils.background import Background2D

# ignores numpy warning
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)

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

# preprocessing function to standardize the image stack for training networks
# mainly taken from https://photutils.readthedocs.io/en/stable/background.html
# modified to deal with 3D signal and more complex background
def astro_preprocess(img, estimator, box = (20, 20), filt = (3, 3), sig_clip = 3, pad = 0, smooth = True, exclude_per = 50, max_iter = 10):
    
    bkg_sub_array = np.zeros(img.shape)
    est = getattr(importlib.import_module('photutils.background'), estimator)
    
    L = 7
    params_m = {'M': 1023.0, 'a': np.full(L, 11), 'p': 0.7}
    patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                )
    
    for depth in range(img.shape[0]):
        
        curr_img = img[depth, :, :]
        sigma_clip = SigmaClip(sigma = sig_clip, maxiters = max_iter)
        # check if padding has been added and mask regions accordingly
        if pad > 0:
            if depth >= pad or depth < (img.shape[0] - pad):

                mask = np.full(curr_img.shape, True)
                mask[pad:-pad, pad:-pad] = False
                
                # Run contrast enhancement using Laplacian Pyramid
                curr_img[pad:-pad, pad:-pad] = pymusica.musica(curr_img[pad:-pad, pad:-pad], L, params_m)
                
                # get background statistics using photoutils
                bkg = Background2D(curr_img, box_size = box, filter_size = filt, 
                                   bkg_estimator = est(), fill_value=0.0, 
                                   sigma_clip=sigma_clip, coverage_mask = mask, 
                                   exclude_percentile = exclude_per)
                


        else:
            curr_img = pymusica.musica(curr_img, L, params_m)
            bkg = Background2D(curr_img, box_size = box, filter_size = filt, 
                               bkg_estimator = est(), fill_value=0.0)
            
        curr_img = curr_img - bkg.background
        curr_img = np.where(curr_img < 0, 0, curr_img)
        
        #sigma_est = np.mean(estimate_sigma(curr_img))
        #curr_img = denoise_nl_means(curr_img, h=0.8 * sigma_est, sigma=sigma_est,
        #                           fast_mode=True, **patch_kw)
        
        bkg_sub_array[depth, :, :] = curr_img
    
    if smooth:
        bkg_sub_array = ndi.gaussian_filter(bkg_sub_array, sigma = 1.0, mode = 'constant', cval = 0, truncate = 2)
        
    return bkg_sub_array

def build_training_stack(img, mask):
    
    max_dim = np.max(img.shape)
    padding = [(int((max_dim - dim) / 2), int((max_dim - dim) / 2)) for dim in img.shape]
    cube_stack = np.pad(img, padding, 'constant')
    cube_mask = np.pad(mask, padding, 'constant')
    

        
    out_stack, out_mask = cube_stack.copy(), cube_mask.copy()
            
    #for i in range(2):
    #    cube_stack = np.transpose(cube_stack, (1, 2, 0))
    #    cube_mask = np.transpose(cube_mask, (1, 2, 0))
    #    
    #    out_stack = np.concatenate((out_stack, cube_stack), axis = 0)
    #    out_mask = np.concatenate((out_mask, cube_mask), axis = 0)
         
    #need to give all masks in mask_array unique values
    mask_counts = 0
    for z in range(out_mask.shape[0]):
        mask_vals = np.unique(out_mask[z, :, :])
        
        if len(mask_vals) == 1:
            pass
        else:
            for c, val in enumerate(mask_vals[1:]):
                out_mask[z, :, :] = np.where(convex_hull_image(np.where(out_mask[z, :, :] == val, c + 1, 0)), c + 1, out_mask[z, :, :])
            
            out_mask[z, :, :] = np.where(out_mask[z, :, :] > 0, out_mask[z, :, :] + mask_counts, out_mask[z, :, :])
            mask_counts += c + 1
            
    return out_stack, out_mask
