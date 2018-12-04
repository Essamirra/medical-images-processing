# -*- coding: utf-8 -*-


import numpy as np
import scipy.stats


def extract_histogramm_features(image):
#    hist = np.zeros(256)
#    unique, counts = np.unique(image, return_counts=True)
#    hist[unique] = counts
#    
#    values = np.arange(256, dtype=np.uint8)
#    density = hist / np.sum(hist)
    
    characteristics = []
    intensity = image.flatten()
    
    characteristics.append(intensity.mean())
    characteristics.append(intensity.std())
    characteristics.append(scipy.stats.entropy(intensity))
    for moment in (2, 3, 4):
        characteristics.append(scipy.stats.moment(intensity, moment=moment))
    for q in (0, 25, 50, 75, 100):
        characteristics.append(np.percentile(intensity, q=q))
    
    return np.array(characteristics)


def extract_haralick_features(image):
    sum_GLCM = np.zeros((256, 256))
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1)]
    h = image.shape[0]
    w = image.shape[1]
    
    for direction in directions:
        shift_x = direction[1]
        shift_y = direction[0]
        
        left = max(0, shift_x)
        right = min(w, w + shift_x)
        top = max(0, shift_y)
        bottom = min(h, h + shift_y)
        sub_image_first = image[top:bottom, left:right]
        
        left = max(0, -shift_x)
        right = min(w, w - shift_x)
        top = max(0, -shift_y)
        bottom = min(h, h - shift_y)
        sub_image_second = image[top:bottom, left:right]
        
        np.add.at(
                sum_GLCM, 
                (sub_image_first.flatten(), sub_image_second.flatten()), 1)
        
    GLCM = sum_GLCM / np.sum(sum_GLCM)
    col_idx, row_idx = np.meshgrid(np.arange(256), np.arange(256))
    
    contrast = np.sum(((row_idx - col_idx) ** 2) * GLCM)
    homogeneity = np.sum(GLCM / (1 + (row_idx - col_idx) ** 2))
    entropy = np.sum(-GLCM * np.log(GLCM + 1e-16))
    ASM = np.sum(GLCM ** 2)
    energy = np.sqrt(ASM)
    
    mu_x = np.sum(col_idx * GLCM)
    mu_y = np.sum(row_idx * GLCM)
    sigma_x = np.sqrt(np.sum(((col_idx - mu_x) ** 2) * GLCM))
    sigma_y = np.sqrt(np.sum(((row_idx - mu_y) ** 2) * GLCM))
    correlation = np.sum(GLCM * (col_idx - mu_x) * (row_idx - mu_y) / sigma_x / sigma_y)
    
    moment_1 = np.sum(((row_idx - col_idx) ** 1) * GLCM)
    moment_3 = np.sum(((row_idx - col_idx) ** 3) * GLCM)
    moment_4 = np.sum(((row_idx - col_idx) ** 4) * GLCM)
    
    autocorrelation = np.sum(row_idx * col_idx * GLCM)
    inverse_difference = np.sum(GLCM / (1 + np.abs(row_idx - col_idx)))
    
    result = np.array([contrast, homogeneity, entropy, ASM, energy, correlation, 
                       moment_1, moment_3, moment_4, autocorrelation, inverse_difference])
        
    return result


def region_growing():
    pass


