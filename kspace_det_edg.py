# -*- coding: utf-8 -*-
"""
k-space based details/edges detection in MRI images with optional k-space based 
denoising and detail control (for Agilent FID data).

Created on Tue Nov 15 2022
Last modified on Thu Dec  1 2022

@author: Beata Wereszczyńska
"""
import nmrglue as ng
import numpy as np
import cv2
from skimage import morphology
import matplotlib.pyplot as plt


def import_kspace(path, number_of_slices, picked_slice):
    """
    Imports k-space corresponding to the picked slice from Agilent FID data.
    Input:
        .fid folder location: path [str],
        total number of slices in the MRI experiment: number_of_slices [int],
        selected slice number: picked_slice [int],
    """
    
    # import k-space data
    echoes = ng.agilent.read(dir=path)[1]
    kspace = echoes[picked_slice - 1 : echoes.shape[0] : number_of_slices, :]  # downsampling to one slice
        
    # return data
    return kspace


def wght_msk_kspace(kspace, weight_power, contrast):
    """
    k-space weighting and masking for denoising of MRI image without blurring or losing contrast, 
    as well as for brightening of the objects in the image with simultaneous noise reduction.
    Input:
        k-space data: kspace [array]
        exponent in the signal weighting equation: weight_power[float]
        restoring contrast: contrast [bool] (1 for denoising, 0 for brightening).
        
    From https://github.com/BeataWereszczynska/k-space_wght_msk_for_MRI_denoising/blob/main/wght_msk_kspace.py
    """
    
    # k-space weighting
    if contrast:
        kspace = kspace * np.power(abs(kspace), 3*weight_power)
        # contrasting
        a = kspace[abs(kspace) > abs(np.max(kspace))/6]
        a = a / np.power(abs(a), weight_power)
        kspace[abs(kspace) > abs(np.max(kspace))/6] = a
        del a
    else:
        kspace = kspace * np.power(abs(kspace), weight_power)
    del contrast, weight_power
    
    # k-space masking
    r = int(kspace.shape[0]/2)
    mask = np.zeros(shape=kspace.shape)
    cv2.circle(img=mask, center=(r,r), radius = r, color =(1,0,0), thickness=-1)
    kspace = np.multiply(kspace, mask)
        
    # return data
    return kspace

    
def grad_mask_kspace(kspace, r):
    """
    Graduate k-space masking for MRI image blurring.
    Input:
        k-space data: kspace [array],
        radius for k-space masking in pixels: r [int].
    
    From https://github.com/BeataWereszczynska/k-space_masking_for_MRI_denoising/blob/main/grad_mask_kspace.py
    """

    # graduate k-space masking
    mask_denoise = np.zeros(shape=kspace.shape)
    mask = np.zeros(shape=kspace.shape)
    for value in range(r, r+15, 2):
        cv2.circle(img=mask, center=(int(kspace.shape[0]/2),int(kspace.shape[1]/2)), 
                    radius = value, color =(1,0,0), thickness=-1)
        mask_denoise = mask_denoise + mask
    kspace = np.multiply(kspace, mask_denoise)
    
    # return data
    return kspace


def FFT_2D(kspace):
    """
    Performs two-dimentional Fast Fourier Transform on k-space to reconstruct MRI image 
    (for Agilent FID data).
    """
    
    # reconstruct MRI image
    ft = np.fft.fft2(kspace)              # 2D FFT
    ft = np.fft.fftshift(ft)              # fixing problem with corner being center of the image
    ft = np.transpose(np.flip(ft, (1,0))) # matching geometry with VnmrJ-calculated image (still a bit shifted)
    
    # return data
    return ft


def kspace_det_edg(kspace, radius_min, radius_max, radius_step, threshold):
    """
    k-space based details/edges detection in MRI images with optional k-space based denoising
    and detail control.
    Input: 
        k-space data: kspace[array],
        denoising option: denoise [tuple (1=on 0=off [bool], weight_power [float], contrast [bool])
        radii for k-space masking in pixels: radii [tuple (min [int], max [int], step [int])],
        detail control option: detail_contr [tuple (1=on 0=off [bool], radius in pixels [int])]
        threshold option: threshold [tuple (type [str], value [int])].
    Threshold options are: ('auto', ), ('manual', threshold value), ('adaptive', C value)
    """
    
    # masking k-space for creating an image of k-space-defined details/edges
    masks = np.zeros(shape=kspace.shape)
    mask = np.ones(shape=kspace.shape)
    for value in range(radius_min, radius_max+1, radius_step):
        cv2.circle(img=mask, center=(int(kspace.shape[0]/2),int(kspace.shape[1]/2)), 
                   radius = value, color =(0,0,0), thickness=-1)
        masks = masks + mask
    kspace = np.multiply(kspace, masks)
    
    # reconstructing the image of k-space-defined details/edges
    ft2 = FFT_2D(kspace)
    del kspace
    
    # normalizing the image of k-space-defined details/edges (0-255)
    ft2 = cv2.normalize(abs(ft2), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # k-space-defined details/edges to binary image
    if threshold[0] == 'adaptive':
        im_bw = cv2.adaptiveThreshold(ft2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 31, threshold[1])
    elif threshold[0] == 'manual':
        im_bw = cv2.threshold(ft2, threshold[1], 255, cv2.THRESH_BINARY)[1]
    elif threshold[0] == 'auto':
        im_bw = cv2.threshold(ft2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    else:
        print("Threshold options are: ('auto', ), ('manual', threshold value), ('adaptive', C value).")
        im_bw = np.zeros(ft2.shape)
    
    del threshold
    
    # denoise the binary image (remove small objects)
    im_bw = morphology.remove_small_objects(im_bw != 0, min_size=3, in_place=True, connectivity=2)
    
    # return data
    return ft2, im_bw


def visualize(ft1, ft2, im_bw):
    """
    Creates summarising figure.
    """
    
    plt.rcParams['figure.dpi'] = 1200
    plt.subplot(131)
    plt.title('MRI image', fontdict = {'fontsize' : 8}), plt.axis('off')
    plt.imshow(abs(ft1), cmap=plt.get_cmap('gray'))
    plt.subplot(132)
    plt.title('k-space-derived details/edges', fontdict = {'fontsize' : 8}), plt.axis('off')
    plt.imshow(ft2, cmap=plt.get_cmap('gray'))
    plt.subplot(133)
    plt.title('binary image of details/edges', fontdict = {'fontsize' : 8}), plt.axis('off')
    plt.imshow(im_bw, cmap=plt.get_cmap('gray'))
    plt.tight_layout(pad=0, w_pad=0.2, h_pad=1.0)
    plt.show()
    

def main():
    
    # import parameters
    path = 'sems_20190203_03.fid'         # .fid folder location [str]
    number_of_slices = 6                  # total number of slices in the MRI experiment [int]
    picked_slice = 4                      # selected slice number [int]
    
    # denoising parameters
    denoise = 1                           # 1=on 0=off [bool]
    weight_power = 0.02                   # exponent in the signal weighting equation [float]
    contrast = 0                          # restoring contrast: 1=on 0=off [bool]
    
    # detail control parameters
    detail_contr = 1                      # 1=on 0=off [bool]
    detail_r = 200                        # radius for graduate masking in pixels [int]
    
    # details/edges detection parameters
    radius_min = 4                        # smalest radius for k-space masking [int]
    radius_max = 40                       # largest radius for k-space masking [int]
    radius_step = 1                       # step betwen the two above values [int]
    threshold = ('adaptive', -18)         # threshold option [tuple (type, value)]
    # threshold options: ('auto', ), ('manual', threshold value), ('adaptive', C value)
    
    # import
    kspace = import_kspace(path, number_of_slices, picked_slice)
    del path, number_of_slices, picked_slice
    
    # reconstructing the original MRI image
    ft1 = FFT_2D(kspace)
    
    # denoising
    if denoise:
        kspace = wght_msk_kspace(kspace, weight_power, contrast)
    del weight_power, contrast, denoise
    
    #detail control
    if detail_contr:
        kspace = grad_mask_kspace(kspace, detail_r)
    del detail_r, detail_contr
    
    # reconstructing the image of details/edges and creating binary image
    ft2, im_bw = kspace_det_edg(kspace, radius_min, radius_max, radius_step, threshold)
    del kspace
    
    # visualization
    visualize(ft1, ft2, im_bw)
    
    # creating global variables to be available after the run completion
    global MRI_complex_img
    MRI_complex_img = ft1
    global details_img
    details_img = ft2
    global binary_details
    binary_details = im_bw
    

if __name__ == "__main__":
    main()
