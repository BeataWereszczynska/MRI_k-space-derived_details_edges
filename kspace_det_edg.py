# -*- coding: utf-8 -*-
"""
k-space based details/edges detection in MRI images from Agilent FID data.

Created on Tue Nov 15 2022
Last modified on Fri Nov 18 2022
@author: Beata Wereszczyńska
"""
import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import morphology

def kspace_det_edg(path, picked_slice, radius_min, radius_max, radius_step, threshold):
    """
    k-space based details/edges detection in MRI images from Agilent FID data. 
    Input: 
        .fid folder location: path [str],
        selected slice number: picked_slice [int],
        smalest radius for k-space masking: radius_min [int],
        largest radius for k-space masking: radius_max [int],
        step betwen the two above values: radius_step [int],
        threshold option: threshold [tuple (type, value)].
    Threshold options are: ('auto', ), ('manual', threshold value), ('adaptive', C value)
    """
    
    # import k-space data
    params, echoes = ng.agilent.read(dir=path)
    number_of_slices = params['ntraces']
    del params, path
    kspace = echoes[picked_slice - 1 : echoes.shape[0] : number_of_slices, :]  # downsampling to one slice
    del echoes, number_of_slices, picked_slice
    
    # create image with 2D FFT
    ft = np.fft.fft2(kspace)
    ft = np.fft.fftshift(ft)                # fixing problem with corner being center of the image
    ft = np.transpose(np.flip(ft, (1,0)))   # matching geometry with VnmrJ-calculated image (still a bit shifted)
    
    # masking k-space and creating an image of k-space-defined details/edges
    sum_of_masked = np.ones(shape=kspace.shape)
    mask = np.ones(shape=kspace.shape)
    
    for value in range(radius_min, radius_max+1, radius_step):
        cv2.circle(img=mask, center=(int(kspace.shape[0]/2),int(kspace.shape[1]/2)), 
                   radius = value, color =(0,0,0), thickness=-1)
        masked_k = np.multiply(kspace,mask)
        sum_of_masked = sum_of_masked + masked_k
    
    del value, mask, masked_k, kspace, radius_min, radius_max, radius_step
    
    ft2 = np.fft.fft2(sum_of_masked)
    ft2 = np.fft.fftshift(ft2)
    ft2 = np.transpose(np.flip(ft2, (1,0)))
    
    del sum_of_masked
    
    # normalizing the image of k-space-defined details/edges (0-255)
    ft2 = cv2.normalize(abs(ft2), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # k-space-defined details/edges to binary image
    if threshold[0] == 'adaptive':
        im_bw = cv2.adaptiveThreshold(ft2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY,int(((ft2.shape[0]*ft2.shape[1])/4)+1),
                                      threshold[1])
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
    
    # visualization
    plt.rcParams['figure.dpi'] = 1200
    plt.subplot(131)
    plt.title('MRI image', fontdict = {'fontsize' : 8}), plt.axis('off')
    plt.imshow(abs(ft), cmap=plt.get_cmap('gray'))
    plt.subplot(132)
    plt.title('k-space-derived details/edges',
              fontdict = {'fontsize' : 8}), plt.axis('off')
    plt.imshow(ft2, cmap=plt.get_cmap('gray'))
    plt.subplot(133)
    plt.title('binary image of details/edges',
              fontdict = {'fontsize' : 8}), plt.axis('off')
    plt.imshow(im_bw, cmap=plt.get_cmap('gray'))
    plt.tight_layout(pad=0, w_pad=0.2, h_pad=1.0)
    plt.show()
    
    # return data
    return ft, ft2, im_bw


def main():
    path = 'sems_20190203_03.fid'         # .fid folder location [str]
    picked_slice = 4                      # selected slice number [int]
    radius_min = 4                        # smalest radius for k-space masking [int]
    radius_max = 45                       # largest radius for k-space masking [int]
    radius_step = 1                       # step betwen the two above values [int]
    threshold = ('auto', )                # threshold option [tuple (type, value)]
    # threshold options: ('auto', ), ('manual', threshold value), ('adaptive', C value)
        
    # running calculations and retrieving the results
    ft, ft2, im_bw = kspace_det_edg(path, picked_slice, radius_min, radius_max, radius_step, threshold)
    
    # creating global variables to be available after the run completion
    global MRI_complex_img
    MRI_complex_img = ft
    global details_img
    details_img = ft2
    global binary_details
    binary_details = im_bw
    

if __name__ == "__main__":
    main()
