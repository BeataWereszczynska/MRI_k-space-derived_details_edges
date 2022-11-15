# -*- coding: utf-8 -*-
"""
k-space based details/edges detection in MRI images from Agilent FID data.

Created on Tue Nov 15 2022
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
    
    # masking k-space and creating multi-image k-space-defined details/edges image
    sum_of_images = np.ones(shape=(kspace.shape[0],kspace.shape[1]))
    mask = np.ones(shape=(kspace.shape[0],kspace.shape[1]))
    
    for value in range(radius_min, radius_max+1, radius_step):
        cv2.circle(img=mask, center=(int(kspace.shape[0]/2),int(kspace.shape[1]/2)), 
                   radius = value, color =(0,0,0), thickness=-1)
        masked_k = np.multiply(kspace,mask)
        ft2 = np.fft.fft2(masked_k)
        ft2 = np.fft.fftshift(ft2)
        ft2 = np.transpose(np.flip(ft2, (1,0)))
        sum_of_images = sum_of_images + ft2
    
    del value, ft2, mask, masked_k, kspace, radius_min, radius_max, radius_step
    
    # normalizing the image of k-space-defined details/edges (0-255)
    sum_of_images = cv2.normalize(abs(sum_of_images), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # k-space-defined details/edges to binary image
    if threshold[0] == 'adaptive':
        im_bw = cv2.adaptiveThreshold(sum_of_images, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY,int(((sum_of_images.shape[0]*sum_of_images.shape[1])/4)+1),
                                      threshold[1])
    elif threshold[0] == 'manual':
        im_bw = cv2.threshold(sum_of_images, threshold[1], 255, cv2.THRESH_BINARY)[1]
    elif threshold[0] == 'auto':
        ret2,im_bw = cv2.threshold(sum_of_images,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        print("Threshold options are: ('auto', ), ('manual', threshold value), ('adaptive', C value).")
        im_bw = np.zeros((sum_of_images.shape[0], sum_of_images.shape[1]))
    
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
    plt.imshow(sum_of_images, cmap=plt.get_cmap('gray'))
    plt.subplot(133)
    plt.title('binary image of details/edges',
              fontdict = {'fontsize' : 8}), plt.axis('off')
    plt.imshow(im_bw, cmap=plt.get_cmap('gray'))
    plt.tight_layout(pad=0, w_pad=0.2, h_pad=1.0)
    plt.show()
    
    # return data
    return ft, sum_of_images, im_bw



def main():
    path = 'sems_20190203_03.fid'         # .fid folder location [str]
    picked_slice = 4                      # selected slice number [int]
    radius_min = 4                        # smalest radius for k-space masking [int]
    radius_max = 45                       # largest radius for k-space masking [int]
    radius_step = 1                       # step betwen the two above values [int]
    threshold = ('auto', )                # threshold option [tuple (type, value)]
    # threshold options: ('auto', ), ('manual', threshold value), ('adaptive', C value)
        
    # running calculations and retrieving the results
    ft, sum_of_images, im_bw = kspace_det_edg(path, picked_slice, radius_min, radius_max, radius_step, threshold)
    
    # creating global variables to be available after the run completion
    global MRI_complex_img
    MRI_complex_img = ft
    global details_img
    details_img = sum_of_images
    global binary
    binary = im_bw
    

if __name__ == "__main__":
    main()
