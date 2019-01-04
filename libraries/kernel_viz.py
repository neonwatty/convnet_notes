# import standard plotting and animation
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import copy
import operator


# import other packages
import numpy as np
#import cv2
from PIL import Image
from PIL import ImageEnhance
from scipy import signal as sig
import time
from sklearn.preprocessing import normalize
import seaborn as sns

        
def show_conv(image, kernels, **kwargs):
    
    contrast_normalization = True
    if 'contrast_normalization' in kwargs:
        contrast_normalization = kwargs['contrast_normalization']
    
    # load image 
    #image = cv2.imread(image_path, 0)
    #image = Image.open(image_path).convert('L')
    #image = np.array(image)

    
    # get number of kernels
    K = np.shape(kernels)[0]
    
    # initialize figure
    fig = plt.figure(figsize=(10,4*K))
    #fig.subplots_adjust(hspace=None)
    gs = gridspec.GridSpec(K, 5, width_ratios=[.4, .05, 1, .05, 1]) 
    
    for i, kernel in enumerate(kernels):
    
        # compute convolution
        conv = sig.convolve2d(image, np.flipud(np.fliplr(kernel)), boundary='fill', fillvalue = 0, mode='same')
   
        # create subplot with 5 panels
        ax1 = plt.subplot(gs[0+5*i]) # kernel 
        ax5 = plt.subplot(gs[4+5*i]); ax5.axis('off') # convolution result
        ax3 = plt.subplot(gs[2+5*i]); ax3.axis('off') # image
        ax2 = plt.subplot(gs[1+5*i]) # convolution symbol 
        ax2.scatter(0, 0, marker="$*$", s=80, c='k'); ax2.set_ylim([-1, 1]); ax2.axis('off');
        ax4 = plt.subplot(gs[3+5*i]) # equal sign
        ax4.scatter(0, 0, marker="$=$", s=80, c='k'); ax4.set_ylim([-1, 1]); ax4.axis('off'); 

    
    
        # plot convolution kernel
        cmap_kernel = ["#34495e"]
        L0 = np.shape(kernel)[0]
        L0_h = int((L0-1)/2)
        L1 = np.shape(kernel)[1]
        L1_h = int((L1-1)/2)
        L = max(L0,L1)
        L_h = int((L-1)/2)  
        mask = np.ones((L,L))
        mask[L_h-L0_h:L_h+L0_h+1, L_h-L1_h:L_h+L1_h+1] = 0 
        
        kernel_new = np.ones((L,L))
        kernel_new [L_h-L0_h:L_h+L0_h+1, L_h-L1_h:L_h+L1_h+1] = kernel
        
        
        sns.heatmap(kernel_new, square=True, mask=mask, cbar=False, cmap=cmap_kernel,
                        annot=True, fmt=".1f", linewidths=.1, yticklabels=False, xticklabels=False,
                        annot_kws={"weight": "bold"}, ax=ax1)
    
    
        # plot input image
        ax3.imshow(image, cmap=plt.get_cmap('gray'), clim=(0, 255))
    

        # plot convolution
        if contrast_normalization:
            conv = normalize_contrast(conv)


        ax5.imshow(np.sign(conv), cmap=plt.get_cmap('gray'))# 'pink'))

        plt.show()
    

def normalize_contrast(image):
    
    # linear transformation 
    a = np.min(image)
    b = np.max(image)        
    image = (image*255/(b-a))-(255*a/(b-a))
    
    # make sure all pixels are integers between 0 and 255
    eps = 1e-4
    image = np.floor(image+eps)
    
    # change data type to uint8
    image = image.astype('uint8')
    
    # equalize histogram using opencv
    #image = cv2.equalizeHist(image)
    
    return image







    
   