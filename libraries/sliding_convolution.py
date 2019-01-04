# import custom JS animator
from mlrefined_libraries.JSAnimation_slider_only import IPython_display_slider_only

# import standard plotting and animation
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
import matplotlib.animation as animation
from IPython.display import clear_output
import time
from matplotlib import gridspec
import matplotlib.patches as patches
import seaborn as sns
from scipy import signal as sig

# import other packages
import numpy as np
import math

class visualizer:
    '''
    Compute 2D convolution graphically 
    '''
    def __init__(self,**args):
        self.image = args['image']                       
        self.kernel = args['kernel']  
               
        
    def draw_it(self,**kwargs):
            
        fmt = '.1f'    
        if 'mode' in kwargs and kwargs['mode']=='int':
            fmt = 'd'
            
        
        num_frames = np.size(self.image)
        
        # initialize figure
        fig = plt.figure(figsize=(11,5))
        artist = fig
        
        # create subplot with 5 panels
        gs = gridspec.GridSpec(1, 5, width_ratios=[.6, .05, 1, .05, 1]) 
        ax1 = plt.subplot(gs[0]) # kernel 
        ax5 = plt.subplot(gs[4]) # convolution result
        ax3 = plt.subplot(gs[2]) # image
        ax2 = plt.subplot(gs[1]) # convolution symbol 
        ax2.scatter(0, 0, marker="$*$", s=80, c='k'); ax2.set_ylim([-1, 1]); ax2.axis('off');
        ax4 = plt.subplot(gs[3]) # equal sign
        ax4.scatter(0, 0, marker="$=$", s=80, c='k'); ax4.set_ylim([-1, 1]); ax4.axis('off');   
        
        
        # a sub-function to pad zeros to the input
        def pad_zeros(image, L0, L1):
            
            padded_image = image
            
            # pad zeros to top/bottom
            l0 = int((L0-1)/2) 
            top_bottom = np.zeros((l0, np.shape(padded_image)[1]))
            padded_image = np.concatenate((top_bottom, padded_image), 0)
            padded_image = np.concatenate((padded_image, top_bottom), 0)
              
            # pad zeros to left/right
            l1 = int((L1-1)/2) 
            left_right = np.zeros((np.shape(padded_image)[0], l1))
            padded_image = np.concatenate((left_right, padded_image),1)
            padded_image = np.concatenate((padded_image, left_right),1)
       
            return padded_image         
        
        
        # get image sizes
        N0 = np.shape(self.image)[0]
        N1 = np.shape(self.image)[1]
        
        # compute half image sizes
        N0_half = int(np.floor(N0/2))
        N1_half = int(np.floor(N1/2))
   
        
        # get kernel sizes
        L0 = np.shape(self.kernel)[0]
        L1 = np.shape(self.kernel)[1]
        
        # compute half kernel sizes
        L0_half = int(np.floor(L0/2))
        L1_half = int(np.floor(L1/2))
        
        
        # make kernel_padded and its mask
        kernel_padded = np.zeros((2*N0_half+1, 2*N1_half+1)) 
        kernel_padded[N0_half-L0_half:N0_half-L0_half+L0, N1_half-L1_half:N1_half-L1_half+L1] = self.kernel    
        mask_kernel_padded = np.ones(np.shape(kernel_padded))
        mask_kernel_padded[N0_half-L0_half:N0_half-L0_half+L0, N1_half-L1_half:N1_half-L1_half+L1] = 0   
        
        
        # make image_padded and its mask
        image_padded = pad_zeros(self.image, L0, L1)   
        mask_image_padded = np.ones(np.shape(self.image))
        mask_image_padded = pad_zeros(mask_image_padded, L0, L1)
        mask_image_padded = np.maximum(-2*mask_image_padded+1, 0) # turn 0s to 1s and vice-versa
        
    
        # make conv_padded and its mask
        conv = sig.convolve2d(self.image, np.flipud(np.fliplr(self.kernel)), boundary='fill', fillvalue = 0, mode='same')
        conv_padded = pad_zeros(conv, L0, L1)
        mask_conv_padded = np.zeros(np.shape(conv))
        mask_conv_padded = pad_zeros(mask_conv_padded, L0, L1)
        mask_conv_padded = np.maximum(-2*mask_conv_padded+1, 0) # turn 0s to 1s and vice-versa
        
        
        slider = []
        for i in range(0, N0):
            for j in range(0, N1):
                slider.append([i,j])
                
                
        fmt = '.1f'    
        if 'mode' in kwargs and kwargs['mode']=='int':
            fmt = 'd'
            kernel_padded = kernel_padded.astype('int')
            image_padded = image_padded.astype('int')
            conv_padded = conv_padded.astype('int')
            
            
        
        print ('starting animation rendering...')
         
        
        # animation sub-function
        def animate(k):
            
            # clear the panel
            ax1.cla()
            ax3.cla()
            ax5.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
                

            # plot kernel
            cmap_kernel = ["#34495e"]
            
            sns.heatmap(kernel_padded, mask=mask_kernel_padded, square=True, cbar=False, cmap=cmap_kernel,
                        annot=True, fmt=fmt, linewidths=.1, yticklabels=False, xticklabels=False,
                        annot_kws={"weight": "bold"}, ax=ax1)
           
            
            
            # plot image
            cmap_white = ["#ffffff"]
            cmap_gray =  ["#cccccc"]
        
        
            sns.heatmap(image_padded, square=True,  cbar=False, cmap=cmap_white,
                        annot=True, fmt=fmt, linewidths=0.1, yticklabels=False, xticklabels=False, ax=ax3)

            sns.heatmap(image_padded, mask=mask_image_padded, square=True,  cbar=False, cmap=cmap_gray,
                        annot=True, fmt=fmt, linewidths=0.1, yticklabels=False, xticklabels=False, ax=ax3)

            # make sliding mask 
            mask_sliding_image = np.ones(np.shape(image_padded))
            mask_sliding_image[slider[k][0]:slider[k][0]+L0, slider[k][1]:slider[k][1]+L1] = 0
            
            sns.heatmap(image_padded, mask=mask_sliding_image, square=True, cbar=False, cmap=cmap_kernel,
                        annot=True, fmt=fmt, linewidths=.1, yticklabels=False, xticklabels=False,
                        annot_kws={"weight": "bold"}, ax=ax3)

            
            
           
            # plot convolution
            cmap_conv =  ["#ffcc00"]
            
            mask_conv_padded[slider[k][0]+L0_half, slider[k][1]+L1_half] = 0
            sns.heatmap(conv_padded, mask=mask_conv_padded, square=True, cbar=False, cmap=cmap_conv,
                        annot=True, fmt=fmt, linewidths=.1, yticklabels=False, xticklabels=False,
                        annot_kws={"weight": "bold"}, ax=ax5)
              
            
            
            return artist,
        
        anim = animation.FuncAnimation(fig, animate,frames=num_frames, interval=num_frames, blit=True)
        
        return(anim)