# import custom JS animator
from mlrefined_libraries.JSAnimation_slider_only import IPython_display_slider_only

# import standard plotting and animation
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
import matplotlib.animation as animation
from IPython.display import clear_output
import time
from matplotlib import gridspec

# import other packages
import numpy as np
import math


class visualizer:
    '''
    Illustrate convolution/cross-correlation of an input image with variable size kernels.
    '''
    def __init__(self,**args):
        self.img = args['img']                       # input image

        
    def draw_it(self,**kwargs):
        
        num_frames = 100                       
        
        if 'num_frames' in kwargs:
            num_frames = kwargs['num_frames']
            
        # initialize figure
        fig = plt.figure(figsize = (10,7))
        artist = fig
        


        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,3, 1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off');
        ax3 = plt.subplot(gs[2]); ax3.axis('off');
        ax = plt.subplot(gs[1])

        
        # slider (horizontal axis)
        slider = np.linspace(10, 1000, num_frames)
        
        print ('starting animation rendering...')
    

        
        
        # animation sub-function
        def animate(k):
            
            # clear the panel
            ax.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            # kernel size for the current frame 
            s = slider[k]
                

            # compute edge image
            edges = cv2.Canny(self.img, 1, s)

            # plot conv image
            ax.imshow(edges, cmap = 'gray')
                
 
            # plot convolution/cross-correlation
            #ax.plot(y_hat, color = 'red', linewidth=2.5)
               
            
            # fix viewing limits on panel
            #ax.set_ylim([min(self.y)-2, max(self.y)+2])

            # set tickmarks
            ax.set_xticks([])
            ax.set_yticks([])     
            
            # label axes
            #ax.set_xlabel('$\mathrm{days\,\,elapsed}$', fontsize = 12)
            #ax.set_ylabel('$\mathrm{approval\,\,ratings\,\,(\%)}$', fontsize = 12, rotation = 90, labelpad = 15)
            
            # set axis 
            #ax.axhline(y=0, color='k', zorder = 0, linewidth = 0.5)
            


            
            
            return artist,
        
        anim = animation.FuncAnimation(fig, animate,frames=num_frames, interval=num_frames, blit=True)
        
        return(anim)