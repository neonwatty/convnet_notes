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

# import other packages
import numpy as np
import math

class visualizer:
    '''
    Illustrate the sliding window idea
    '''
    def __init__(self,**args):
        self.image = args['image']                       # input image
        
        
    def draw_it(self,**kwargs):                    
        
        sliding_window_size = (3,3)
        if 'sliding_window_size' in kwargs:
            sliding_window_size = kwargs['sliding_window_size']
    
        stride = 1
        if 'stride' in kwargs:
            stride = kwargs['stride']
             
            
        # initialize figure
        fig = plt.figure(figsize=(10,4))
        artist = fig
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,3, 1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off');
        ax3 = plt.subplot(gs[2]); ax3.axis('off');
        ax = plt.subplot(gs[1])
       
        
        slider=[]     
        for i in np.arange(0, np.shape(self.image)[0]-sliding_window_size[0]+1, stride):
            for j in np.arange(0, np.shape(self.image)[1]-sliding_window_size[1]+1, stride):
                slider.append((i,j))
        
        num_frames = np.shape(slider)[0]
        
        
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
                

            # plot original image
            ax.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            
            # plot sliding window
            ax.add_patch(
                patches.Rectangle(
                    (slider[k][1], slider[k][0]),    # (x,y)
                    sliding_window_size[1],          # width
                    sliding_window_size[0],          # height
                    edgecolor="red",
                    linewidth=2,
                    facecolor="none"
                ))
              
            
            # remove tickmarks
            ax.set_xticks([])
            ax.set_yticks([]) 
            
            
            return artist,
        
        anim = animation.FuncAnimation(fig, animate,frames=num_frames, interval=num_frames, blit=True)
        
        return(anim)