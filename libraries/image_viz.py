# import standard plotting and animation
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import copy
#import cv2
from PIL import Image
from scipy import ndimage


# import other packages
import numpy as np
#import cv2
from scipy import signal as sig
import time
from sklearn.preprocessing import normalize
import seaborn as sns

# Edge detection using opencv
def edge_detect(image_path, **kwargs):
    
    #image = cv2.imread(image_path)    
        
    # Gaussian blur params
    GaussianBlurSize = (7,7)
    if 'GaussianBlurSize' in kwargs:
        GaussianBlurSize = kwargs['GaussianBlurSize']
    
    GaussianBlurSigma = 2
    if 'GaussianBlurSigma' in kwargs:
        GaussianBlurSigma = kwargs['GaussianBlurSigma']
    
    
    # Canny edge detector params
    low_threshold = 1
    if 'low_threshold' in kwargs:
        low_threshold = kwargs['low_threshold']
        
    high_threshold = 125
    if 'high_threshold' in kwargs:
        high_threshold = kwargs['high_threshold']
    
    # default dilation params
    dilationSize = (7,7)
    if 'dilationSize' in kwargs:
        dilationSize = kwargs['dilationSize']    
    
    num_iterations = 1
    if 'num_iterations' in kwargs:
        num_iterations = kwargs['num_iterations']    
    
  
    # apply Gaussian blur to smooth out the input image 
    #img = cv2.GaussianBlur(image, GaussianBlurSize, GaussianBlurSigma);

    # apply Canny edge detector
    #edges = cv2.Canny(img, low_threshold, high_threshold)

    # apply dilation to thicken the edges
    #edges = cv2.dilate(edges, np.ones(dilationSize), iterations=num_iterations)


    # initialize figure
    plt.figure(figsize=(10,4))

    # plot input image
    plt.subplot(121)
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.xticks([]), plt.yticks([])

    # plot edge-detected image
    plt.subplot(122)
    plt.imshow(edges)
    plt.xticks([]), plt.yticks([])

    plt.show()  
    
    
def create_image(image_name, **kwargs):
    
    
    if image_name == '2':
        img = 50*np.ones((500,500),dtype='uint8') 
        img[80:120,100:400]=200
        img[380:420,100:400]=200
        img[230:270,100:400]=200
        img[100:250,360:400]=200
        img[250:400,100:140]=200
        
    if image_name == '5':
        img = 50*np.ones((500,500),dtype='uint8') 
        img[80:120,100:400]=200
        img[380:420,100:400]=200
        img[230:270,100:400]=200
        img[100:250,100:140]=200
        img[250:400,360:400]=200        
    
    
    if image_name == 'square_top':
        img = 50*np.ones((500,500),dtype='uint8') 
        img[60:240,60:240]=200
        
        
    if image_name == 'triangle_top':
        img = 50*np.ones((500,500),dtype='uint8') 
        for i in range(60,240):
            for j in range(60,i):
                img[i,j]=200
                
    if image_name == 'square_bottom':
        img = 50*np.ones((500,500),dtype='uint8') 
        img[260:440,260:440]=200            
                
        
    if image_name == 'triangle_bottom':
        img = 50*np.ones((500,500),dtype='uint8') 
        for i in range(260,440):
            for j in range(260,i):
                img[i,j]=200
                
                
    if image_name == 'snowflake':      
        img = 50*np.ones((500,500), dtype='uint8') 
        img[150:350,150:350]=200
        for i in range(0,50):
            for j in range(0,i):
                img[i+100,j+250]=200
                img[i+200,j+350]=200
                img[-i+300,-j+150]=200
                img[-i+399,-j+250]=200
                img[-i+399,j+250]=200
                img[-i+299,j+350]=200
                img[i+100,-j+250]=200
                img[i+201,-j+149]=200    
                
                
                
    if 'plot' in kwargs and kwargs['plot']:
        plt.figure(figsize=(2,2))
        plt.imshow(img, vmin=0, vmax=255, cmap=plt.get_cmap('gray'))
        plt.xticks([]), plt.yticks([])
        plt.show()
        
    
    return img    
                

def load_kernels():
    
    kernels = {'5': np.array([[-1,  0,  1],
                              [-1,  0,  1],
                              [-1,  0,  1]]),
           
               '6': np.array([[ 0,  1,  1],
                              [-1,  0,  1],
                              [-1, -1,  0]]),
               
               '7': np.array([[ 1,  1,  1],
                              [ 0,  0,  0],
                              [-1, -1, -1]]),
           
               '8': np.array([[ 1,  1,  0],
                              [ 1,  0, -1],
                              [ 0, -1, -1]]),
           
               '1': np.array([[ 1,  0, -1],
                              [ 1,  0, -1],
                              [ 1,  0, -1]]),
                       
               '2': np.array([[ 0, -1, -1],
                              [ 1,  0, -1],
                              [ 1,  1,  0]]),
               
               '3': np.array([[-1, -1, -1],
                              [ 0,  0,  0],
                              [ 1,  1,  1]]),
           
               '4': np.array([[-1, -1,  0],
                              [-1,  0,  1],
                              [ 0,  1,  1]])}    
        
    return kernels


def load_kernels_v2(num_kernels, kernel_size):
    
    kernels = {}
    for i in np.arange(1,num_kernels+1):
        kernels[str(i)] = np.random.randn(kernel_size[0], kernel_size[1])   
        
    return kernels
  

def load_directions():

    directions = {'5':'$\\leftarrow$',
                  '6':'$\\swarrow$',
                  '7':'$\\downarrow$',
                  '8':'$\\searrow$',  
                  '1':'$\\rightarrow$',
                  '2':'$\\nearrow$',
                  '3':'$\\uparrow$',
                  '4':'$\\nwarrow$'}                   
                    
    return directions



def show_conv_images(image, **kwargs):
    
    GaussianBlurSigma = 2
    if 'GaussianBlurSigma' in kwargs:
        GaussianBlurSigma = kwargs['GaussianBlurSigma']
    
    kernels = load_kernels()
    directions = load_directions()
   
    # initialize figure
    fig = plt.figure(figsize=(10,4))
    gs=GridSpec(3,8) 

    # plot convolutions
    for i in range(1, 9):
        fig.add_subplot(gs[int(i>4)*4+i+1])
        conv = myConv(image, kernels[str(i)])
        conv = np.round(255*conv/500)  
        #conv = cv2.dilate(conv, np.ones((11,11)), iterations=1)
        plt.imshow(conv, vmin=0, vmax=255, cmap=plt.get_cmap('gnuplot2'))
        plt.xticks([]), plt.yticks([])
        plt.title(directions[str(i)], fontsize = 16)
    
    fig.add_subplot(gs[2,2:6]) # colorbar
    plt.axis('off')
    cbar = plt.colorbar(ticks=[0, 255], orientation='horizontal')
    cbar.ax.set_xticklabels(['$\mathrm{low\,edge\,content}', '$\mathrm{high\,edge\,content}'])# vertically oriented colorbar
    
    fig.add_subplot(gs[0:2,0:2]) # input image
    #plt.imshow(cv2.bitwise_not(image), vmin=0, vmax=255, cmap=plt.get_cmap('Greys'))
    plt.imshow(image, vmin=0, vmax=255, cmap=plt.get_cmap('Greys'))

    plt.xticks([]), plt.yticks([])
    plt.title('$\mathrm{input\,\,image}', fontsize = 10)
    
    fig.add_subplot(gs[0:2,6:8]) # histogram
    hist = calc_conv_hist(image, kernels)     
    make_circ_hist(hist)
    plt.xticks([]), plt.yticks([])
    #plt.title('$\mathrm{edge\,\,histogram}', fontsize = 10)
    
    

    plt.show()  
    
    
def show_conv_hist(images):
    
    kernels = load_kernels()
    
    num_images = np.shape(images)[0]
    
    # initialize figure
    fig = plt.figure(figsize=(2*num_images,4))
    gs = GridSpec(2, num_images) 
    
    for i in range(0, num_images):

        # compute histogram values
        hist = calc_conv_hist(images[i], kernels)
        
        # plot circular histogram
        fig.add_subplot(gs[1,i]) 
        make_circ_hist(hist)
        
        # plot input image
        fig.add_subplot(gs[0,i]) 
        plt.imshow(cv2.bitwise_not(images[i]), vmin=0, vmax=255, cmap=plt.get_cmap('Greys'))
        plt.xticks([]), plt.yticks([])
        
        
    
def make_circ_hist(hist):
    
    # make circular histogram
    t = np.linspace(0, 1.75*np.pi, 8)
    x = .1*np.cos(t)
    y = .1*np.sin(t)
    colors = np.arctan2(x, y)
    norm = Normalize()
    norm.autoscale(colors)
    colormap = cm.viridis
    
    hist = 100*hist**2
    
    plt.quiver(x, y, hist*x, hist*y, color=colormap(norm(colors)),  angles='xy', 
           scale_units='xy', scale=1, width=.013)

    circle=plt.Circle((0, 0), .1, color='k', fill=False)
    
    ax = plt.gca()
    ax.add_artist(circle)
    #ax.set_facecolor((250/256, 250/256, 250/256))
    ax.axis('equal')
    ax.set_aspect('equal')
    plt.axis([-.6, .6, -.6, .6])
    plt.xticks([]), plt.yticks([])
    plt.axis('off')
    plt.show()
    


def calc_conv_hist(image, kernels):
    
    hist=[]
    for i in range(1, 9):
        conv = myConv(image, kernels[str(i)])
        np.maximum(conv, 0, conv) # in place ReLU on conv
        hist.append(conv.sum()) 
        
    hist = np.array(hist)/sum(hist)
    return hist
     

def myConv(image, kernel):
    # flip kernel
    kernel = np.flipud(np.fliplr(kernel))
    
    # compute convolution (more precisely: cross-crorrelation)
    conv = sig.convolve2d(image, kernel, boundary='fill', fillvalue=0, mode='same')
    
    return conv    


def sliding_window(image, sliding_window_size, stride):
    for i in np.arange(0, np.shape(image)[0]-sliding_window_size[0]+1, stride):
        for j in np.arange(0, np.shape(image)[1]-sliding_window_size[1]+1, stride):
            yield image[i:i+sliding_window_size[0], j:j+sliding_window_size[1]]
            
            
def make_feat(image, kernels, **kwargs):
    
    sliding_window_size = (3,3)
    if 'sliding_window_size' in kwargs:
        sliding_window_size = kwargs['sliding_window_size']
    
    stride = 1
    if 'stride' in kwargs:
        stride = kwargs['stride']    
    
    pooling_func = 'max'
    if 'pooling_func' in kwargs:
        pooling_func = kwargs['pooling_func']   
        
    if  pooling_func == 'max':
        pool = lambda window: window.max()   
        
    if  pooling_func == 'mean':
        pool = lambda window: window.mean()  
        
    norm = 'l2'
    if 'norm' in kwargs:
        norm = kwargs['norm']     
    
    
    feat=[]

    new_kernels = []
    for ind, kernel in kernels.items():
        new_kernels.append(kernel)
    new_kernels = np.asarray(new_kernels)
    kernels = copy.deepcopy(new_kernels)

    for kernel in kernels:    
        
        # compute convolution (more precisely: cross-crorrelation)  
        conv = myConv(image, kernel)
        
        # shove result through relu - in place operation below
        np.maximum(0,conv,conv)
        
        # pooling
        conv_feats=[]
        for window in sliding_window(conv, sliding_window_size, stride):
            conv_feats.append(pool(window))
    
        feat.append(conv_feats)
    
    # feat = normalize(feat, norm=norm, axis=0)
        
    return np.reshape(feat, np.size(feat), 1)   


def scan_image(image, **kwargs):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    sliding_window_size = (3,3)
    if 'sliding_window_size' in kwargs:
        sliding_window_size = kwargs['sliding_window_size']
    
    stride = 1
    if 'stride' in kwargs:
        stride = kwargs['stride']         
     
    for i in np.arange(0, np.shape(image)[0]-sliding_window_size[0]+1, stride):
        for j in np.arange(0, np.shape(image)[1]-sliding_window_size[1]+1, stride):
            
            clone = image.copy()
            plt.show()
            ax.add_patch(
                patches.Rectangle(
                    (j, i),   # (x,y)
                    sliding_window_size[1],          # width
                    sliding_window_size[0],          # height
                    edgecolor="red",
                    linewidth=2,
                    facecolor="none"
                ))
            
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            time.sleep(0.025)
            
            

def show_conv_kernels():
    
    kernels = load_kernels()
    directions = load_directions()
   
    # initialize figure
    fig = plt.figure(figsize=(8,4))
    gs=GridSpec(2,4) 

    # plot convolutions
    for i in range(1, 9):
        ax = plt. subplot(gs[i-1])
        cmap_kernel = ["#34495e"]
        sns.heatmap(kernels[str(i)].astype('int'), square=True, cbar=False, cmap=cmap_kernel,
                        annot=True, fmt="d", linewidths=.1, yticklabels=False, xticklabels=False,
                        annot_kws={"weight": "bold"}, ax=ax)

        
        plt.xticks([]), plt.yticks([])
        plt.title(directions[str(i)], fontsize = 16)
    
    plt.show()  
    
    

     
            
    