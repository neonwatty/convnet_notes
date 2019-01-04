import autograd.numpy as np
from timeit import default_timer as timer
import copy

class Setup:
    def __init__(self,kernel_sizes,**kwargs):
        # select kernel sizes and scale
        self.kernel_sizes = kernel_sizes
        self.scale = 0.1
        self.conv_stride = 2
        if 'scale' in kwargs:
            self.scale = kwargs['scale'] 
        if 'conv_stride' in kwargs:
            self.conv_stride = kwargs['conv_stride']
                
    # convolution function
    def conv_function(self,tensor_windows,kernels):
        a = np.tensordot(tensor_windows,kernels.T)
        return a

    # activation 
    def activation(self,tensor_window):
        return np.maximum(0,tensor_window)
    
    # sliding window for image augmentation
    def sliding_window_tensor(self,tensor,window_size,stride):
        # grab image size, set container for results
        image_size = tensor.shape[1]
        num_images = tensor.shape[0]
        num_kernels = self.kernels.shape[0]
        results = []

        #### gather indices for all tensor blocks ####
        batch_x = []
        batch_y = []
        # slide window over input image with given window size / stride and function
        for i in np.arange(0, image_size - window_size + 1, stride):
            for j in np.arange(0, image_size - window_size + 1, stride):
                # take a window of input tensor
                batch_x.append(i)
                batch_y.append(j)
        batch_inds = np.asarray([batch_x,batch_y])

        # grab indecies for single image
        b,m,n = tensor.shape
        K = int(np.floor(window_size/2.0))
        R = np.arange(0,K+2)     
        extractor_inds = R[:,None]*n + R + (batch_inds[0]*n+batch_inds[1])[:,None,None]

        # extend to the entire tensor
        base = [copy.deepcopy(extractor_inds)]
        ind_size = image_size**2
        for i in range(tensor.shape[0] - 1):
            base.append(extractor_inds + ((i+1)*ind_size))
        base = np.array(base) 

        # extract windows using numpy (to avoid for loops involving kernel weights)
        # tensor_windows = np.take(tensor,base)
        tensor_windows = tensor.flatten()[base]

        # process tensor windows
        results = self.conv_function(tensor_windows,self.kernels)
        results = results.swapaxes(0,2)
        results = results.swapaxes(1,2)

        return results 

    # make feature map
    def make_feature_tensor(self,tensor):
        # create feature map via convolution --> returns flattened convolution calculations
        feature_tensor = self.sliding_window_tensor(tensor,self.kernel_size,self.conv_stride) 
        
        # shove feature map through nonlinearity
        downsampled_feature_map = self.activation(feature_tensor)

        # return downsampled feature map --> flattened
        return downsampled_feature_map

    # convolution layer
    def conv_layer(self,tensor,kernels): 
        #### prep input tensor #####
        # pluck out dimensions for image-tensor reshape
        num_images = np.shape(tensor)[0]
        num_kernels = np.shape(kernels)[0]
        
        # create tensor out of input images (assumed to be stacked vertically as columns)
        tensor = np.reshape(tensor,(np.shape(tensor)[0],int((np.shape(tensor)[1])**(0.5)),int( (np.shape(tensor)[1])**(0.5))),order = 'F')
        
        # pad tensor
        kernel = kernels[0]
        self.kernel_size = np.shape(kernel)[0]
                
        #### prep kernels - reshape into array for more effecient computation ####
        self.kernels = kernels  #np.reshape(kernels,(np.shape(kernels)[0],np.shape(kernels)[1]*np.shape(kernels)[2]))
        
        #### compute convolution feature maps / downsample via pooling one map at a time over entire tensor #####
        # compute feature map for current image using current convolution kernel
        feature_tensor = self.make_feature_tensor(tensor)   
        #print ('----')
       # print (feature_tensor.shape)
        feature_tensor = feature_tensor.swapaxes(0,1)
        feature_tensor = np.reshape(feature_tensor, (np.shape(feature_tensor)[0],np.shape(feature_tensor)[1]*np.shape(feature_tensor)[2]),order = 'F')
        #print (feature_tensor.shape)
        
        return feature_tensor
        
    def conv_initializer(self):
        '''
        Initialization function: produces initializer to produce weights for 
        kernels and final layer touching fully connected layer
        '''
        # random initialization for kernels
        k0 = self.kernel_sizes[0]
        k1 = self.kernel_sizes[1]
        k2 = self.kernel_sizes[2]
        kernel_weights = self.scale*np.random.randn(k0,k1,k2) 
        return kernel_weights