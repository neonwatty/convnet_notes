import autograd.numpy as np
from timeit import default_timer as timer
import copy

class Setup:
    def __init__(self,kernel_sizes,**kwargs):
        # select kernel sizes and scale
        self.kernel_sizes = kernel_sizes
        self.o = np.ones((kernel_sizes[-1],kernel_sizes[-1]))
        self.scale = 0.1
        self.conv_stride = 1
        self.pool_stride = 2
        if 'scale' in kwargs:
            self.scale = kwargs['scale'] 
        if 'conv_stride' in kwargs:
            self.conv_stride = kwargs['conv_stride']
        if 'pool_stride' in kwargs:
            self.pool_stride = kwargs['pool_stride']
            
    # convolution function
    def conv_function(self,tensor_windows):
        # compute convolutions
        a = np.tensordot(tensor_windows,self.kernels.T)

        # swap axes to match up with earlier versions
        a = a.swapaxes(0,2)
        a = a.swapaxes(1,2)
        return a
    
    # pooling / downsampling parameters
    def pool_function(self,tensor_windows):
        #t = np.max(tensor_windows,axis = (2,3))
        t = np.tensordot(tensor_windows,self.o)/float(np.size(self.o))
        return t

    # activation 
    def activation(self,tensor_windows):
        return np.maximum(0,tensor_windows)

    # sliding window for image augmentation
    def sliding_window_tensor(self,tensor,window_size,stride,operation):
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
        tensor_windows = tensor.flatten()[base]

        # process tensor windows
        results = []
        if operation == 'convolution':
            results = self.conv_function(tensor_windows)
        if operation == 'pool':
            #print (tensor_windows.shape)
            results = self.pool_function(tensor_windows)
            #print (results.shape)
        return results 

    # make feature map
    def make_feature_tensor(self,tensor):
        ##### convolution #####
        window_size = self.kernel_sizes[1]
        stride = self.conv_stride
        operation = 'convolution'
        feature_tensor = self.sliding_window_tensor(tensor,window_size,stride,operation)
        
        ##### take maximum of convolution output #####
        feature_tensor = self.activation(feature_tensor)
        
        ##### pooling step #####
        # re-shape convolution output ---> append all feature maps together
        num_filters = self.kernel_sizes[0]
        num_images = tensor.shape[0]
        square_dim = int((feature_tensor.shape[2]**(0.5)))
        feature_tensor = np.reshape(feature_tensor,(num_filters*num_images,square_dim,square_dim),order = 'C')    
        
        # pooling step
        window_size = self.kernel_sizes[1]
        stride = self.pool_stride
        operation = 'pool'
        downsampled_feature_map = self.sliding_window_tensor(feature_tensor,window_size,stride,operation)
        
        # return downsampled feature map --> flattened
        return downsampled_feature_map
    
    # pad image with appropriate number of zeros for convolution
    def pad_tensor(self,tensor,kernel_size):
        odd_nums = np.array([int(2*n + 1) for n in range(100)])
        pad_val = np.argwhere(odd_nums == kernel_size)[0][0]
        tensor_padded = np.zeros((np.shape(tensor)[0], np.shape(tensor)[1] + 2*pad_val,np.shape(tensor)[2] + 2*pad_val))
        tensor_padded[:,pad_val:-pad_val,pad_val:-pad_val] = tensor
        return tensor_padded    
    
    # convolution layer
    def conv_layer(self,tensor,kernels): 
        #### prep input tensor #####
        # pluck out dimensions for image-tensor reshape
        num_images = np.shape(tensor)[0]
        num_kernels = np.shape(kernels)[0]
        
        # create tensor out of input images (assumed to be stacked vertically as columns)
        tensor = np.reshape(tensor,(np.shape(tensor)[0],int((np.shape(tensor)[1])**(0.5)),int( (np.shape(tensor)[1])**(0.5))),order = 'F')
        
        # pad tensor
        padded_tensor = self.pad_tensor(tensor,kernels.shape[2])
                
        # make kernels universal element in class
        self.kernels = kernels 
        
        #### compute convolution feature maps / downsample via pooling one map at a time over entire tensor #####
        # compute feature map for current image using current convolution kernel
        feature_tensor = self.make_feature_tensor(padded_tensor)   

        # reshape appropriately
        
        ind1 = int(feature_tensor.shape[0]/float(self.kernels.shape[0]))
        ind2 = feature_tensor.shape[1]*self.kernels.shape[0]
        feature_tensor = np.reshape(feature_tensor,(ind1,ind2),order = 'F')
        
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