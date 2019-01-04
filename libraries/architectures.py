# import autograd functionality
from autograd import grad as compute_grad  
import autograd.numpy as np
import copy

class Setup:
    '''
    Normalized multilayer perceptron / feedforward network architectures
    '''
 
    def choose_architecture(self,activation_name):
        self.activation_name = activation_name
        
        # set activation
        if activation_name == 'relu':
            self.activation = self.relu
        if activation_name == 'maxout':
            self.activation = self.maxout
        if activation_name == 'tanh':
            self.activation = self.tanh
        if activation_name == 'linear':
            self.activation = self.linear
            
        # set architecture and initializer (basically just a switch for maxout vs others)
        self.training_architecture = self.compute_general_network_features
        self.initializer = self.initialize_general_network_weights
        self.testing_architecture = self.compute_network_features_testing
        if self.activation_name == 'maxout':
            self.training_architecture = self.compute_maxout_network_features            
            self.initializer = self.initialize_maxout_network_weights   
            self.testing_architecture = self.compute_maxout_network_features_testing
            
    ####### convolution layer #######
    # sliding window for image augmentation
    def sliding_window_tensor(self,tensor, kernel_size, stride):
        windowed_tensor = []
        for i in np.arange(0, np.shape(tensor)[1]-kernel_size[0]+1, stride):
            for j in np.arange(0, np.shape(tensor)[2]-kernel_size[1]+1, stride):
                sock = copy.deepcopy(tensor[:,i:i+kernel_size[0], j:j+kernel_size[1]])
                windowed_tensor.append(sock)
        
        # re-shape properly 
        windowed_tensor = np.array(windowed_tensor)
        windowed_tensor = windowed_tensor.swapaxes(0,1)
        windowed_tensor = np.reshape(windowed_tensor,(np.shape(windowed_tensor)[0]*np.shape(windowed_tensor)[1],np.shape(windowed_tensor)[2]*np.shape(windowed_tensor)[3])) 
        
        return windowed_tensor

    # pad image with appropriate number of zeros for convolution
    def pad_tensor(self,tensor,kernel_size):
        odd_nums = np.asarray([int(2*n + 1) for n in range(100)])
        val = kernel_size[0]
        pad_val = np.argwhere(odd_nums == val)[0][0]
        tensor_padded = np.zeros((np.shape(tensor)[0], np.shape(tensor)[1] + 2*pad_val,np.shape(tensor)[2] + 2*pad_val))
        tensor_padded[:,pad_val:-pad_val,pad_val:-pad_val] = tensor
        return tensor_padded    


    def conv_layer(self,tensor,kernels):
        # square up tensor into tensor of patches
        tensor = np.reshape(tensor,(np.shape(tensor)[0],int((np.shape(tensor)[1])**(0.5)),int( (np.shape(tensor)[1])**(0.5))),order = 'F')

        # pad tensor
        kernel = kernels[0]
        kernel_size = kernel.shape
        padded_tensor = self.pad_tensor(tensor,kernel_size)

        # window tensor
        wind_tensor = self.sliding_window_tensor(padded_tensor,kernel_size,stride = 1)

        # normalize windows since they touch weights
       # a_means = np.mean(wind_tensor,axis = 0)
       # a_stds = np.std(wind_tensor,axis = 0)
       # wind_tensor = self.normalize(wind_tensor,a_means,a_stds)

        #### compute convolution feature maps / downsample via pooling one map at a time over entire tensor #####
        kernel2 = np.ones((6,6))
        stride = 3
        new_tensors = []
        for kernel in kernels:
            #### make convolution feature map - via matrix multiplication over windowed tensor 
            feature_map = np.dot(wind_tensor,kernel.flatten()[:,np.newaxis])
            
            # reshape convolution feature map into array
            feature_map = np.reshape(feature_map,np.shape(tensor))
            
            # now shove result through nonlinear activation
            feature_map = self.activation(feature_map)
            
            #### now pool / downsample feature map, first window then pool on each window
            wind_featmap = self.sliding_window_tensor(feature_map,kernel2.shape,stride = stride)
            
            # max pool on each collected patch
            ### mean or max on each dude
            max_pool = np.max(wind_featmap,axis = 1)
            
            # reshape into new tensor
            max_pool = np.reshape(max_pool,(np.shape(tensor)[0],int((np.shape(max_pool)[0]/float(np.shape(tensor)[0]))**(0.5)),int((np.shape(max_pool)[0]/float(np.shape(tensor)[0]))**(0.5))))

            # reshape into new downsampled pooled feature map
            new_tensors.append(max_pool)

        # turn into array
        new_tensors = np.array(new_tensors)

        # reshape into final feature vector to touch fully connected layer(s), otherwise keep as is in terms of shape
        new_tensors = new_tensors.swapaxes(0,1)
        new_tensors = np.reshape(new_tensors, (np.shape(new_tensors)[0],np.shape(new_tensors)[1],np.shape(new_tensors)[2]*np.shape(new_tensors)[3]))
        new_tensors = np.reshape(new_tensors, (np.shape(new_tensors)[0],np.shape(new_tensors)[1]*np.shape(new_tensors)[2]),order = 'F')
        return new_tensors
    
    # our normalization function
    def normalize(self,data,data_mean,data_std):
        normalized_data = (data - data_mean)/(data_std + 10**(-5))
        return normalized_data

    ########## architectures ##########
    def compute_general_network_features(self,x,inner_weights,kernels):
        # pass input through convolution layers
        x_conv = self.conv_layer(x,kernels)

        # pad data with ones to deal with bias
        o = np.ones((np.shape(x_conv)[0],1))
        a_padded = np.concatenate((o,x_conv),axis = 1)

        # loop through weights and update each layer of the network
        for W in inner_weights:            
            print (np.shape(a_padded))
            print (np.shape(W))

            
            
            # output of layer activation
            a = self.activation(np.dot(a_padded,W))

            ### normalize output of activation
            # compute the mean and standard deviation of the activation output distributions
            #a_means = np.mean(a,axis = 0)
            #a_stds = np.std(a,axis = 0)

            # normalize the activation outputs
            #a_normed = self.normalize(a,a_means,a_stds)

            a_normed = a
            
            # pad with ones for bias
            o = np.ones((np.shape(a_normed)[0],1))
            a_padded = np.concatenate((o,a_normed),axis = 1)
        print (np.shape(inner_weights))
        print (a_normed)

        return a_padded

    def compute_maxout_network_features(self,x,inner_weights,kernels):
        # pass input through convolution layers
        x_conv = self.conv_layer(x,kernels)
        
        # pad data with ones to deal with bias
        o = np.ones((np.shape(x_conv)[0],1))
        a_padded = np.concatenate((o,x_conv),axis = 1)

        # loop through weights and update each layer of the network
        for W1,W2 in inner_weights:                                 
            # output of layer activation  
            a = self.activation(np.dot(a_padded,W1),np.dot(a_padded,W2))  

            ### normalize output of activation
            # compute the mean and standard deviation of the activation output distributions
            a_means = np.mean(a,axis = 0)
            a_stds = np.std(a,axis = 0)

            # normalize the activation outputs
            a_normed = self.normalize(a,a_means,a_stds)

            # pad with ones for bias
            o = np.ones((np.shape(a_normed)[0],1))
            a_padded = np.concatenate((o,a_normed),axis = 1)

        return a_padded

    ########## test versions of the architecture to extract stats ##########
    def conv_layer_testing(self,tensor,kernels,stats):
        # square up tensor into tensor of patches
        tensor = tensor.reshape(np.shape(tensor)[0],int((np.shape(tensor)[1])**(0.5)),int( (np.shape(tensor)[1])**(0.5)),order = 'F')

        # pad tensor
        kernel = kernels[0]
        padded_tensor = self.pad_tensor(tensor,kernel)

        # window tensor
        wind_tensor = self.sliding_window_tensor(padded_tensor,kernel,stride = 1)

        # normalize windows since they touch weights
        a_means = 0
        a_stds = 0
        if np.size(stats) == 0:
            a_means = np.mean(wind_tensor,axis = 0)
            a_stds = np.std(wind_tensor,axis = 0)
            stats = [a_means,a_stds]
        else:
            a_means = stats[0][0]
            a_stds = stats[0][1]
        wind_tensor = self.normalize(wind_tensor,a_means,a_stds)

        #### compute convolution feature maps / downsample via pooling one map at a time over entire tensor #####
        kernel2 = np.ones((6,6))
        stride = 3
        new_tensors = []
        for kernel in kernels:
            #### make convolution feature map - via matrix multiplication over windowed tensor 
            feature_map = np.dot(wind_tensor,kernel.flatten()[:,np.newaxis])

            # reshape convolution feature map into array
            feature_map.shape = (np.shape(tensor))
            feature_map = np.asarray(feature_map)

            # now shove result through nonlinear activation
            feature_map = self.activation(feature_map)

            #### now pool / downsample feature map, first window then pool on each window
            wind_featmap = self.sliding_window_tensor(feature_map,kernel2,stride = stride)

            # max pool on each collected patch
            max_pool = np.max(wind_featmap,axis = 1)

            # reshape into new tensor
            max_pool.shape = (np.shape(tensor)[0],int((np.shape(max_pool)[0]/float(np.shape(tensor)[0]))**(0.5)),int((np.shape(max_pool)[0]/float(np.shape(tensor)[0]))**(0.5)))

            # reshape into new downsampled pooled feature map
            new_tensors.append(max_pool)

        # turn into array
        new_tensors = np.asarray(new_tensors)

        # reshape into final feature vector to touch fully connected layer(s), otherwise keep as is in terms of shape
        new_tensors = new_tensors.swapaxes(0,1)
        new_tensors = np.reshape(new_tensors, (np.shape(new_tensors)[0],np.shape(new_tensors)[1],np.shape(new_tensors)[2]*np.shape(new_tensors)[3]))
        new_tensors = np.reshape(new_tensors, (np.shape(new_tensors)[0],np.shape(new_tensors)[1]*np.shape(new_tensors)[2]),order = 'F')

        return new_tensors,stats
    
    
    def compute_network_features_testing(self,x,inner_weights,kernels,stats):
        '''
        An adjusted normalized architecture compute function that collects network statistics as the training data
        passes through each layer, and applies them to properly normalize test data.
        '''
        # are you using this to compute stats on training data (stats empty) or to normalize testing data (stats not empty)
        switch =  'testing'
        if np.size(stats) == 0:
            switch = 'training'

        # pass input through convolution layers
        x_conv,conv_stats = self.conv_layer_testing(x,kernels,stats)
        if switch == 'training':  
            stats.append([conv_stats[0],conv_stats[1]])
                
        # pad data with ones to deal with bias
        o = np.ones((np.shape(x_conv)[0],1))
        a_padded = np.concatenate((o,x_conv),axis = 1)

        # loop through weights and update each layer of the network
        c = 1
        for W in inner_weights:
            # output of layer activation
            a = self.activation(np.dot(a_padded,W))

            ### normalize output of activation
            a_means = 0
            a_stds = 0
            if switch == 'training':
                # compute the mean and standard deviation of the activation output distributions
                a_means = np.mean(a,axis = 0)
                a_stds = np.std(a,axis = 0)
                stats.append([a_means,a_stds])
            elif switch == 'testing':
                a_means = stats[c][0]
                a_stds = stats[c][1]

            # normalize the activation outputs
            a_normed = self.normalize(a,a_means,a_stds)

            # pad with ones for bias
            o = np.ones((np.shape(a_normed)[0],1))
            a_padded = np.concatenate((o,a_normed),axis = 1)
            c+=1

        return a_padded,stats

    def compute_maxout_network_features_testing(self,x,inner_weights,kernels,stats):
        '''
        An adjusted normalized architecture compute function that collects network statistics as the training data
        passes through each layer, and applies them to properly normalize test data.
        '''
            
        # are you using this to compute stats on training data (stats empty) or to normalize testing data (stats not empty)
        switch =  'testing'
        if np.size(stats) == 0:
            switch = 'training'

        # pass input through convolution layers
        x_conv,conv_stats = self.conv_layer_testing(x,kernels,stats)
        if switch == 'training':  
            stats.append([conv_stats[0],conv_stats[1]])

        # pad data with ones to deal with bias
        o = np.ones((np.shape(x)[0],1))
        a_padded = np.concatenate((o,x),axis = 1)

        # loop through weights and update each layer of the network
        c = 0
        for W1,W2 in inner_weights:                                  
            # output of layer activation
            a = self.activation(np.dot(a_padded,W1),np.dot(a_padded,W2))     

            ### normalize output of activation
            a_means = 0
            a_stds = 0
            if switch == 'training':
                # compute the mean and standard deviation of the activation output distributions
                a_means = np.mean(a,axis = 0)
                a_stds = np.std(a,axis = 0)
                stats.append([a_means,a_stds])
            elif switch == 'testing':
                a_means = stats[c][0]
                a_stds = stats[c][1]

            # normalize the activation outputs
            a_normed = self.normalize(a,a_means,a_stds)

            # pad with ones for bias
            o = np.ones((np.shape(a_normed)[0],1))
            a_padded = np.concatenate((o,a_normed),axis = 1)
            c+=1

        return a_padded,stats

    ########## weight initializers ##########
    # create initial weights for arbitrary feedforward network
    def initialize_general_network_weights(self,layer_sizes,num_kernels,scale):
        # container for entire weight tensor
        weights = []
        kernel_weights = []
        
        # loop over desired kernel sizes and create appropriately sized initial 
        # weight matrix for each kernel
        for k in range(num_kernels):
            # make weight matrix
            weight = scale*np.random.randn(3,3)
            kernel_weights.append(weight)
        kernel_weights = np.asarray(kernel_weights)
        
        # loop over desired layer sizes and create appropriately sized initial 
        # weight matrix for each layer
        for k in range(len(layer_sizes)-1):
            # get layer sizes for current weight matrix
            U_k = layer_sizes[k]
            U_k_plus_1 = layer_sizes[k+1]

            # make weight matrix
            weight = scale*np.random.randn(U_k + 1,U_k_plus_1)
            weights.append(weight)

        # re-express weights so that w_init[0] = omega_inner contains all 
        # internal weight matrices, and w_init[1] = w contains weights of 
        # final linear combination in predict function
        w_init = [weights[:-1],kernel_weights,weights[-1]]
        return w_init
    
    # create initial weights for maxout feedforward network
    def initialize_maxout_network_weights(self,layer_sizes,num_kernels,scale):
        # container for entire weight tensor
        weights = []
        kernel_weights = []
        
        # loop over desired kernel sizes and create appropriately sized initial 
        # weight matrix for each kernel
        for k in range(num_kernels):
            # make weight matrix
            weight = scale*np.random.randn(3,3)
            kernel_weights.append(weight)
        kernel_weights = np.asarray(kernel_weights)
        
        # loop over desired layer sizes and create appropriately sized initial 
        # weight matrix for each layer
        for k in range(len(layer_sizes)-1):
            # get layer sizes for current weight matrix
            U_k = layer_sizes[k]
            U_k_plus_1 = layer_sizes[k+1]

            # make weight matrix
            weight1 = scale*np.random.randn(U_k + 1,U_k_plus_1)

            # add second matrix for inner weights
            if k < len(layer_sizes)-2:
                weight2 = scale*np.random.randn(U_k + 1,U_k_plus_1)
                weights.append([weight1,weight2])
            else:
                weights.append(weight1)

        # re-express weights so that w_init[0] = omega_inner contains all 
        # internal weight matrices, and w_init = w contains weights of 
        # final linear combination in predict function
        w_init = [weights[:-1],kernel_weights,weights[-1]]

        return w_init
    
    ########## activation functions ##########
    def maxout(self,t1,t2):
        # maxout activation
        f = np.maximum(t1,t2)
        return f
    
    def relu(self,t):
        # relu activation
        f = np.maximum(0,t)
        return f    
    
    def tanh(self,t):
        # tanh activation
        f = np.tanh(t)
        return f    
    
    def linear(self,t):
        # linear activation
        f = t
        return f 