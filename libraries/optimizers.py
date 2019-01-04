# clear display
from IPython.display import clear_output

# import autograd functionality
from autograd import grad as compute_grad  
import autograd.numpy as np
import math
import time
import copy
from autograd.misc.flatten import flatten_func
from autograd.misc.flatten import flatten


class Setup:
    '''
    Optimizer(s) for multilayer perceptron function
    '''    
        
    ########## optimizer ##########
    # gradient descent function
    def gradient_descent(self,g,w_unflat,alpha,max_its,version,**kwargs):
        verbose = False
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
            
        # flatten the input function, create gradient based on flat function
        g_flat, unflatten, w = flatten_func(g, w_unflat)
        grad = compute_grad(g)

        # record history
        w_hist = []
        w_hist.append(w_unflat)
            
        # over the line
        for k in range(max_its):   
            # plug in value into func and derivative
            grad_eval = grad(w_unflat)
            grad_eval, _ = flatten(grad_eval)

            ### normalized or unnormalized descent step? ###
            if version == 'normalized':
                grad_norm = np.linalg.norm(grad_eval)
                if grad_norm == 0:
                    grad_norm += 10**-6*np.sign(2*np.random.rand(1) - 1)
                grad_eval /= grad_norm

            # take descent step 
            w = w - alpha*grad_eval

            # record weight update
            w_unflat = unflatten(w)
            w_hist.append(w_unflat)

        if verbose == True:
            print ('...optimization complete!')
            time.sleep(1.5)
            clear_output()
            
        return w_hist