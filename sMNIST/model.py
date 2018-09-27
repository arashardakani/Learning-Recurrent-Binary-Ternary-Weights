# Copyright 2018

# Contains code from BinaryConnect, Copyright 2015 Matthieu Courbariaux. All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with the codes.  If not, see <http://www.gnu.org/licenses/>.

import gzip
import _pickle
import numpy as np
import os
import os.path
import sys
import theano 
import theano.tensor as T
import time       
        
class Network(object):
    
    layer = []                
    
    def __init__(self, n_hidden_layer, BN_LSTM, length):
        
   
        self.n_hidden_layers = n_hidden_layer
        print ("    n_hidden_layers = "+str(n_hidden_layer))    
        self.BN_LSTM = BN_LSTM
        print ("    BN_LSTM = "+str(BN_LSTM))
        self.length = length


    
    def fprop(self, x, can_fit, eval):
        for k in range(0,self.n_hidden_layers+1):
            x = self.layer[k].fprop(x, can_fit, eval)
        
        return x


    # when you use fixed point, you cannot use T.grad directly -> bprop modifications.
    def bprop(self, y, t):
        
        yhat = T.nnet.softmax(y)
        cross_entropies = T.nnet.categorical_crossentropy(yhat, t)
        cost = cross_entropies.mean()

        for k in range(self.n_hidden_layers,-1,-1):
            self.layer[k].bprop(cost)

        
    def BN_reset(self):
        
        updates = self.layer[0].BN_reset()
        for k in range(1,self.n_hidden_layers+1):
            updates = updates + self.layer[k].BN_reset()
        
        return updates
    

    def parameters_updates(self, x, t, LR):
        x = x.reshape((x.shape[0], self.length + 0, 1))
        t = t.flatten(1)
        x = x.dimshuffle(1, 0, 2)
        x = x[0:, :, :]
        y = self.fprop(x=x, can_fit=True, eval=False) 

        self.bprop(y, t)

        # updates
        updates = self.layer[0].parameters_updates(LR)
        for k in range(1,self.n_hidden_layers+1):
            updates = updates + self.layer[k].parameters_updates(LR)

        return updates
        
    
    def errors(self, x, t):
        x = x.reshape((x.shape[0], self.length + 0, 1))
        t = t.flatten(1)
        x = x.dimshuffle(1, 0, 2)
        x = x[0:, :, :]
        y = self.fprop(x=x, can_fit=True,eval=False)
        yhat = T.nnet.softmax(y)
        errors = T.neq(t, T.argmax(yhat, axis=1))
        return errors.mean() 

    def fprop_update(self):

        updates = []
        updates = updates + self.updates
        
        return updates    
        

    def save_params_file(self, path):        
        
        # Open the file and overwrite current contents
        save_file = open(path, 'wb')
        
        # write all the parameters in the file
        for k in range(0,self.n_hidden_layers+1):
            if (k == 0):
                _pickle.dump(self.layer[k].Wx.get_value(), save_file, -1)
                _pickle.dump(self.layer[k].Wa.get_value(), save_file, -1)
                _pickle.dump(self.layer[k].bn_a_beta.get_value(), save_file, -1)
                _pickle.dump(self.layer[k].bn_c_beta.get_value(), save_file, -1)
                _pickle.dump(self.layer[k].h0.get_value(), save_file, -1)
                _pickle.dump(self.layer[k].c0.get_value(), save_file, -1)
                if self.BN_LSTM == True:
                    _pickle.dump(self.layer[k].bn_a_gamma.get_value(), save_file, -1)
                    _pickle.dump(self.layer[k].bn_b_gamma.get_value(), save_file, -1)
                    _pickle.dump(self.layer[k].bn_c_gamma.get_value(), save_file, -1)
            else:
                _pickle.dump(self.layer[k].W.get_value(), save_file, -1)
                _pickle.dump(self.layer[k].b.get_value(), save_file, -1)
            
            
        # close the file
        save_file.close()
        
    def load_params_file(self, path): 
        
        # Open the file
        save_file = open(path, 'rb')
        
        # read an load all the parameters
        for k in range(0,self.n_hidden_layers+1):
            if (k == 0):
                self.layer[k].Wx.set_value(_pickle.load(save_file))
                self.layer[k].Wa.set_value(_pickle.load(save_file))
                self.layer[k].bn_a_beta.set_value(_pickle.load(save_file))
                self.layer[k].bn_c_beta.set_value(_pickle.load(save_file))
                self.layer[k].h0.set_value(_pickle.load(save_file))
                self.layer[k].c0.set_value(_pickle.load(save_file))
                if self.BN_LSTM == True:
                    self.layer[k].bn_a_gamma.set_value(_pickle.load(save_file))
                    self.layer[k].bn_b_gamma.set_value(_pickle.load(save_file))
                    self.layer[k].bn_c_gamma.set_value(_pickle.load(save_file))
            else:
                self.layer[k].W.set_value(_pickle.load(save_file))
                self.layer[k].b.set_value(_pickle.load(save_file))

        # close the file
        save_file.close()
        
