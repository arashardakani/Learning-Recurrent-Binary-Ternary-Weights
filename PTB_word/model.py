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
    
    def __init__(self, n_hidden_layer, BN_LSTM):
        
   
        self.n_hidden_layers = n_hidden_layer
        print ("    n_hidden_layers = "+str(n_hidden_layer))    
        self.BN_LSTM = BN_LSTM
        print ("    BN_LSTM = "+str(BN_LSTM)) 


    def fprop1(self, x, h0, c0, can_fit, eval):
        h, c = h0, c0
        for k in range(0,self.n_hidden_layers):
            x, h_temp, c_temp = self.layer[k].fprop1(x, h0[:,:,k], c0[:,:,k], can_fit, eval)
            h = T.set_subtensor(h[:,:,k], h_temp)
            c = T.set_subtensor(c[:,:,k], c_temp)
        x = self.layer[self.n_hidden_layers].fprop(x, can_fit, eval)
        return x, h, c


    # when you use fixed point, you cannot use T.grad directly -> bprop modifications.
    def bprop(self, y, t):
        yhat = T.nnet.softmax(y.reshape((-1, y.shape[-1])))
        cross_entropies = -(t.reshape((-1, t.shape[-1])) * T.log(yhat)).sum(axis=yhat.ndim - 1)
        cost = cross_entropies.mean()
        norm = 0.
        for k in range(self.n_hidden_layers,-1,-1):
            norm = norm + self.layer[k].bprop(cost)
        norm = norm ** (1. / 2)
        clip_coef = 0.25 / (norm + 1e-6)
        clip_coef = T.clip(clip_coef, 0., 1.)
        return clip_coef

        
    def BN_reset(self):
        
        updates = self.layer[0].BN_reset()
        for k in range(1,self.n_hidden_layers+1):
            updates = updates + self.layer[k].BN_reset()
        
        return updates
    

    def parameters_updates(self, x,y, h0, c0, LR):
        t = y
        eye = T.eye(10000)
        t = eye[t]
        y, h, c = self.fprop1(x=x, h0= h0, c0=c0, can_fit=True, eval=False) 
        norm_factor = self.bprop(y, t)

        updates = self.layer[0].parameters_updates(LR,norm_factor)
        for k in range(1,self.n_hidden_layers+1):
            updates = updates + self.layer[k].parameters_updates(LR, norm_factor)
        return updates
        
    def parameters_updates1(self, x,y, h0, c0, LR):
        t = y
        eye = T.eye(10000)
        t = eye[t]
        y, h, c  = self.fprop1(x=x, h0= h0, c0=c0, can_fit=True, eval=False) 

        return h, c
    
    def errors(self, x, y, h0, c0):
        t = y
        eye = T.eye(10000)
        t = eye[t]
        y, h, c = self.fprop1(x=x, h0= h0, c0=c0, can_fit=True,eval=True)
        yhat = T.nnet.softmax(y.reshape((-1, y.shape[-1])))
        cross_entropies = -(t.reshape((-1, t.shape[-1])) * T.log(yhat)).sum(axis=yhat.ndim - 1)
        cost = cross_entropies.sum()
        return cost, h, c

    def fprop_update(self):

        updates = []
        updates = updates + self.updates
        
        return updates    
        
    def save_params_file(self, path):        
        
        save_file = open(path, 'wb')
        
        for k in range(0,self.n_hidden_layers+1):
            if (k == 0):
                _pickle.dump(self.layer[k].Wx.get_value(), save_file, -1)
                _pickle.dump(self.layer[k].Wa.get_value(), save_file, -1)
                _pickle.dump(self.layer[k].bn_a_beta.get_value(), save_file, -1)
                _pickle.dump(self.layer[k].bn_c_beta.get_value(), save_file, -1)
                _pickle.dump(self.layer[k].h0.get_value(), save_file, -1)
                _pickle.dump(self.layer[k].c0.get_value(), save_file, -1)
                if self.BN == True:
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
                self.layer[k].We.set_value(np.asarray(_pickle.load(save_file)))
                self.layer[k].Wx.set_value(np.transpose(np.asarray(_pickle.load(save_file))))
                self.layer[k].Wa.set_value(np.transpose(np.asarray(_pickle.load(save_file))))
                self.layer[k].bn_a_beta.set_value(np.asarray(_pickle.load(save_file)))
                self.layer[k].bn_b_beta.set_value(np.asarray(_pickle.load(save_file)))
                if self.BN == True:
                    self.layer[k].bn_a_gamma.set_value(_pickle.load(save_file))
                    self.layer[k].bn_b_gamma.set_value(_pickle.load(save_file))
                    self.layer[k].bn_c_gamma.set_value(_pickle.load(save_file))
            else:
                self.layer[k].W.set_value(np.transpose(np.asarray(_pickle.load(save_file))))
                self.layer[k].b.set_value(np.asarray(_pickle.load(save_file)))


        # close the file
        save_file.close()
        
