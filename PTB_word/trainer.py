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
import gc


class Trainer(object):
    
    def __init__(self,
            rng,
            train_set, valid_set, test_set, length, n_units,n_hidden_layer,
            model, save_path, load_path,
            lr, lr_decay, final_lr,
            batch_size, n_epoch, monitor_step):
        
        print ('    Learning rate = %f' %(lr))
        print ('    Learning rate decay = %f' %(lr_decay))
        print ('    LR_fin = %f' %(final_lr))
             
        
        self.batch_size = batch_size
        print ('    batch size = %i' %(batch_size))
        
        print ('    Number of epochs = %i' %(n_epoch))
        print ('    Monitor step = %i' %(monitor_step))
        self.length = length
        print ('    sequence length = %i' %(length))
        
        # save the dataset
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.rng = rng
        self.n_units = n_units
        
        
        # save the model
        self.model = model
        self.load_path = load_path
        self.save_path = save_path
        
        # save the parameters
        self.lr = lr
        self.lr_decay = lr_decay
        self.final_lr = final_lr
        self.n_epoch = n_epoch
        self.step = monitor_step
        self.n_hidden_layer = n_hidden_layer


    
    def set_BN_mean_var(self):
            
        # reset cumulative mean and var
        self.reset_mean_var()
        return
    

    def init(self):
        
        if self.load_path != None:
            self.model.load_params_file(self.load_path)
        
        self.epoch = 0
        self.best_epoch = self.epoch
        
        # test it on the validation set
        self.validation_ER = self.test_epoch(self.valid_set)
        # test it on the test set
        self.test_ER = self.test_epoch(self.test_set)

        #if self.BN == True: 
        self.set_BN_mean_var()

        
        self.best_validation_ER = self.validation_ER
        self.best_test_ER = self.test_ER
            
    
    def train(self):        
        
        self.init()
        self.monitor()
        
        while (self.epoch<self.n_epoch):
            
            self.update()   
            self.monitor()
    
    def update(self):
        
        
        
        for k in range(self.step):
        
            # train the model on all training examples
            self.train_epoch(self.train_set)
            
          
        # update the epoch counter
        self.epoch += self.step
        

        
        # test it on the validation set
        self.validation_ER = self.test_epoch(self.valid_set)
        
        # test it on the test set
        self.test_ER = self.test_epoch(self.test_set) 

        
        # save the best parameters
        if self.validation_ER <= self.best_validation_ER:
        
            self.best_validation_ER = self.validation_ER
            self.best_test_ER = self.test_ER
            self.best_epoch = self.epoch
            if self.save_path != None:
                self.model.save_params_file(self.save_path)
        else:
            self.lr/= 4.0


    def get_batch(self,source, i):
        seq_len = min(self.length, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target


    
    def train_epoch(self, input_data):
        
        h0 = np.zeros((self.batch_size, self.n_units,self.n_hidden_layer)).astype(theano.config.floatX)
        c0 = np.zeros((self.batch_size, self.n_units,self.n_hidden_layer)).astype(theano.config.floatX)
        for i in range(0, input_data.size(0) - 1, self.length):
            data, targets = self.get_batch(input_data, i)
            h0, c0 = self.train_batch(np.int32(data),np.int32(np.reshape(targets,np.shape(data))),h0,c0,self.lr) 


    
    def test_epoch(self, input_data):
        
        error_rate = float(0.)
        h0 = np.zeros((self.batch_size, self.n_units,self.n_hidden_layer)).astype(theano.config.floatX)
        c0 = np.zeros((self.batch_size, self.n_units,self.n_hidden_layer)).astype(theano.config.floatX)
        for i in range(0, input_data.size(0) - 1, self.length):
            data, targets = self.get_batch(input_data, i)
            cost, h0, c0 = self.test_batch(np.int32(data),np.int32(np.reshape(targets,np.shape(data))),h0,c0)
            error_rate += cost 

        size_ = np.shape(input_data)
        
        error_rate = error_rate / (size_[0] * size_[1])     #(size_*self.length) #(n_batches*self.batch_size*self.length)

        
        return np.exp(error_rate)
        
    
    def monitor(self):
    
        print ('    epoch %i:' %(self.epoch))
        print ('        learning rate %f' %(self.lr))
        print ('        validation error rate %f%%' %(self.validation_ER))
        print ('        test error rate %f%%' %(self.test_ER))
        print ('        epoch associated to best validation error %i' %(self.best_epoch))
        print ('        best validation error rate %f%%' %(self.best_validation_ER))
        print ('        test error rate associated to best validation error %f%%' %(self.best_test_ER))
    
    def build(self):
        
        x = T.imatrix('x')
        y = T.imatrix('y')
        h0 = T.ftensor3('h0')
        c0 = T.ftensor3('c0')
        index = T.scalar('index', dtype='int64')
        lr = T.scalar('LR', dtype=theano.config.floatX)
        M = T.scalar('M', dtype=theano.config.floatX)

        self.train_batch = theano.function(inputs=[x,y,h0,c0,lr], outputs = self.model.parameters_updates1(x,y,h0,c0,lr), updates=self.model.parameters_updates(x,y,h0,c0,lr), name = "train_batch", on_unused_input='warn')


        self.test_batch = theano.function(inputs = [x,y,h0,c0],  outputs=self.model.errors(x,y,h0,c0), name = "test_batch", on_unused_input='warn')

        self.reset_mean_var = theano.function(inputs = [], updates=self.model.BN_reset(), name = "reset_mean_var")  
 
               
