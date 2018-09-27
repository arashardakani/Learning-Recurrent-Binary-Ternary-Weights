# Copyright 2018

# Contains code from BinaryConnect, Copyright 2015 Matthieu Courbariaux. All rights reserved.
# Contains code from DeepMind-Teaching-Machines-to-Read-and-Comprehend, Copyright (c) 2016 Thomas Mesnard. All rights reserved.

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


# TRAINING

class Trainer(object):
    
    def __init__(self,
            rng,
            train_set, valid_set, test_set, length,
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


        '''data = train_set.get_epoch_iterator(as_dict=True)
        train_set_in = next(data)

        data = valid_set.get_epoch_iterator(as_dict=True)
        valid_set_in = next(data) 

        data = test_set.get_epoch_iterator(as_dict=True)
        test_set_in = next(data)


        n_batches = len(list(train_set.get_epoch_iterator()))
        data = train_set.get_epoch_iterator(as_dict=True)
        z = [None for i in range(n_batches)]
        for i in range(n_batches):
            data_in = next(data)
            z[i] = data_in

        self.train_set = z

        n_batches = len(list(valid_set.get_epoch_iterator()))
        data = valid_set.get_epoch_iterator(as_dict=True)
        z = [None for i in range(n_batches)]
        for i in range(n_batches):
            data_in = next(data)
            z[i] = data_in

        self.valid_set = z


        n_batches = len(list(test_set.get_epoch_iterator()))
        data = test_set.get_epoch_iterator(as_dict=True)
        z = [None for i in range(n_batches)]
        for i in range(n_batches):
            data_in = next(data)
            z[i] = data_in

        self.test_set = z'''



        self.train_set = train_set#_in
        self.valid_set = valid_set#_in
        self.test_set = test_set#_in
        self.rng = rng
        
        
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
            if self.lr>self.final_lr:
                self.lr*=self.lr_decay
            
          
        # update the epoch counter
        self.epoch += self.step
        

        
        # test it on the validation set
        self.validation_ER = self.test_epoch(self.valid_set)
        
        # test it on the test set
        self.test_ER = self.test_epoch(self.test_set) 


        #if self.BN == True: 
        self.set_BN_mean_var()
        
        # save the best parameters
        if self.validation_ER <= self.best_validation_ER:
        
            self.best_validation_ER = self.validation_ER
            self.best_test_ER = self.test_ER
            self.best_epoch = self.epoch
            if self.save_path != None:
                self.model.save_params_file(self.save_path)

    
    
    def train_epoch(self, input_data):
        
        


        n_batches = len(list(input_data.get_epoch_iterator()))
        data = input_data.get_epoch_iterator(as_dict=True)


        #n_batches = len(input_data)
        ##data_in = input_data
        tmp = 0
        for i in range(n_batches):
            #data_in = input_data[i]
            data_in = next(data)
            self.train_batch(np.int32(data_in["question"]), np.float32(data_in["question_mask"]), np.int32(data_in["context"]), np.float32(data_in["context_mask"]),np.int32(data_in["candidates"]), np.int32(data_in["candidates_mask"]), np.int32(data_in["answer"]),self.lr)


            ##self.train_batch(np.int32(data_in["question"][0+32*i:32*(i+1),:]), np.float32(data_in["question_mask"][0+32*i:32*(i+1),:]), np.int32(data_in["candidates"][0+32*i:32*(i+1),:]), np.int32(data_in["candidates_mask"][0+32*i:32*(i+1),:]), np.int32(data_in["answer"][0+32*i:32*(i+1)]), self.lr)
            tmp += 1
            if ( ((tmp%1000)==0) & (tmp>0) ):
                print("iter:",tmp)
                self.validation_ER = self.test_epoch(self.valid_set)
                self.test_ER = self.test_epoch(self.test_set) 
                self.set_BN_mean_var()
                if self.validation_ER <= self.best_validation_ER:
                    self.best_validation_ER = self.validation_ER
                    self.best_test_ER = self.test_ER
                    self.best_epoch = self.epoch     
                    if self.save_path != None:
                        self.model.save_params_file(self.save_path)           
                self.monitor()
                

    
    def test_epoch(self, input_data):
                
        error_rate = 0.
        #n_batches = len(input_data)
        n_batches = len(list(input_data.get_epoch_iterator()))
        data = input_data.get_epoch_iterator(as_dict=True)
        ##data_in = input_data
        for i in range(n_batches):
            
            #data_in = input_data[i]
            data_in = next(data)
            #error_rate += self.test_batch(np.int32(np.concatenate((data_in["context"],data_in["question"]),axis=1)), np.float32(np.concatenate((data_in["context_mask"],data_in["question_mask"]),axis=1)), np.int32(data_in["candidates"]), np.int32(data_in["candidates_mask"]), np.int32(data_in["answer"]))

            #print(np.shape(data_in["context"]) )
            #print(np.shape(data_in["question_mask"]) )

            ##error_rate += self.test_batch(np.int32(data_in["question"][32*i:32*(i+1),:]), np.float32(data_in["question_mask"][32*i:32*(i+1),:]), np.int32(data_in["candidates"][32*i:32*(i+1),:]), np.int32(data_in["candidates_mask"][32*i:32*(i+1),:]), np.int32(data_in["answer"][32*i:32*(i+1)]))  
            error_rate += self.test_batch(np.int32(data_in["question"]), np.float32(data_in["question_mask"]), np.int32(data_in["context"]), np.float32(data_in["context_mask"]),np.int32(data_in["candidates"]), np.int32(data_in["candidates_mask"]), np.int32(data_in["answer"]))
            #x = self.test_batch(np.int32(data_in["question"]), np.float32(data_in["question_mask"]), np.int32(data_in["candidates"]), np.int32(data_in["candidates_mask"]), np.int32(data_in["answer"]))
            #print(  x  )
            #np.save('./h.npy',x)
            #np.save('./mask.npy',data_in["question_mask"])
            #sys.exit(1)

        #size_ = np.size(input_data.get_data())
        error_rate /= (n_batches*32)
        return error_rate
    
    def monitor(self):
    
        print ('epoch %i:' %(self.epoch))
        print ('    learning rate %f' %(self.lr))
        print ('    validation error rate %f%%' %(self.validation_ER))
        print ('    test error rate %f%%' %(self.test_ER))
        print ('    epoch associated to best error rate %i' %(self.best_epoch))
        print ('    best validation error rate %f%%' %(self.best_validation_ER))
        print ('    test error rate associated to best error rate %f%%' %(self.best_test_ER))
        
    
    def build(self):
        
        # input and output variables
        que = T.imatrix('que')
        que_m = T.fmatrix('que_m')
        con = T.imatrix('con')
        con_m = T.fmatrix('con_m')
        cand = T.imatrix('cand')
        cand_m = T.imatrix('cand_m')
        y = T.ivector('y')
        lr = T.scalar('lr', dtype=theano.config.floatX)
        self.train_batch = theano.function(inputs=[que,que_m, con,con_m,cand,cand_m,y,lr], updates=self.model.parameters_updates(que,que_m, con,con_m,cand,cand_m,y, lr), name = "train_batch", on_unused_input='warn')
        self.test_batch = theano.function(inputs = [que,que_m, con,con_m,cand,cand_m,y],  outputs=self.model.errors(que,que_m, con,con_m,cand,cand_m,y), name = "test_batch", on_unused_input='warn')
        self.reset_mean_var = theano.function(inputs = [], updates=self.model.BN_reset(), name = "reset_mean_var")  
