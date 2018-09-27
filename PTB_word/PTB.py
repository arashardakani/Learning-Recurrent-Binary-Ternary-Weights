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
import time
import fuel.datasets, fuel.streams, fuel.transformers, fuel.schemes
import theano
from trainer import Trainer
from model import Network
from layer import linear_layer, LSTM



import torch
import torch.nn as nn

import data


_data_cache = dict()


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


device = torch.device("cuda")

if __name__ == "__main__":
    
    rng = np.random.RandomState(1234)
    batch_size = 64
    BN_LSTM = True
    BN_epsilon=1e-5
    BN_fast_eval= True
    dropout_input = 0.5
    length = 35
    initial_gamma = 1e-1
    initial_beta = 0

    # Termination criteria
    n_epoch = 80
    monitor_step = 1
    
    # LR
    lr = 20
    final_lr = 20
    lr_decay = (final_lr/lr)**(1./n_epoch)
    
    # architecture
    n_inputs = 10000
    n_units = 300
    n_classes = 10000
    n_hidden_layer = 1
    mask_p = -1.
    
    # binary/ternary weights for LSTM
    binary_training = False
    ternary_training = False
    stochastic_training = True
    
    print ('Loading the dataset') 
    corpus = data.Corpus('./data/wikitext-2')
    train_set = batchify(corpus.train, batch_size)
    valid_set = batchify(corpus.valid, batch_size)
    test_set  = batchify(corpus.test, batch_size)


    print ('Creating the model')
    
    class create_model(Network):

        def __init__(self, rng):
            
            Network.__init__(self, n_hidden_layer = n_hidden_layer, BN_LSTM = BN_LSTM)
            
            print ("LSTM layer:")
            self.layer.append(LSTM(rng = rng, n_inputs = n_inputs, n_units = n_units, initial_gamma = initial_gamma, 
                initial_beta = initial_beta, length = length - 1, batch_size = batch_size, BN = BN_LSTM, BN_epsilon=BN_epsilon,
                dropout=dropout_input, binary_training=binary_training, ternary_training = ternary_training, 
                stochastic_training = stochastic_training))
            
            print ("Softmax layer:")
            self.layer.append(linear_layer(rng = rng, n_inputs = n_units, n_units = n_classes, dropout=dropout_input))
    
    model = create_model(rng = rng)
    
    print ('Creating the trainer')
    
    trainer = Trainer(rng = rng,
        train_set = train_set, valid_set = valid_set, test_set = test_set, length = length, n_units = n_units, n_hidden_layer = n_hidden_layer,
        model = model, load_path = None, save_path = None,
        lr = lr, lr_decay = lr_decay, final_lr = final_lr,
        batch_size = batch_size, n_epoch = n_epoch, monitor_step = monitor_step)
    
    print ('Building')
    trainer.build()
    
    print ('Training')
    trainer.train()

