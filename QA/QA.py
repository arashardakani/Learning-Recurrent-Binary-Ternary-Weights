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
import time
import fuel.datasets, fuel.streams, fuel.transformers, fuel.schemes
import theano
import importlib
import data

from trainer import Trainer
from model import Network
from layer import eLSTM
#from reader import get_stream


if __name__ == "__main__":
    
    rng = np.random.RandomState(1234)
    batch_size = 128
    BN_LSTM = False
    BN_epsilon=1e-5
    BN_fast_eval= True
    dropout_input = 1.
    length = 240
    initial_gamma = 1e-1
    initial_beta = 0

    # Termination criteria
    n_epoch = 80
    monitor_step = 1
    
    # Learning Rate 
    lr = 0.003
    final_lr = 0.001
    lr_decay = (final_lr/lr)**(1./n_epoch)
    


    # binary/ternary weights for LSTM
    binary_training = True
    ternary_training = False
    stochastic_training = True


    model_name = 'deepmind_attentive_reader'
    config = importlib.import_module('.%s' % model_name, 'config')

    # Build datastream
    path = os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn/questions/training")
    valid_path = os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn/questions/validation")
    test_path = os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn/questions/test")
    vocab_path = os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn/stats/training/vocab.txt")
    
    ds, train_stream = data.setup_datastream(path, vocab_path, config)
    _, valid_stream = data.setup_datastream(valid_path, vocab_path, config)
    _, test_stream = data.setup_datastream(test_path, vocab_path, config)


    # layer's parameters
    n_inputs = ds.vocab_size
    n_units = 256
    n_classes = 550
    n_hidden_layer = 1
    
    print ('Preparing the dataset') 
    train_set = train_stream
    valid_set = valid_stream
    test_set  = test_stream

    
    print ('Creating the model')
    
    class create_model(Network):

        def __init__(self, rng):
            
            Network.__init__(self, n_hidden_layer = n_hidden_layer, BN_LSTM = BN_LSTM)


            print ("LSTM with embedding layer:")
            self.layer.append(eLSTM(rng = rng, n_inputs = n_inputs, n_units = n_units, initial_gamma = initial_gamma, 
                initial_beta = initial_beta, length = length - 1, batch_size = batch_size, BN = BN_LSTM, BN_epsilon=BN_epsilon,
                dropout=dropout_input, binary_training=binary_training, ternary_training = ternary_training, 
                stochastic_training = stochastic_training))
            
    
    model = create_model(rng = rng)
    
    print ('Creating the training functions')
    
    trainer = Trainer(rng = rng,
        train_set = train_set, valid_set = valid_set, test_set = test_set, length = length,
        model = model, load_path = None, save_path = None,
        lr = lr, lr_decay = lr_decay, final_lr = final_lr,
        batch_size = batch_size, n_epoch = n_epoch, monitor_step = monitor_step)
    
    print ('Building')
    trainer.build()

    print ('Training') 
    trainer.train()

