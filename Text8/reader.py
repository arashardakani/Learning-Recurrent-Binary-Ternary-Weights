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




class Text8(fuel.datasets.Dataset):
    provides_sources = ('features',)
    example_iteration_scheme = None

    def __init__(self, which_set, length, augment=False):
        self.which_set = which_set
        self.length = length
        
        self.augment = augment
        data = np.load("CHAR_LEVEL_TEXT8_NPZ")
        self.data = data[which_set]
        self.vocab = 27 #data["vocab"]
        self.num_examples = int(len(self.data) / self.length)
        
        if self.augment:
            self.num_examples -= 1
        super(Text8, self).__init__()

    def open(self):
        data = self.data
        if self.augment:
            offset = np.random.randint(self.length)
            data = data[offset:]
        data = (data[:self.num_examples * self.length]
                .reshape((self.num_examples, self.length)))
        return data

    def get_data(self, state, request):
        one_hot_batch = np.eye((self.vocab), dtype=theano.config.floatX)[state[request]]
        return (one_hot_batch,)

def get_stream(which_set, batch_size, length, num_examples=None, augment=False):
    dataset = Text8(which_set, length=length, augment=augment)
    if num_examples is None or num_examples > dataset.num_examples:
        num_examples = dataset.num_examples
    stream = fuel.streams.DataStream.default_stream(
        dataset,
        iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size))
    return stream


