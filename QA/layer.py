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
import theano.printing as P 
from theano import pp
import time



class eLSTM(object):
    
    def __init__(self, rng, n_inputs, n_units, initial_gamma, initial_beta, length, batch_size,
        BN=False, BN_epsilon=1e-4, dropout=1.,
        binary_training=False, ternary_training=False, stochastic_training=False):
        
        self.rng = rng
        
        self.n_units = n_units
        print ("        n_units = "+str(n_units))
        self.n_inputs = n_inputs
        print ("        n_inputs = "+str(n_inputs))
        self.BN = BN
        print ("        BN = "+str(BN))
        self.BN_epsilon = BN_epsilon
        print ("        BN_epsilon = "+str(BN_epsilon))
        self.dropout = dropout
        print ("        dropout = "+str(dropout))
        
        self.binary_training = binary_training
        print ("        binary_training = "+str(binary_training))
        self.stochastic_training = stochastic_training
        print ("        stochastic_training = "+str(stochastic_training))     
        self.ternary_training = ternary_training
        print ("        ternary_training = "+str(ternary_training))
   
        
        self.high = np.float32(np.sqrt(6. / (n_inputs + n_units)))
        self.W0 = np.float32(self.high/2)
        
        self.initial_gamma = initial_gamma
        self.initial_beta = initial_beta
        self.activation = T.tanh
        self.length = length
        self.batch_size = batch_size
        self.size_we = 200
        self.n_entities = 550
        self.attention_mlp_hidden = 100

        def initializer(shape, interval_n, interval_p):
            return np.random.uniform(interval_n, interval_p, size=shape).astype(theano.config.floatX)




        self.high_a = np.float32(np.sqrt(6. / (self.n_units + 4 * self.n_units)))
        self.W0_a = np.float32(self.high_a/2)

        self.high_x = np.float32(np.sqrt(6. / (self.n_inputs + 4 * self.n_units)))
        self.W0_x = np.float32(self.high_x/2)    


        '''*****************************************************************'''
        self.bn_a_gamma_f_q = theano.shared(self.initial_gamma * np.ones((4 * self.n_units,)).astype(theano.config.floatX), name="bn_a_gamma_f_q")

        bn_a_beta_value_f_q = np.zeros((4 * self.n_units,))
        #bn_a_beta_value[self.n_units:2*self.n_units] = 1.
        bn_a_beta_value_f_q = bn_a_beta_value_f_q.astype(theano.config.floatX)
        self.bn_a_beta_f_q = theano.shared(bn_a_beta_value_f_q, name="bn_a_beta_f_q")


        self.bn_b_gamma_f_q = theano.shared(self.initial_gamma * np.ones((4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_gamma_f_q")
        self.bn_b_beta_f_q = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_beta_f_q")

        self.bn_c_gamma_f_q = theano.shared(self.initial_gamma * np.ones((self.n_units,)).astype(theano.config.floatX), name="bn_c_gamma_f_q")
        self.bn_c_beta_f_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="bn_c_beta_f_q")



        self.bn_a_mean_f_q = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_a_mean_f_q")
        self.bn_a_var_f_q = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_a_var_f_q")

        self.bn_b_mean_f_q = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_mean_f_q")
        self.bn_b_var_f_q = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_var_f_q")

        self.bn_c_mean_f_q = theano.shared(np.zeros((self.length,self.n_units,)).astype(theano.config.floatX), name="bn_c_mean_f_q")
        self.bn_c_var_f_q = theano.shared(np.zeros((self.length,self.n_units,)).astype(theano.config.floatX), name="bn_c_var_f_q")

        
        shape0 = (self.size_we, 4 * self.n_units)
        shape1 = (self.n_units , 4 * self.n_units)
        shape2 = (self.n_inputs , self.size_we)

        
        Wa_f_q = initializer(shape1, -self.high_a, self.high_a).astype(theano.config.floatX)
        We = np.random.uniform(-0.1, 0.1, size=shape2).astype(theano.config.floatX)

        self.h0_f_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="h0_f_q")
        self.c0_f_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="c0_f_q")
        self.Wa_f_q = theano.shared(Wa_f_q, name="Wa_f_q")
        self.We = theano.shared(We, name="We")
        self.Wx_f_q = theano.shared(initializer(shape0, -self.high_x, self.high_x).astype(theano.config.floatX), name="Wx_f_q")

        self.v_We = theano.shared(np.zeros(shape2).astype(theano.config.floatX), name='v_We_f_q')
        self.m_We = theano.shared(np.zeros(shape2).astype(theano.config.floatX), name='m_We_f_q')

        self.m_Wa_f_q = theano.shared(np.zeros(shape1).astype(theano.config.floatX), name='m_Wa_f_q')
        self.v_Wx_f_q = theano.shared(np.zeros(shape0).astype(theano.config.floatX), name='v_Wx_f_q')
        self.v_Wa_f_q = theano.shared(np.zeros(shape1).astype(theano.config.floatX), name='v_Wa_f_q')
        self.m_Wx_f_q = theano.shared(np.zeros(shape0).astype(theano.config.floatX), name='m_Wx_f_q')



        self.m_bn_a_beta_f_q = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='m_bn_a_beta_f_q')
        self.v_bn_c_beta_f_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='v_bn_c_beta_f_q')
        self.v_bn_a_beta_f_q = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='v_bn_a_beta_f_q')
        self.m_bn_c_beta_f_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='m_bn_c_beta_f_q')


        self.m_bn_a_gamma_f_q = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='m_bn_a_gamma_f_q')
        self.v_bn_a_gamma_f_q = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='v_bn_a_gamma_f_q')


        self.m_bn_b_gamma_f_q = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='m_bn_b_gamma_f_q')
        self.v_bn_b_gamma_f_q = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='v_bn_b_gamma_f_q')

        self.m_bn_c_gamma_f_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='m_bn_c_gamma_f_q')
        self.v_bn_c_gamma_f_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='v_bn_c_gamma_f_q')


        self.m_h0_f_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="m_h0_f_q")
        self.m_c0_f_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="m_c0_f_q")

        self.v_h0_f_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="v_h0_f_q")
        self.v_c0_f_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="v_c0_f_q")
        '''*****************************************************************'''
        self.bn_a_gamma_b_q = theano.shared(self.initial_gamma * np.ones((4 * self.n_units,)).astype(theano.config.floatX), name="bn_a_gamma_b_q")

        bn_a_beta_value = np.zeros((4 * self.n_units,))

        bn_a_beta_value = bn_a_beta_value.astype(theano.config.floatX)
        self.bn_a_beta_b_q = theano.shared(bn_a_beta_value, name="bn_a_beta_b_q")


        self.bn_b_gamma_b_q = theano.shared(self.initial_gamma * np.ones((4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_gamma_b_q")
        self.bn_b_beta_b_q = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_beta_b_q")

        self.bn_c_gamma_b_q = theano.shared(self.initial_gamma * np.ones((self.n_units,)).astype(theano.config.floatX), name="bn_c_gamma_b_q")
        self.bn_c_beta_b_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="bn_c_beta_b_q")



        self.bn_a_mean_b_q = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_a_mean_b_q")
        self.bn_a_var_b_q = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_a_var_b_q")

        self.bn_b_mean_b_q = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_mean_b_q")
        self.bn_b_var_b_q = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_var_b_q")

        self.bn_c_mean_b_q = theano.shared(np.zeros((self.length,self.n_units,)).astype(theano.config.floatX), name="bn_c_mean_b_q")
        self.bn_c_var_b_q = theano.shared(np.zeros((self.length,self.n_units,)).astype(theano.config.floatX), name="bn_c_var_b_q")

        
        Wa_b_q = initializer(shape1, -self.high_a, self.high_a).astype(theano.config.floatX)

        """if self.identity_hh:
            Wa[:self.n_units, :self.n_units] = np.eye(self.n_units)"""

        self.h0_b_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="h0_b_q")
        self.c0_b_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="c0_b_q")
        self.Wa_b_q = theano.shared(Wa_b_q, name="Wa_b_q")

        self.Wx_b_q = theano.shared(initializer(shape0, -self.high_x, self.high_x).astype(theano.config.floatX), name="Wx_b_q")




        self.m_Wa_b_q = theano.shared(np.zeros(shape1).astype(theano.config.floatX), name='m_Wa_b_q')
        self.v_Wx_b_q = theano.shared(np.zeros(shape0).astype(theano.config.floatX), name='v_Wx_b_q')
        self.v_Wa_b_q = theano.shared(np.zeros(shape1).astype(theano.config.floatX), name='v_Wa_b_q')
        self.m_Wx_b_q = theano.shared(np.zeros(shape0).astype(theano.config.floatX), name='m_Wx_b_q')



        self.m_bn_a_beta_b_q = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='m_bn_a_beta_b_q')
        self.v_bn_c_beta_b_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='v_bn_c_beta_b_q')
        self.v_bn_a_beta_b_q = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='v_bn_a_beta_b_q')
        self.m_bn_c_beta_b_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='m_bn_c_beta_b_q')


        self.m_bn_a_gamma_b_q = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='m_bn_a_gamma_b_q')
        self.v_bn_a_gamma_b_q = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='v_bn_a_gamma_b_q')


        self.m_bn_b_gamma_b_q = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='m_bn_b_gamma_b_q')
        self.v_bn_b_gamma_b_q = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='v_bn_b_gamma_b_q')

        self.m_bn_c_gamma_b_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='m_bn_c_gamma_b_q')
        self.v_bn_c_gamma_b_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='v_bn_c_gamma_b_q')


        self.m_h0_b_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="m_h0_b_q")
        self.m_c0_b_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="m_c0_b_q")

        self.v_h0_b_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="v_h0_b_q")
        self.v_c0_b_q = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="v_c0_b_q")

        '''*****************************************************************'''
        self.bn_a_gamma_f_c = theano.shared(self.initial_gamma * np.ones((4 * self.n_units,)).astype(theano.config.floatX), name="bn_a_gamma_f_c")

        bn_a_beta_value = np.zeros((4 * self.n_units,))
        #bn_a_beta_value[self.n_units:2*self.n_units] = 1.
        bn_a_beta_value = bn_a_beta_value.astype(theano.config.floatX)
        self.bn_a_beta_f_c = theano.shared(bn_a_beta_value, name="bn_a_beta_f_c")


        self.bn_b_gamma_f_c = theano.shared(self.initial_gamma * np.ones((4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_gamma_f_c")
        self.bn_b_beta_f_c = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_beta_f_c")

        self.bn_c_gamma_f_c = theano.shared(self.initial_gamma * np.ones((self.n_units,)).astype(theano.config.floatX), name="bn_c_gamma_f_c")
        self.bn_c_beta_f_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="bn_c_beta_f_c")



        self.bn_a_mean_f_c = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_a_mean_f_c")
        self.bn_a_var_f_c = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_a_var_f_c")

        self.bn_b_mean_f_c = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_mean_f_c")
        self.bn_b_var_f_c = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_var_f_c")

        self.bn_c_mean_f_c = theano.shared(np.zeros((self.length,self.n_units,)).astype(theano.config.floatX), name="bn_c_mean_f_c")
        self.bn_c_var_f_c = theano.shared(np.zeros((self.length,self.n_units,)).astype(theano.config.floatX), name="bn_c_var_f_c")

        

        
        Wa_f_c = initializer(shape1, -self.high_a, self.high_a).astype(theano.config.floatX)


        """if self.identity_hh:
            Wa[:self.n_units, :self.n_units] = np.eye(self.n_units)"""

        self.h0_f_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="h0_f_c")
        self.c0_f_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="c0_f_c")
        self.Wa_f_c = theano.shared(Wa_f_c, name="Wa_f_c")

        self.Wx_f_c = theano.shared(initializer(shape0, -self.high_x, self.high_x).astype(theano.config.floatX), name="Wx_f_c")


        self.m_Wa_f_c = theano.shared(np.zeros(shape1).astype(theano.config.floatX), name='m_Wa_f_c')
        self.v_Wx_f_c = theano.shared(np.zeros(shape0).astype(theano.config.floatX), name='v_Wx_f_c')
        self.v_Wa_f_c = theano.shared(np.zeros(shape1).astype(theano.config.floatX), name='v_Wa_f_c')
        self.m_Wx_f_c = theano.shared(np.zeros(shape0).astype(theano.config.floatX), name='m_Wx_f_c')



        self.m_bn_a_beta_f_c = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='m_bn_a_beta_f_c')
        self.v_bn_c_beta_f_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='v_bn_c_beta_f_c')
        self.v_bn_a_beta_f_c = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='v_bn_a_beta_f_c')
        self.m_bn_c_beta_f_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='m_bn_c_beta_f_c')


        self.m_bn_a_gamma_f_c = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='m_bn_a_gamma_f_c')
        self.v_bn_a_gamma_f_c = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='v_bn_a_gamma_f_c')


        self.m_bn_b_gamma_f_c = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='m_bn_b_gamma_f_c')
        self.v_bn_b_gamma_f_c = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='v_bn_b_gamma_f_c')

        self.m_bn_c_gamma_f_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='m_bn_c_gamma_f_c')
        self.v_bn_c_gamma_f_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='v_bn_c_gamma_f_c')


        self.m_h0_f_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="m_h0_f_c")
        self.m_c0_f_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="m_c0_f_c")

        self.v_h0_f_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="v_h0_f_c")
        self.v_c0_f_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="v_c0_f_c")


        '''*****************************************************************'''
        self.bn_a_gamma_b_c = theano.shared(self.initial_gamma * np.ones((4 * self.n_units,)).astype(theano.config.floatX), name="bn_a_gamma_b_c")

        bn_a_beta_value = np.zeros((4 * self.n_units,))
        #bn_a_beta_value[self.n_units:2*self.n_units] = 1.
        bn_a_beta_value = bn_a_beta_value.astype(theano.config.floatX)
        self.bn_a_beta_b_c = theano.shared(bn_a_beta_value, name="bn_a_beta_b_c")


        self.bn_b_gamma_b_c = theano.shared(self.initial_gamma * np.ones((4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_gamma_b_c")
        self.bn_b_beta_b_c = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_beta_b_c")

        self.bn_c_gamma_b_c = theano.shared(self.initial_gamma * np.ones((self.n_units,)).astype(theano.config.floatX), name="bn_c_gamma_b_c")
        self.bn_c_beta_b_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="bn_c_beta_b_c")



        self.bn_a_mean_b_c = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_a_mean_b_c")
        self.bn_a_var_b_c = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_a_var_b_c")

        self.bn_b_mean_b_c = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_mean_b_c")
        self.bn_b_var_b_c = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_var_b_c")

        self.bn_c_mean_b_c = theano.shared(np.zeros((self.length,self.n_units,)).astype(theano.config.floatX), name="bn_c_mean_b_c")
        self.bn_c_var_b_c = theano.shared(np.zeros((self.length,self.n_units,)).astype(theano.config.floatX), name="bn_c_var_b_c")


        
        Wa_b_c = initializer(shape1, -self.high_a, self.high_a).astype(theano.config.floatX)


        """if self.identity_hh:
            Wa[:self.n_units, :self.n_units] = np.eye(self.n_units)"""

        self.h0_b_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="h0_b_c")
        self.c0_b_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="c0_b_c")
        self.Wa_b_c = theano.shared(Wa_b_c, name="Wa_b_c")
        self.Wx_b_c = theano.shared(initializer(shape0, -self.high_x, self.high_x).astype(theano.config.floatX), name="Wx_b_c")





        self.m_Wa_b_c = theano.shared(np.zeros(shape1).astype(theano.config.floatX), name='m_Wa_b_c')
        self.v_Wx_b_c = theano.shared(np.zeros(shape0).astype(theano.config.floatX), name='v_Wx_b_c')
        self.v_Wa_b_c = theano.shared(np.zeros(shape1).astype(theano.config.floatX), name='v_Wa_b_c')
        self.m_Wx_b_c = theano.shared(np.zeros(shape0).astype(theano.config.floatX), name='m_Wx_b_c')



        self.m_bn_a_beta_b_c = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='m_bn_a_beta_b_c')
        self.v_bn_c_beta_b_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='v_bn_c_beta_b_c')
        self.v_bn_a_beta_b_c = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='v_bn_a_beta_b_c')
        self.m_bn_c_beta_b_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='m_bn_c_beta_b_c')


        self.m_bn_a_gamma_b_c = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='m_bn_a_gamma_b_c')
        self.v_bn_a_gamma_b_c = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='v_bn_a_gamma_b_c')


        self.m_bn_b_gamma_b_c = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='m_bn_b_gamma_b_c')
        self.v_bn_b_gamma_b_c = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='v_bn_b_gamma_b_c')

        self.m_bn_c_gamma_b_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='m_bn_c_gamma_b_c')
        self.v_bn_c_gamma_b_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='v_bn_c_gamma_b_c')


        self.m_h0_b_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="m_h0_b_c")
        self.m_c0_b_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="m_c0_b_c")

        self.v_h0_b_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="v_h0_b_c")
        self.v_c0_b_c = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="v_c0_b_c")

        '''*****************************************************************'''

        W_values_q = np.asarray(initializer((2*self.n_units, self.attention_mlp_hidden),0.,1.),dtype=theano.config.floatX)
        b_values_q = np.zeros((self.attention_mlp_hidden), dtype=theano.config.floatX)
        
        self.Wq = theano.shared(value=W_values_q, name='Wq')
        self.bq = theano.shared(value=b_values_q, name='bq')
        
        
        # momentum
        self.m_Wq = theano.shared(value=np.zeros((2*self.n_units, self.attention_mlp_hidden), dtype=theano.config.floatX), name='m_t_Wq')
        self.v_Wq = theano.shared(value=np.zeros((2*self.n_units, self.attention_mlp_hidden), dtype=theano.config.floatX), name='v_t_Wq')
        self.m_bq = theano.shared(value=b_values_q, name='m_t_bq')
        self.v_bq = theano.shared(value=b_values_q, name='v_t_bq')

        W_values_c = np.asarray(initializer((2*self.n_units, self.attention_mlp_hidden),0.,1.),dtype=theano.config.floatX)
        self.Wc = theano.shared(value=W_values_c, name='Wc')
        
        # momentum
        self.m_Wc = theano.shared(value=np.zeros((2*self.n_units, self.attention_mlp_hidden), dtype=theano.config.floatX), name='m_t_Wc')
        self.v_Wc = theano.shared(value=np.zeros((2*self.n_units, self.attention_mlp_hidden), dtype=theano.config.floatX), name='v_t_Wc')




        
        W_values = np.asarray(initializer((4*self.n_units,self.n_entities),0.,1.),dtype=theano.config.floatX)
        b_values = np.zeros((self.n_entities), dtype=theano.config.floatX)
        
        self.W = theano.shared(value=W_values, name='W')
        self.b = theano.shared(value=b_values, name='b')

        # momentum
        self.m_W = theano.shared(value=np.zeros((4*self.n_units,self.n_entities), dtype=theano.config.floatX), name='m_t_W')
        self.v_W = theano.shared(value=np.zeros((4*self.n_units,self.n_entities), dtype=theano.config.floatX), name='v_t_W')
        self.m_b = theano.shared(value=b_values, name='m_t_b')
        self.v_b = theano.shared(value=b_values, name='v_t_b')




        W_values_att = np.asarray(initializer((self.attention_mlp_hidden, 1),0.,1.),dtype=theano.config.floatX)
        b_values_att = np.zeros((1), dtype=theano.config.floatX)
        
        self.W_att = theano.shared(value=W_values_att, name='W_att')
        self.b_att = theano.shared(value=b_values_att, name='b_att')

        # momentum
        self.m_W_att = theano.shared(value=np.zeros((self.attention_mlp_hidden, 1), dtype=theano.config.floatX), name='m_t_W_att')
        self.v_W_att = theano.shared(value=np.zeros((self.attention_mlp_hidden, 1), dtype=theano.config.floatX), name='v_t_W_att')
        self.m_b_att = theano.shared(value=b_values_att, name='m_t_b_att')
        self.v_b_att = theano.shared(value=b_values_att, name='v_t_b_att')


        self.n_samples = theano.shared(value=np.float32(0),name='n_samples')    




    def hard_sigm(self,x):
        return T.clip((x+1)/2,0,1)


    def clipped_v(self, x):
        return T.clip(T.abs_(x), 0, 1)
    
    def binarize_weights_x(self,W,eval):
        if self.binary_training == True:
            if self.stochastic_training == True:
                p = self.hard_sigm(W / self.W0_x)
                srng = theano.sandbox.rng_mrg.MRG_RandomStreams(self.rng.randint(999998))
                p_mask =  T.cast(srng.binomial(n=1, p=p, size=T.shape(W)), theano.config.floatX)
                Wb = T.switch(p_mask,self.W0_x,-self.W0_x)
            else:
                Wb = T.switch(T.ge(W,0),self.W0_x,-self.W0_x)
        elif self.ternary_training == True:
            if self.stochastic_training == True:
                w_sign = T.gt(W,0) * 2 - 1
                p = self.clipped_v(W / self.W0_x)
                srng = theano.sandbox.rng_mrg.MRG_RandomStreams(self.rng.randint(999998))
                Wb = self.W0_x * w_sign * T.cast(srng.binomial(n=1, p=p, size=T.shape(W)), theano.config.floatX)
            else:
                larger_than_neg_0_5 = T.gt(W, -self.W0_x/3)
                larger_than_pos_0_5 = T.gt(W, self.W0_x/3)
                W_val = larger_than_neg_0_5 * 1 + larger_than_pos_0_5 * 1 - 1
                Wb = W_val * self.W0_x
        else:
            Wb = W

        return Wb


    def binarize_weights_a(self,W,eval):
        
        if self.binary_training == True:
            if self.stochastic_training == True:
                p = self.hard_sigm(W / self.W0_a)
                srng = theano.sandbox.rng_mrg.MRG_RandomStreams(self.rng.randint(999998))
                p_mask =  T.cast(srng.binomial(n=1, p=p, size=T.shape(W)), theano.config.floatX)
                Wb = T.switch(p_mask,self.W0_a,-self.W0_a)
            else:
                Wb = T.switch(T.ge(W,0),self.W0_a,-self.W0_a)
        elif self.ternary_training == True:
            if self.stochastic_training == True:
                w_sign = T.gt(W,0) * 2 - 1
                p = self.clipped_v(W / self.W0_a)
                srng = theano.sandbox.rng_mrg.MRG_RandomStreams(self.rng.randint(999998))

                Wb = self.W0_a * w_sign * T.cast(srng.binomial(n=1, p=p, size=T.shape(W)), theano.config.floatX)
            else:
                larger_than_neg_0_5 = T.gt(W, -self.W0_a/3)
                larger_than_pos_0_5 = T.gt(W, self.W0_a/3)
                W_val = larger_than_neg_0_5 * 1 + larger_than_pos_0_5 * 1 - 1
                Wb = W_val * self.W0_a
        else:
            Wb = W

        return Wb

    def BatchNormalization_a_f_q(self, x, t, can_fit = True):

        if can_fit == True:
            mean = T.mean(x,axis=0)
            var  = T.var (x,axis=0)
        else:
            mean = self.bn_a_mean_f_q[t]
            var = self.bn_a_var_f_q[t]



        if (self.BN == False):
            y = x + self.bn_a_beta_f_q
        else:
            y = theano.tensor.nnet.bn.batch_normalization(
                inputs=x,
                gamma=self.bn_a_gamma_f_q, beta = self.bn_a_beta_f_q,
                mean=T.shape_padleft(mean),
                std=T.shape_padleft(T.sqrt(var + self.BN_epsilon)))
        return y , mean, var

    def BatchNormalization_b_f_q(self, x, t, can_fit = True):

        
        if can_fit == True:
            mean = T.mean(x,axis=0)
            var  = T.var (x,axis=0)
        else:
            mean = self.bn_b_mean_f_q[t]
            var = self.bn_b_var_f_q[t]


        if (self.BN == False):
            y = x + self.bn_b_beta_f_q
        else:
            y = theano.tensor.nnet.bn.batch_normalization(
                inputs=x,
                gamma=self.bn_b_gamma_f_q, beta= self.bn_b_beta_f_q,
                mean=T.shape_padleft(mean),
                std=T.shape_padleft(T.sqrt(var + self.BN_epsilon)))
        return y, mean, var

    def BatchNormalization_c_f_q(self, x, t, can_fit = True):


        if can_fit == True:
            mean = T.mean(x,axis=0)
            var  = T.var (x,axis=0)
        else:
            mean = self.bn_c_mean_f_q[t]
            var = self.bn_c_var_f_q[t]
        


        if (self.BN == False):
            y = x + self.bn_c_beta_f_q
        else:
            y = theano.tensor.nnet.bn.batch_normalization(
                inputs=x,
                gamma=self.bn_c_gamma_f_q, beta=self.bn_c_beta_f_q,
                mean=T.shape_padleft(mean),
                std=T.shape_padleft(T.sqrt(var + self.BN_epsilon)))
        return y, mean, var



    def BatchNormalization_a_b_q(self, x, t, can_fit = True):

        if can_fit == True:
            mean = T.mean(x,axis=0)
            var  = T.var (x,axis=0)
        else:
            mean = self.bn_a_mean_b_q[t]
            var = self.bn_a_var_b_q[t]



        if (self.BN == False):
            y = x + self.bn_a_beta_b_q
        else:
            y = theano.tensor.nnet.bn.batch_normalization(
                inputs=x,
                gamma=self.bn_a_gamma_b_q, beta = self.bn_a_beta_b_q,
                mean=T.shape_padleft(mean),
                std=T.shape_padleft(T.sqrt(var + self.BN_epsilon)))
        return y , mean, var

    def BatchNormalization_b_b_q(self, x, t, can_fit = True):

        
        if can_fit == True:
            mean = T.mean(x,axis=0)
            var  = T.var (x,axis=0)
        else:
            mean = self.bn_b_mean_b_q[t]
            var = self.bn_b_var_b_q[t]


        if (self.BN == False):
            y = x + self.bn_b_beta_b_q
        else:
            y = theano.tensor.nnet.bn.batch_normalization(
                inputs=x,
                gamma=self.bn_b_gamma_b_q, beta= self.bn_b_beta_b_q,
                mean=T.shape_padleft(mean),
                std=T.shape_padleft(T.sqrt(var + self.BN_epsilon)))
        return y, mean, var

    def BatchNormalization_c_b_q(self, x, t, can_fit = True):


        if can_fit == True:
            mean = T.mean(x,axis=0)
            var  = T.var (x,axis=0)
        else:
            mean = self.bn_c_mean_b_q[t]
            var = self.bn_c_var_b_q[t]
        


        if (self.BN == False):
            y = x + self.bn_c_beta_b_q
        else:
            y = theano.tensor.nnet.bn.batch_normalization(
                inputs=x,
                gamma=self.bn_c_gamma_b_q, beta=self.bn_c_beta_b_q,
                mean=T.shape_padleft(mean),
                std=T.shape_padleft(T.sqrt(var + self.BN_epsilon)))
        return y, mean, var




    def BatchNormalization_a_f_c(self, x, t, can_fit = True):

        if can_fit == True:
            mean = T.mean(x,axis=0)
            var  = T.var (x,axis=0)
        else:
            mean = self.bn_a_mean_f_c[t]
            var = self.bn_a_var_f_c[t]



        if (self.BN == False):
            y = x + self.bn_a_beta_f_c
        else:
            y = theano.tensor.nnet.bn.batch_normalization(
                inputs=x,
                gamma=self.bn_a_gamma_f_c, beta = self.bn_a_beta_f_c,
                mean=T.shape_padleft(mean),
                std=T.shape_padleft(T.sqrt(var + self.BN_epsilon)))
        return y , mean, var

    def BatchNormalization_b_f_c(self, x, t, can_fit = True):

        
        if can_fit == True:
            mean = T.mean(x,axis=0)
            var  = T.var (x,axis=0)
        else:
            mean = self.bn_b_mean_f_c[t]
            var = self.bn_b_var_f_c[t]


        if (self.BN == False):
            y = x + self.bn_b_beta_f_c
        else:
            y = theano.tensor.nnet.bn.batch_normalization(
                inputs=x,
                gamma=self.bn_b_gamma_f_c, beta= self.bn_b_beta_f_c,
                mean=T.shape_padleft(mean),
                std=T.shape_padleft(T.sqrt(var + self.BN_epsilon)))
        return y, mean, var

    def BatchNormalization_c_f_c(self, x, t, can_fit = True):


        if can_fit == True:
            mean = T.mean(x,axis=0)
            var  = T.var (x,axis=0)
        else:
            mean = self.bn_c_mean_f_c[t]
            var = self.bn_c_var_f_c[t]
        


        if (self.BN == False):
            y = x + self.bn_c_beta_f_c
        else:
            y = theano.tensor.nnet.bn.batch_normalization(
                inputs=x,
                gamma=self.bn_c_gamma_f_c, beta=self.bn_c_beta_f_c,
                mean=T.shape_padleft(mean),
                std=T.shape_padleft(T.sqrt(var + self.BN_epsilon)))
        return y, mean, var



    def BatchNormalization_a_b_c(self, x, t, can_fit = True):

        if can_fit == True:
            mean = T.mean(x,axis=0)
            var  = T.var (x,axis=0)
        else:
            mean = self.bn_a_mean_b_c[t]
            var = self.bn_a_var_b_c[t]



        if (self.BN == False):
            y = x + self.bn_a_beta_b_c
        else:
            y = theano.tensor.nnet.bn.batch_normalization(
                inputs=x,
                gamma=self.bn_a_gamma_b_c, beta = self.bn_a_beta_b_c,
                mean=T.shape_padleft(mean),
                std=T.shape_padleft(T.sqrt(var + self.BN_epsilon)))
        return y , mean, var

    def BatchNormalization_b_b_c(self, x, t, can_fit = True):

        
        if can_fit == True:
            mean = T.mean(x,axis=0)
            var  = T.var (x,axis=0)
        else:
            mean = self.bn_b_mean_b_c[t]
            var = self.bn_b_var_b_c[t]


        if (self.BN == False):
            y = x + self.bn_b_beta_b_c
        else:
            y = theano.tensor.nnet.bn.batch_normalization(
                inputs=x,
                gamma=self.bn_b_gamma_b_c, beta= self.bn_b_beta_b_c,
                mean=T.shape_padleft(mean),
                std=T.shape_padleft(T.sqrt(var + self.BN_epsilon)))
        return y, mean, var

    def BatchNormalization_c_b_c(self, x, t, can_fit = True):


        if can_fit == True:
            mean = T.mean(x,axis=0)
            var  = T.var (x,axis=0)
        else:
            mean = self.bn_c_mean_b_c[t]
            var = self.bn_c_var_b_c[t]
        


        if (self.BN == False):
            y = x + self.bn_c_beta_b_c
        else:
            y = theano.tensor.nnet.bn.batch_normalization(
                inputs=x,
                gamma=self.bn_c_gamma_b_c, beta=self.bn_c_beta_b_c,
                mean=T.shape_padleft(mean),
                std=T.shape_padleft(T.sqrt(var + self.BN_epsilon)))
        return y, mean, var




    
    def fprop(self, que, que_m, con, con_m, can_fit, eval):

        symlength_q = que.shape[0]
        t_q = T.cast(T.arange(symlength_q), "int16")
        batch_size_q = que.shape[1]
        dummy_states_q = dict(h=T.zeros((symlength_q, batch_size_q, self.n_units)),
                            c=T.zeros((symlength_q, batch_size_q, self.n_units)))

        output_names_f_q = "h c".split()
        output_names_b_q = "h c".split()

        symlength_c = con.shape[0]
        t_c = T.cast(T.arange(symlength_c), "int16")
        batch_size_c = con.shape[1]
        dummy_states_c = dict(h=T.zeros((symlength_c, batch_size_c, self.n_units)),
                            c=T.zeros((symlength_c, batch_size_c, self.n_units)))

        output_names_f_c = "h c".split()
        output_names_b_c = "h c".split()



        # binarize the weights
        self.Wbx_f_q = self.binarize_weights_x(self.Wx_f_q,eval)
        self.Wba_f_q = self.binarize_weights_a(self.Wa_f_q,eval)

        self.Wbx_b_q = self.binarize_weights_x(self.Wx_b_q,eval)
        self.Wba_b_q = self.binarize_weights_a(self.Wa_b_q,eval)

        self.Wbx_f_c = self.binarize_weights_x(self.Wx_f_c,eval)
        self.Wba_f_c = self.binarize_weights_a(self.Wa_f_c,eval)

        self.Wbx_b_c = self.binarize_weights_x(self.Wx_b_c,eval)
        self.Wba_b_c = self.binarize_weights_a(self.Wa_b_c,eval)



        emb_que = self.We[T.cast(que,"int16"),:]
        emb_con = self.We[T.cast(con,"int16"),:]



        if (self.dropout < 1.):
            if eval == False:
                srng = theano.sandbox.rng_mrg.MRG_RandomStreams(self.rng.randint(999999))
                mask = T.cast(srng.binomial(n=1, p=self.dropout, size=T.shape(emb_que)), theano.config.floatX)
                emb_que = (1./self.dropout) * emb_que * mask
            else:
                emb_que = emb_que

        if (self.dropout < 1.):
            if eval == False:
                srng = theano.sandbox.rng_mrg.MRG_RandomStreams(self.rng.randint(999999))
                mask = T.cast(srng.binomial(n=1, p=self.dropout, size=T.shape(emb_con)), theano.config.floatX)
                emb_con = (1./self.dropout) * emb_con * mask
            else:
                emb_con = emb_con


        #x_m = c = T.repeat(x_m[:,:,None],self.n_units,axis = 1)
        #x_m = T.reshape(x_m,(symlength,batch_size,self.n_units))

        def stepfn_f_q(t, x, x_m, dummy_h, dummy_c, h, c, Wba, Wbx):

            if (self.dropout < 1.):
                if eval == False:
                    srng = theano.sandbox.rng_mrg.MRG_RandomStreams(self.rng.randint(999999))
                    mask = T.cast(srng.binomial(n=1, p=self.dropout, size=(batch_size , self.n_units) ), theano.config.floatX)
                    h_drop = h * mask * (1./self.dropout)
                else:
                    h_drop = h 
            else:
                h_drop = h

            btilde = T.dot(x, Wbx)
            atilde = T.dot(h_drop, Wba)

            a_normal, a_mean, a_var = self.BatchNormalization_a_f_q(atilde, t, can_fit)    
            b_normal, b_mean, b_var = self.BatchNormalization_b_f_q(btilde, t, can_fit)





            ab = a_normal + b_normal

            g, f, i, o = [fn(ab[:, j * self.n_units:(j + 1) * self.n_units])
                          for j, fn in enumerate([T.tanh] + 3 * [T.nnet.sigmoid])]

            c_state = c
            h_state = h
            c = dummy_c + f * c + i * g

            c_normal, c_mean, c_var = self.BatchNormalization_c_f_q(c, t, can_fit)

            h = dummy_h + o * T.tanh(c_normal)

            c = (x_m[:, None] * c + (1. - x_m[:, None]) * c_state)
            h = (x_m[:, None] * h + (1. - x_m[:, None]) * h_state)

            return (h, c)


        def stepfn_b_q(t, x, x_m, dummy_h, dummy_c, h, c, Wba, Wbx):

            if (self.dropout < 1.):
                if eval == False:
                    srng = theano.sandbox.rng_mrg.MRG_RandomStreams(self.rng.randint(999999))
                    mask = T.cast(srng.binomial(n=1, p=self.dropout, size=(batch_size , self.n_units) ), theano.config.floatX)
                    h_drop = h * mask * (1./self.dropout)
                else:
                    h_drop = h 
            else:
                h_drop = h

            btilde = T.dot(x, Wbx)
            atilde = T.dot(h_drop, Wba)

            a_normal, a_mean, a_var = self.BatchNormalization_a_b_q(atilde, t, can_fit)    
            b_normal, b_mean, b_var = self.BatchNormalization_b_b_q(btilde, t, can_fit)





            ab = a_normal + b_normal

            g, f, i, o = [fn(ab[:, j * self.n_units:(j + 1) * self.n_units])
                          for j, fn in enumerate([T.tanh] + 3 * [T.nnet.sigmoid])]

            c_state = c
            h_state = h
            c = dummy_c + f * c + i * g

            c_normal, c_mean, c_var = self.BatchNormalization_c_b_q(c, t, can_fit)

            h = dummy_h + o * T.tanh(c_normal)

            c = (x_m[:, None] * c + (1. - x_m[:, None]) * c_state)
            h = (x_m[:, None] * h + (1. - x_m[:, None]) * h_state)


            return (h, c)


        def stepfn_f_c(t, x, x_m, dummy_h, dummy_c, h, c, Wba, Wbx):

            if (self.dropout < 1.):
                if eval == False:
                    srng = theano.sandbox.rng_mrg.MRG_RandomStreams(self.rng.randint(999999))
                    mask = T.cast(srng.binomial(n=1, p=self.dropout, size=(batch_size , self.n_units) ), theano.config.floatX)
                    h_drop = h * mask * (1./self.dropout)
                else:
                    h_drop = h 
            else:
                h_drop = h

            btilde = T.dot(x, Wbx)
            atilde = T.dot(h_drop, Wba)

            a_normal, a_mean, a_var = self.BatchNormalization_a_f_c(atilde, t, can_fit)    
            b_normal, b_mean, b_var = self.BatchNormalization_b_f_c(btilde, t, can_fit)





            ab = a_normal + b_normal

            g, f, i, o = [fn(ab[:, j * self.n_units:(j + 1) * self.n_units])
                          for j, fn in enumerate([T.tanh] + 3 * [T.nnet.sigmoid])]

            c_state = c
            h_state = h
            c = dummy_c + f * c + i * g

            c_normal, c_mean, c_var = self.BatchNormalization_c_f_c(c, t, can_fit)

            h = dummy_h + o * T.tanh(c_normal)

            c = (x_m[:, None] * c + (1. - x_m[:, None]) * c_state)
            h = (x_m[:, None] * h + (1. - x_m[:, None]) * h_state)

            #c = (x_m * c + (1. - x_m) * c_state)
            #h = (x_m * h + (1. - x_m) * h_state)

            return (h, c)


        def stepfn_b_c(t, x, x_m, dummy_h, dummy_c, h, c, Wba, Wbx):

            if (self.dropout < 1.):
                if eval == False:
                    srng = theano.sandbox.rng_mrg.MRG_RandomStreams(self.rng.randint(999999))
                    mask = T.cast(srng.binomial(n=1, p=self.dropout, size=(batch_size , self.n_units) ), theano.config.floatX)
                    h_drop = h * mask * (1./self.dropout)
                else:
                    h_drop = h 
            else:
                h_drop = h

            btilde = T.dot(x, Wbx)
            atilde = T.dot(h_drop, Wba)

            a_normal, a_mean, a_var = self.BatchNormalization_a_b_c(atilde, t, can_fit)    
            b_normal, b_mean, b_var = self.BatchNormalization_b_b_c(btilde, t, can_fit)





            ab = a_normal + b_normal

            g, f, i, o = [fn(ab[:, j * self.n_units:(j + 1) * self.n_units])
                          for j, fn in enumerate([T.tanh] + 3 * [T.nnet.sigmoid])]

            c_state = c
            h_state = h
            c = dummy_c + f * c + i * g

            c_normal, c_mean, c_var = self.BatchNormalization_c_b_c(c, t, can_fit)

            h = dummy_h + o * T.tanh(c_normal)

            c = (x_m[:, None] * c + (1. - x_m[:, None]) * c_state)
            h = (x_m[:, None] * h + (1. - x_m[:, None]) * h_state)


            return (h, c)




        sequences_f_q = [t_q, emb_que, que_m, dummy_states_q["h"], dummy_states_q["c"]]
        non_sequences_f_q = [self.Wba_f_q, self.Wbx_f_q]
        outputs_info_f_q = [
            T.repeat(self.h0_f_q[None, :], batch_size_q, axis=0),
            T.repeat(self.c0_f_q[None, :], batch_size_q, axis=0),
        ]

        outputs_f_q, updates_f_q = theano.scan(
            stepfn_f_q,
            sequences=sequences_f_q,
            non_sequences=non_sequences_f_q,
            outputs_info=outputs_info_f_q)


        outputs_f_q = dict(zip(output_names_f_q, outputs_f_q))






        sequences_b_q = [t_q, emb_que[::-1], que_m[::-1], dummy_states_q["h"], dummy_states_q["c"]]
        non_sequences_b_q = [self.Wba_b_q, self.Wbx_b_q]
        outputs_info_b_q = [
            T.repeat(self.h0_b_q[None, :], batch_size_q, axis=0),
            T.repeat(self.c0_b_q[None, :], batch_size_q, axis=0),
        ]

        outputs_b_q, updates_b_q = theano.scan(
            stepfn_b_q,
            sequences=sequences_b_q,
            non_sequences=non_sequences_b_q,
            outputs_info=outputs_info_b_q)


        outputs_b_q = dict(zip(output_names_b_q, outputs_b_q))





        sequences_f_c = [t_c, emb_con, con_m, dummy_states_c["h"], dummy_states_c["c"]]
        non_sequences_f_c = [self.Wba_f_c, self.Wbx_f_c]
        outputs_info_f_c = [
            T.repeat(self.h0_f_c[None, :], batch_size_c, axis=0),
            T.repeat(self.c0_f_c[None, :], batch_size_c, axis=0),
        ]

        outputs_f_c, updates_f_c = theano.scan(
            stepfn_f_c,
            sequences=sequences_f_c,
            non_sequences=non_sequences_f_c,
            outputs_info=outputs_info_f_c)


        outputs_f_c = dict(zip(output_names_f_c, outputs_f_c))






        sequences_b_c = [t_c, emb_con[::-1], con_m[::-1], dummy_states_c["h"], dummy_states_c["c"]]
        non_sequences_b_c = [self.Wba_b_c, self.Wbx_b_c]
        outputs_info_b_c = [
            T.repeat(self.h0_b_c[None, :], batch_size_c, axis=0),
            T.repeat(self.c0_b_c[None, :], batch_size_c, axis=0),
        ]

        outputs_b_c, updates_b_c = theano.scan(
            stepfn_b_c,
            sequences=sequences_b_c,
            non_sequences=non_sequences_b_c,
            outputs_info=outputs_info_b_c)


        outputs_b_c = dict(zip(output_names_b_c, outputs_b_c))


        qenc = T.concatenate([outputs_f_q["h"][-1], outputs_b_q["h"][-1]], axis=1)
        cenc = T.concatenate([outputs_f_c["h"], outputs_b_c["h"]], axis=2)

        'applying dropout probability of 0.2'
        if eval == False:
            srng = theano.sandbox.rng_mrg.MRG_RandomStreams(self.rng.randint(999999))
            mask = T.cast(srng.binomial(n=1, p=self.dropout, size=T.shape(qenc)), theano.config.floatX)
            qenc = (1./0.8) * qenc * mask
        else:
            qenc = qenc

        if eval == False:
            srng = theano.sandbox.rng_mrg.MRG_RandomStreams(self.rng.randint(999999))
            mask = T.cast(srng.binomial(n=1, p=self.dropout, size=T.shape(cenc)), theano.config.floatX)
            cenc = (1./0.8) * cenc * mask
        else:
            cenc = cenc



        attention_que = T.dot(qenc , self.Wq) + self.bq
        attention_con = T.dot(cenc.reshape((cenc.shape[0]*cenc.shape[1], cenc.shape[2])) , self.Wc)
        attention_con = attention_con.reshape((cenc.shape[0],cenc.shape[1],self.attention_mlp_hidden))
        attention = T.tanh(attention_con + attention_que[None, :, :])

        att_weights = T.dot(attention.reshape((attention.shape[0]*attention.shape[1], attention.shape[2])) , self.W_att) + self.b_att
        att_weights = att_weights.reshape((attention.shape[0], attention.shape[1]))

        attended = T.sum(cenc * T.nnet.softmax(att_weights.T).T[:, :, None], axis=0)
        
        out = T.dot(T.concatenate([attended, qenc], axis=1) , self.W) + self.b
        
        
        return out
    

    def bprop(self, cost):

        if self.binary_training == True:
            self.dEdWa_f_q = T.grad(cost=cost, wrt = self.Wba_f_q)
            self.dEdWx_f_q = T.grad(cost=cost, wrt = self.Wbx_f_q)

            self.dEdWa_b_q = T.grad(cost=cost, wrt = self.Wba_b_q)
            self.dEdWx_b_q = T.grad(cost=cost, wrt = self.Wbx_b_q)

            self.dEdWa_f_c = T.grad(cost=cost, wrt = self.Wba_f_c)
            self.dEdWx_f_c = T.grad(cost=cost, wrt = self.Wbx_f_c)

            self.dEdWa_b_c = T.grad(cost=cost, wrt = self.Wba_b_c)
            self.dEdWx_b_c = T.grad(cost=cost, wrt = self.Wbx_b_c)
            
        else:
            self.dEdWa_f_q = T.grad(cost=cost, wrt = self.Wba_f_q)
            self.dEdWx_f_q = T.grad(cost=cost, wrt = self.Wbx_f_q) 

            self.dEdWa_b_q = T.grad(cost=cost, wrt = self.Wba_b_q)
            self.dEdWx_b_q = T.grad(cost=cost, wrt = self.Wbx_b_q) 

            self.dEdWa_f_c = T.grad(cost=cost, wrt = self.Wba_f_c)
            self.dEdWx_f_c = T.grad(cost=cost, wrt = self.Wbx_f_c) 

            self.dEdWa_b_c = T.grad(cost=cost, wrt = self.Wba_b_c)
            self.dEdWx_b_c = T.grad(cost=cost, wrt = self.Wbx_b_c) 



        if self.BN == True:
            self.dEdbn_a_gamma_f_q = T.grad(cost=cost, wrt=self.bn_a_gamma_f_q)
            self.dEdbn_a_beta_f_q = T.grad(cost=cost, wrt=self.bn_a_beta_f_q)
            self.dEdbn_b_gamma_f_q = T.grad(cost=cost, wrt=self.bn_b_gamma_f_q)
            self.dEdbn_c_gamma_f_q = T.grad(cost=cost, wrt=self.bn_c_gamma_f_q)
            self.dEdbn_c_beta_f_q = T.grad(cost=cost, wrt=self.bn_c_beta_f_q)

            self.dEdbn_a_gamma_b_q = T.grad(cost=cost, wrt=self.bn_a_gamma_b_q)
            self.dEdbn_a_beta_b_q = T.grad(cost=cost, wrt=self.bn_a_beta_b_q)
            self.dEdbn_b_gamma_b_q = T.grad(cost=cost, wrt=self.bn_b_gamma_b_q)
            self.dEdbn_c_gamma_b_q = T.grad(cost=cost, wrt=self.bn_c_gamma_b_q)
            self.dEdbn_c_beta_b_q = T.grad(cost=cost, wrt=self.bn_c_beta_b_q)

            self.dEdbn_a_gamma_f_c = T.grad(cost=cost, wrt=self.bn_a_gamma_f_c)
            self.dEdbn_a_beta_f_c = T.grad(cost=cost, wrt=self.bn_a_beta_f_c)
            self.dEdbn_b_gamma_f_c = T.grad(cost=cost, wrt=self.bn_b_gamma_f_c)
            self.dEdbn_c_gamma_f_c = T.grad(cost=cost, wrt=self.bn_c_gamma_f_c)
            self.dEdbn_c_beta_f_c = T.grad(cost=cost, wrt=self.bn_c_beta_f_c)

            self.dEdbn_a_gamma_b_c = T.grad(cost=cost, wrt=self.bn_a_gamma_b_c)
            self.dEdbn_a_beta_b_c = T.grad(cost=cost, wrt=self.bn_a_beta_b_c)
            self.dEdbn_b_gamma_b_c = T.grad(cost=cost, wrt=self.bn_b_gamma_b_c)
            self.dEdbn_c_gamma_b_c = T.grad(cost=cost, wrt=self.bn_c_gamma_b_c)
            self.dEdbn_c_beta_b_c = T.grad(cost=cost, wrt=self.bn_c_beta_b_c)
        else:
            self.dEdbn_a_beta_f_q = T.grad(cost=cost, wrt=self.bn_a_beta_f_q)
            self.dEdbn_c_beta_f_q = T.grad(cost=cost, wrt=self.bn_c_beta_f_q)

            self.dEdbn_a_beta_b_q = T.grad(cost=cost, wrt=self.bn_a_beta_b_q)
            self.dEdbn_c_beta_b_q = T.grad(cost=cost, wrt=self.bn_c_beta_b_q)

            self.dEdbn_a_beta_f_c = T.grad(cost=cost, wrt=self.bn_a_beta_f_c)
            self.dEdbn_c_beta_f_c = T.grad(cost=cost, wrt=self.bn_c_beta_f_c)

            self.dEdbn_a_beta_b_c = T.grad(cost=cost, wrt=self.bn_a_beta_b_c)
            self.dEdbn_c_beta_b_c = T.grad(cost=cost, wrt=self.bn_c_beta_b_c)

        self.dEdWe = T.grad(cost=cost, wrt = self.We) 

        self.dEdh0_f_q = T.grad(cost=cost, wrt=self.h0_f_q)
        self.dEdc0_f_q = T.grad(cost=cost, wrt=self.c0_f_q)

        self.dEdh0_b_q = T.grad(cost=cost, wrt=self.h0_b_q)
        self.dEdc0_b_q = T.grad(cost=cost, wrt=self.c0_b_q)

        self.dEdh0_f_c = T.grad(cost=cost, wrt=self.h0_f_c)
        self.dEdc0_f_c = T.grad(cost=cost, wrt=self.c0_f_c)

        self.dEdh0_b_c = T.grad(cost=cost, wrt=self.h0_b_c)
        self.dEdc0_b_c = T.grad(cost=cost, wrt=self.c0_b_c)

        self.dEdW = T.grad(cost=cost, wrt=self.W) 
        self.dEdb = T.grad(cost=cost, wrt=self.b)

        self.dEdWq = T.grad(cost=cost, wrt=self.Wq) 
        self.dEdbq = T.grad(cost=cost, wrt=self.bq)

        self.dEdWc = T.grad(cost=cost, wrt=self.Wc) 
        
        self.dEdW_att = T.grad(cost=cost, wrt=self.W_att) 
        self.dEdb_att = T.grad(cost=cost, wrt=self.b_att)

        
    def parameters_updates(self, LR):    
        
        updates = []
        

        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        alpha = 0.05

        t = self.n_samples + 1
        a_t = LR * T.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)


        #updates.append((self.Wba, self.Wba))

        m_t_Wa_f_q = beta1 * self.m_Wa_f_q + (1 - beta1) * self.dEdWa_f_q
        v_t_Wa_f_q = beta2 * self.v_Wa_f_q + (1 - beta2) * self.dEdWa_f_q ** 2
        step_Wa_f_q = a_t * m_t_Wa_f_q / (T.sqrt(v_t_Wa_f_q) + epsilon)

        if self.binary_training==True:
            step_Wa_f_q = T.clip(step_Wa_f_q, -self.W0_a, self.W0_a)

        updates.append((self.m_Wa_f_q, m_t_Wa_f_q))
        updates.append((self.v_Wa_f_q, v_t_Wa_f_q))
        updates.append((self.Wa_f_q, self.Wa_f_q - step_Wa_f_q))


        m_t_Wa_b_q = beta1 * self.m_Wa_b_q + (1 - beta1) * self.dEdWa_b_q
        v_t_Wa_b_q = beta2 * self.v_Wa_b_q + (1 - beta2) * self.dEdWa_b_q ** 2
        step_Wa_b_q = a_t * m_t_Wa_b_q / (T.sqrt(v_t_Wa_b_q) + epsilon)

        if self.binary_training==True:
            step_Wa_b_q = T.clip(step_Wa_b_q, -self.W0_a, self.W0_a)

        updates.append((self.m_Wa_b_q, m_t_Wa_b_q))
        updates.append((self.v_Wa_b_q, v_t_Wa_b_q))
        updates.append((self.Wa_b_q, self.Wa_b_q - step_Wa_b_q))


        m_t_Wa_f_c = beta1 * self.m_Wa_f_c + (1 - beta1) * self.dEdWa_f_c
        v_t_Wa_f_c = beta2 * self.v_Wa_f_c + (1 - beta2) * self.dEdWa_f_c ** 2
        step_Wa_f_c = a_t * m_t_Wa_f_c / (T.sqrt(v_t_Wa_f_c) + epsilon)

        if self.binary_training==True:
            step_Wa_f_c = T.clip(step_Wa_f_c, -self.W0_a, self.W0_a)

        updates.append((self.m_Wa_f_c, m_t_Wa_f_c))
        updates.append((self.v_Wa_f_c, v_t_Wa_f_c))
        updates.append((self.Wa_f_c, self.Wa_f_c - step_Wa_f_c))


        m_t_Wa_b_c = beta1 * self.m_Wa_b_c + (1 - beta1) * self.dEdWa_b_c
        v_t_Wa_b_c = beta2 * self.v_Wa_b_c + (1 - beta2) * self.dEdWa_b_c ** 2
        step_Wa_b_c = a_t * m_t_Wa_b_c / (T.sqrt(v_t_Wa_b_c) + epsilon)

        if self.binary_training==True:
            step_Wa_b_c = T.clip(step_Wa_b_c, -self.W0_a, self.W0_a)

        updates.append((self.m_Wa_b_c, m_t_Wa_b_c))
        updates.append((self.v_Wa_b_c, v_t_Wa_b_c))
        updates.append((self.Wa_b_c, self.Wa_b_c - step_Wa_b_c))






        m_t_Wx_f_q = beta1 * self.m_Wx_f_q + (1 - beta1) * self.dEdWx_f_q
        v_t_Wx_f_q = beta2 * self.v_Wx_f_q + (1 - beta2) * self.dEdWx_f_q ** 2
        step_Wx_f_q = a_t * m_t_Wx_f_q / (T.sqrt(v_t_Wx_f_q) + epsilon)

        if self.binary_training==True:
            step_Wx_f_q = T.clip(step_Wx_f_q, -self.W0_x, self.W0_x)

        updates.append((self.m_Wx_f_q, m_t_Wx_f_q))
        updates.append((self.v_Wx_f_q, v_t_Wx_f_q))
        updates.append((self.Wx_f_q, self.Wx_f_q - step_Wx_f_q))


        m_t_Wx_b_q = beta1 * self.m_Wx_b_q + (1 - beta1) * self.dEdWx_b_q
        v_t_Wx_b_q = beta2 * self.v_Wx_b_q + (1 - beta2) * self.dEdWx_b_q ** 2
        step_Wx_b_q = a_t * m_t_Wx_b_q / (T.sqrt(v_t_Wx_b_q) + epsilon)

        if self.binary_training==True:
            step_Wx = T.clip(step_Wx_b_q, -self.W0_x, self.W0_x)

        updates.append((self.m_Wx_b_q, m_t_Wx_b_q))
        updates.append((self.v_Wx_b_q, v_t_Wx_b_q))
        updates.append((self.Wx_b_q, self.Wx_b_q - step_Wx_b_q))


        m_t_Wx_f_c = beta1 * self.m_Wx_f_c + (1 - beta1) * self.dEdWx_f_c
        v_t_Wx_f_c = beta2 * self.v_Wx_f_c + (1 - beta2) * self.dEdWx_f_c ** 2
        step_Wx_f_c = a_t * m_t_Wx_f_c / (T.sqrt(v_t_Wx_f_c) + epsilon)

        if self.binary_training==True:
            step_Wx_f_c = T.clip(step_Wx_f_c, -self.W0_x, self.W0_x)

        updates.append((self.m_Wx_f_c, m_t_Wx_f_c))
        updates.append((self.v_Wx_f_c, v_t_Wx_f_c))
        updates.append((self.Wx_f_c, self.Wx_f_c - step_Wx_f_c))


        m_t_Wx_b_c = beta1 * self.m_Wx_b_c + (1 - beta1) * self.dEdWx_b_c
        v_t_Wx_b_c = beta2 * self.v_Wx_b_c + (1 - beta2) * self.dEdWx_b_c ** 2
        step_Wx_b_c = a_t * m_t_Wx_b_c / (T.sqrt(v_t_Wx_b_c) + epsilon)

        if self.binary_training==True:
            step_Wx_b_c = T.clip(step_Wx_b_c, -self.W0_x, self.W0_x)

        updates.append((self.m_Wx_b_c, m_t_Wx_b_c))
        updates.append((self.v_Wx_b_c, v_t_Wx_b_c))
        updates.append((self.Wx_b_c, self.Wx_b_c - step_Wx_b_c))







        m_t_We = beta1 * self.m_We + (1 - beta1) * self.dEdWe
        v_t_We = beta2 * self.v_We + (1 - beta2) * self.dEdWe ** 2

        step_We = a_t * m_t_We / (T.sqrt(v_t_We) + epsilon)


        updates.append((self.v_We, v_t_We))
        #step_We = T.clip(step_We, -1, 1)
        updates.append((self.We, self.We - step_We))

        
        

        if self.BN == True:
            m_t_bn_a_beta_f_q = beta1 * self.m_bn_a_beta_f_q + (1 - beta1) * self.dEdbn_a_beta_f_q
            v_t_bn_a_beta_f_q = beta2 * self.v_bn_a_beta_f_q + (1 - beta2) * self.dEdbn_a_beta_f_q ** 2
            step_bn_a_beta_f_q = a_t * m_t_bn_a_beta_f_q / (T.sqrt(v_t_bn_a_beta_f_q) + epsilon)
            updates.append((self.m_bn_a_beta_f_q, m_t_bn_a_beta_f_q))
            updates.append((self.v_bn_a_beta_f_q, v_t_bn_a_beta_f_q))
            updates.append((self.bn_a_beta_f_q, self.bn_a_beta_f_q - step_bn_a_beta_f_q))

            m_t_bn_a_gamma_f_q = beta1 * self.m_bn_a_gamma_f_q + (1 - beta1) * self.dEdbn_a_gamma_f_q
            v_t_bn_a_gamma_f_q = beta2 * self.v_bn_a_gamma_f_q + (1 - beta2) * self.dEdbn_a_gamma_f_q ** 2
            step_bn_a_gamma_f_q = a_t * m_t_bn_a_gamma_f_q / (T.sqrt(v_t_bn_a_gamma_f_q) + epsilon)
            updates.append((self.m_bn_a_gamma_f_q, m_t_bn_a_gamma_f_q))
            updates.append((self.v_bn_a_gamma_f_q, v_t_bn_a_gamma_f_q))
            updates.append((self.bn_a_gamma_f_q, self.bn_a_gamma_f_q - step_bn_a_gamma_f_q))

            m_t_bn_b_gamma_f_q = beta1 * self.m_bn_b_gamma_f_q + (1 - beta1) * self.dEdbn_b_gamma_f_q
            v_t_bn_b_gamma_f_q = beta2 * self.v_bn_b_gamma_f_q + (1 - beta2) * self.dEdbn_b_gamma_f_q ** 2
            step_bn_b_gamma_f_q = a_t * m_t_bn_b_gamma_f_q / (T.sqrt(v_t_bn_b_gamma_f_q) + epsilon)
            updates.append((self.m_bn_b_gamma_f_q, m_t_bn_b_gamma_f_q))
            updates.append((self.v_bn_b_gamma_f_q, v_t_bn_b_gamma_f_q))
            updates.append((self.bn_b_gamma_f_q, self.bn_b_gamma_f_q - step_bn_b_gamma_f_q))

            m_t_bn_c_beta_f_q = beta1 * self.m_bn_c_beta_f_q + (1 - beta1) * self.dEdbn_c_beta_f_q
            v_t_bn_c_beta_f_q = beta2 * self.v_bn_c_beta_f_q + (1 - beta2) * self.dEdbn_c_beta_f_q ** 2
            step_bn_c_beta_f_q = a_t * m_t_bn_c_beta_f_q / (T.sqrt(v_t_bn_c_beta_f_q) + epsilon)
            updates.append((self.m_bn_c_beta_f_q, m_t_bn_c_beta_f_q))
            updates.append((self.v_bn_c_beta_f_q, v_t_bn_c_beta_f_q))
            updates.append((self.bn_c_beta_f_q, self.bn_c_beta_f_q - step_bn_c_beta_f_q))

            m_t_bn_c_gamma_f_q = beta1 * self.m_bn_c_gamma_f_q + (1 - beta1) * self.dEdbn_c_gamma_f_q
            v_t_bn_c_gamma_f_q = beta2 * self.v_bn_c_gamma_f_q + (1 - beta2) * self.dEdbn_c_gamma_f_q ** 2
            step_bn_c_gamma_f_q = a_t * m_t_bn_c_gamma_f_q / (T.sqrt(v_t_bn_c_gamma_f_q) + epsilon)
            updates.append((self.m_bn_c_gamma_f_q, m_t_bn_c_gamma_f_q))
            updates.append((self.v_bn_c_gamma_f_q, v_t_bn_c_gamma_f_q))
            updates.append((self.bn_c_gamma_f_q, self.bn_c_gamma_f_q - step_bn_c_gamma_f_q))





            m_t_bn_a_beta_b_q = beta1 * self.m_bn_a_beta_b_q + (1 - beta1) * self.dEdbn_a_beta_b_q
            v_t_bn_a_beta_b_q = beta2 * self.v_bn_a_beta_b_q + (1 - beta2) * self.dEdbn_a_beta_b_q ** 2
            step_bn_a_beta_b_q = a_t * m_t_bn_a_beta_b_q / (T.sqrt(v_t_bn_a_beta_b_q) + epsilon)
            updates.append((self.m_bn_a_beta_b_q, m_t_bn_a_beta_b_q))
            updates.append((self.v_bn_a_beta_b_q, v_t_bn_a_beta_b_q))
            updates.append((self.bn_a_beta_b_q, self.bn_a_beta_b_q - step_bn_a_beta_b_q))

            m_t_bn_a_gamma_b_q = beta1 * self.m_bn_a_gamma_b_q + (1 - beta1) * self.dEdbn_a_gamma_b_q
            v_t_bn_a_gamma_b_q = beta2 * self.v_bn_a_gamma_b_q + (1 - beta2) * self.dEdbn_a_gamma_b_q ** 2
            step_bn_a_gamma_b_q = a_t * m_t_bn_a_gamma_b_q / (T.sqrt(v_t_bn_a_gamma_b_q) + epsilon)
            updates.append((self.m_bn_a_gamma_b_q, m_t_bn_a_gamma_b_q))
            updates.append((self.v_bn_a_gamma_b_q, v_t_bn_a_gamma_b_q))
            updates.append((self.bn_a_gamma_b_q, self.bn_a_gamma_b_q - step_bn_a_gamma_b_q))

            m_t_bn_b_gamma_b_q = beta1 * self.m_bn_b_gamma_b_q + (1 - beta1) * self.dEdbn_b_gamma_b_q
            v_t_bn_b_gamma_b_q = beta2 * self.v_bn_b_gamma_b_q + (1 - beta2) * self.dEdbn_b_gamma_b_q ** 2
            step_bn_b_gamma_b_q = a_t * m_t_bn_b_gamma_b_q / (T.sqrt(v_t_bn_b_gamma_b_q) + epsilon)
            updates.append((self.m_bn_b_gamma_b_q, m_t_bn_b_gamma_b_q))
            updates.append((self.v_bn_b_gamma_b_q, v_t_bn_b_gamma_b_q))
            updates.append((self.bn_b_gamma_b_q, self.bn_b_gamma_b_q - step_bn_b_gamma_b_q))

            m_t_bn_c_beta_b_q = beta1 * self.m_bn_c_beta_b_q + (1 - beta1) * self.dEdbn_c_beta_b_q
            v_t_bn_c_beta_b_q = beta2 * self.v_bn_c_beta_b_q + (1 - beta2) * self.dEdbn_c_beta_b_q ** 2
            step_bn_c_beta_b_q = a_t * m_t_bn_c_beta_b_q / (T.sqrt(v_t_bn_c_beta_b_q) + epsilon)
            updates.append((self.m_bn_c_beta_b_q, m_t_bn_c_beta_b_q))
            updates.append((self.v_bn_c_beta_b_q, v_t_bn_c_beta_b_q))
            updates.append((self.bn_c_beta_b_q, self.bn_c_beta_b_q - step_bn_c_beta_b_q))

            m_t_bn_c_gamma_b_q = beta1 * self.m_bn_c_gamma_b_q + (1 - beta1) * self.dEdbn_c_gamma_b_q
            v_t_bn_c_gamma_b_q = beta2 * self.v_bn_c_gamma_b_q + (1 - beta2) * self.dEdbn_c_gamma_b_q ** 2
            step_bn_c_gamma_b_q = a_t * m_t_bn_c_gamma_b_q / (T.sqrt(v_t_bn_c_gamma_b_q) + epsilon)
            updates.append((self.m_bn_c_gamma_b_q, m_t_bn_c_gamma_b_q))
            updates.append((self.v_bn_c_gamma_b_q, v_t_bn_c_gamma_b_q))
            updates.append((self.bn_c_gamma_b_q, self.bn_c_gamma_b_q - step_bn_c_gamma_b_q))









            m_t_bn_a_beta_f_c = beta1 * self.m_bn_a_beta_f_c + (1 - beta1) * self.dEdbn_a_beta_f_c
            v_t_bn_a_beta_f_c = beta2 * self.v_bn_a_beta_f_c + (1 - beta2) * self.dEdbn_a_beta_f_c ** 2
            step_bn_a_beta_f_c = a_t * m_t_bn_a_beta_f_c / (T.sqrt(v_t_bn_a_beta_f_c) + epsilon)
            updates.append((self.m_bn_a_beta_f_c, m_t_bn_a_beta_f_c))
            updates.append((self.v_bn_a_beta_f_c, v_t_bn_a_beta_f_c))
            updates.append((self.bn_a_beta_f_c, self.bn_a_beta_f_c - step_bn_a_beta_f_c))

            m_t_bn_a_gamma_f_c = beta1 * self.m_bn_a_gamma_f_c + (1 - beta1) * self.dEdbn_a_gamma_f_c
            v_t_bn_a_gamma_f_c = beta2 * self.v_bn_a_gamma_f_c + (1 - beta2) * self.dEdbn_a_gamma_f_c ** 2
            step_bn_a_gamma_f_c = a_t * m_t_bn_a_gamma_f_c / (T.sqrt(v_t_bn_a_gamma_f_c) + epsilon)
            updates.append((self.m_bn_a_gamma_f_c, m_t_bn_a_gamma_f_c))
            updates.append((self.v_bn_a_gamma_f_c, v_t_bn_a_gamma_f_c))
            updates.append((self.bn_a_gamma_f_c, self.bn_a_gamma_f_c - step_bn_a_gamma_f_c))

            m_t_bn_b_gamma_f_c = beta1 * self.m_bn_b_gamma_f_c + (1 - beta1) * self.dEdbn_b_gamma_f_c
            v_t_bn_b_gamma_f_c = beta2 * self.v_bn_b_gamma_f_c + (1 - beta2) * self.dEdbn_b_gamma_f_c ** 2
            step_bn_b_gamma_f_c = a_t * m_t_bn_b_gamma_f_c / (T.sqrt(v_t_bn_b_gamma_f_c) + epsilon)
            updates.append((self.m_bn_b_gamma_f_c, m_t_bn_b_gamma_f_c))
            updates.append((self.v_bn_b_gamma_f_c, v_t_bn_b_gamma_f_c))
            updates.append((self.bn_b_gamma_f_c, self.bn_b_gamma_f_c - step_bn_b_gamma_f_c))

            m_t_bn_c_beta_f_c = beta1 * self.m_bn_c_beta_f_c + (1 - beta1) * self.dEdbn_c_beta_f_c
            v_t_bn_c_beta_f_c = beta2 * self.v_bn_c_beta_f_c + (1 - beta2) * self.dEdbn_c_beta_f_c ** 2
            step_bn_c_beta_f_c = a_t * m_t_bn_c_beta_f_c / (T.sqrt(v_t_bn_c_beta_f_c) + epsilon)
            updates.append((self.m_bn_c_beta_f_c, m_t_bn_c_beta_f_c))
            updates.append((self.v_bn_c_beta_f_c, v_t_bn_c_beta_f_c))
            updates.append((self.bn_c_beta_f_c, self.bn_c_beta_f_c - step_bn_c_beta_f_c))

            m_t_bn_c_gamma_f_c = beta1 * self.m_bn_c_gamma_f_c + (1 - beta1) * self.dEdbn_c_gamma_f_c
            v_t_bn_c_gamma_f_c = beta2 * self.v_bn_c_gamma_f_c + (1 - beta2) * self.dEdbn_c_gamma_f_c ** 2
            step_bn_c_gamma_f_c = a_t * m_t_bn_c_gamma_f_c / (T.sqrt(v_t_bn_c_gamma_f_c) + epsilon)
            updates.append((self.m_bn_c_gamma_f_c, m_t_bn_c_gamma_f_c))
            updates.append((self.v_bn_c_gamma_f_c, v_t_bn_c_gamma_f_c))
            updates.append((self.bn_c_gamma_f_c, self.bn_c_gamma_f_c - step_bn_c_gamma_f_c))













            m_t_bn_a_beta_b_c = beta1 * self.m_bn_a_beta_b_c + (1 - beta1) * self.dEdbn_a_beta_b_c
            v_t_bn_a_beta_b_c = beta2 * self.v_bn_a_beta_b_c + (1 - beta2) * self.dEdbn_a_beta_b_c ** 2
            step_bn_a_beta_b_c = a_t * m_t_bn_a_beta_b_c / (T.sqrt(v_t_bn_a_beta_b_c) + epsilon)
            updates.append((self.m_bn_a_beta_b_c, m_t_bn_a_beta_b_c))
            updates.append((self.v_bn_a_beta_b_c, v_t_bn_a_beta_b_c))
            updates.append((self.bn_a_beta_b_c, self.bn_a_beta_b_c - step_bn_a_beta_b_c))

            m_t_bn_a_gamma_b_c = beta1 * self.m_bn_a_gamma_b_c + (1 - beta1) * self.dEdbn_a_gamma_b_c
            v_t_bn_a_gamma_b_c = beta2 * self.v_bn_a_gamma_b_c + (1 - beta2) * self.dEdbn_a_gamma_b_c ** 2
            step_bn_a_gamma_b_c = a_t * m_t_bn_a_gamma_b_c / (T.sqrt(v_t_bn_a_gamma_b_c) + epsilon)
            updates.append((self.m_bn_a_gamma_b_c, m_t_bn_a_gamma_b_c))
            updates.append((self.v_bn_a_gamma_b_c, v_t_bn_a_gamma_b_c))
            updates.append((self.bn_a_gamma_b_c, self.bn_a_gamma_b_c - step_bn_a_gamma_b_c))

            m_t_bn_b_gamma_b_c = beta1 * self.m_bn_b_gamma_b_c + (1 - beta1) * self.dEdbn_b_gamma_b_c
            v_t_bn_b_gamma_b_c = beta2 * self.v_bn_b_gamma_b_c + (1 - beta2) * self.dEdbn_b_gamma_b_c ** 2
            step_bn_b_gamma_b_c = a_t * m_t_bn_b_gamma_b_c / (T.sqrt(v_t_bn_b_gamma_b_c) + epsilon)
            updates.append((self.m_bn_b_gamma_b_c, m_t_bn_b_gamma_b_c))
            updates.append((self.v_bn_b_gamma_b_c, v_t_bn_b_gamma_b_c))
            updates.append((self.bn_b_gamma_b_c, self.bn_b_gamma_b_c - step_bn_b_gamma_b_c))

            m_t_bn_c_beta_b_c = beta1 * self.m_bn_c_beta_b_c + (1 - beta1) * self.dEdbn_c_beta_b_c
            v_t_bn_c_beta_b_c = beta2 * self.v_bn_c_beta_b_c + (1 - beta2) * self.dEdbn_c_beta_b_c ** 2
            step_bn_c_beta_b_c = a_t * m_t_bn_c_beta_b_c / (T.sqrt(v_t_bn_c_beta_b_c) + epsilon)
            updates.append((self.m_bn_c_beta_b_c, m_t_bn_c_beta_b_c))
            updates.append((self.v_bn_c_beta_b_c, v_t_bn_c_beta_b_c))
            updates.append((self.bn_c_beta_b_c, self.bn_c_beta_b_c - step_bn_c_beta_b_c))

            m_t_bn_c_gamma_b_c = beta1 * self.m_bn_c_gamma_b_c + (1 - beta1) * self.dEdbn_c_gamma_b_c
            v_t_bn_c_gamma_b_c = beta2 * self.v_bn_c_gamma_b_c + (1 - beta2) * self.dEdbn_c_gamma_b_c ** 2
            step_bn_c_gamma_b_c = a_t * m_t_bn_c_gamma_b_c / (T.sqrt(v_t_bn_c_gamma_b_c) + epsilon)
            updates.append((self.m_bn_c_gamma_b_c, m_t_bn_c_gamma_b_c))
            updates.append((self.v_bn_c_gamma_b_c, v_t_bn_c_gamma_b_c))
            updates.append((self.bn_c_gamma_b_c, self.bn_c_gamma_b_c - step_bn_c_gamma_b_c))


        else:
            m_t_bn_a_beta_f_q = beta1 * self.m_bn_a_beta_f_q + (1 - beta1) * self.dEdbn_a_beta_f_q
            v_t_bn_a_beta_f_q = beta2 * self.v_bn_a_beta_f_q + (1 - beta2) * self.dEdbn_a_beta_f_q ** 2
            step_bn_a_beta_f_q = a_t * m_t_bn_a_beta_f_q / (T.sqrt(v_t_bn_a_beta_f_q) + epsilon)
            updates.append((self.m_bn_a_beta_f_q, m_t_bn_a_beta_f_q))
            updates.append((self.v_bn_a_beta_f_q, v_t_bn_a_beta_f_q))
            updates.append((self.bn_a_beta_f_q, self.bn_a_beta_f_q - step_bn_a_beta_f_q))

            m_t_bn_c_beta_f_q = beta1 * self.m_bn_c_beta_f_q + (1 - beta1) * self.dEdbn_c_beta_f_q
            v_t_bn_c_beta_f_q = beta2 * self.v_bn_c_beta_f_q + (1 - beta2) * self.dEdbn_c_beta_f_q ** 2
            step_bn_c_beta_f_q = a_t * m_t_bn_c_beta_f_q / (T.sqrt(v_t_bn_c_beta_f_q) + epsilon)
            updates.append((self.m_bn_c_beta_f_q, m_t_bn_c_beta_f_q))
            updates.append((self.v_bn_c_beta_f_q, v_t_bn_c_beta_f_q))
            updates.append((self.bn_c_beta_f_q, self.bn_c_beta_f_q - step_bn_c_beta_f_q))




            m_t_bn_a_beta_b_q = beta1 * self.m_bn_a_beta_b_q + (1 - beta1) * self.dEdbn_a_beta_b_q
            v_t_bn_a_beta_b_q = beta2 * self.v_bn_a_beta_b_q + (1 - beta2) * self.dEdbn_a_beta_b_q ** 2
            step_bn_a_beta_b_q = a_t * m_t_bn_a_beta_b_q / (T.sqrt(v_t_bn_a_beta_b_q) + epsilon)
            updates.append((self.m_bn_a_beta_b_q, m_t_bn_a_beta_b_q))
            updates.append((self.v_bn_a_beta_b_q, v_t_bn_a_beta_b_q))
            updates.append((self.bn_a_beta_b_q, self.bn_a_beta_b_q - step_bn_a_beta_b_q))

            m_t_bn_c_beta_b_q = beta1 * self.m_bn_c_beta_b_q + (1 - beta1) * self.dEdbn_c_beta_b_q
            v_t_bn_c_beta_b_q = beta2 * self.v_bn_c_beta_b_q + (1 - beta2) * self.dEdbn_c_beta_b_q ** 2
            step_bn_c_beta_b_q = a_t * m_t_bn_c_beta_b_q / (T.sqrt(v_t_bn_c_beta_b_q) + epsilon)
            updates.append((self.m_bn_c_beta_b_q, m_t_bn_c_beta_b_q))
            updates.append((self.v_bn_c_beta_b_q, v_t_bn_c_beta_b_q))
            updates.append((self.bn_c_beta_b_q, self.bn_c_beta_b_q - step_bn_c_beta_b_q))





            m_t_bn_a_beta_f_c = beta1 * self.m_bn_a_beta_f_c + (1 - beta1) * self.dEdbn_a_beta_f_c
            v_t_bn_a_beta_f_c = beta2 * self.v_bn_a_beta_f_c + (1 - beta2) * self.dEdbn_a_beta_f_c ** 2
            step_bn_a_beta_f_c = a_t * m_t_bn_a_beta_f_c / (T.sqrt(v_t_bn_a_beta_f_c) + epsilon)
            updates.append((self.m_bn_a_beta_f_c, m_t_bn_a_beta_f_c))
            updates.append((self.v_bn_a_beta_f_c, v_t_bn_a_beta_f_c))
            updates.append((self.bn_a_beta_f_c, self.bn_a_beta_f_c - step_bn_a_beta_f_c))

            m_t_bn_c_beta_f_c = beta1 * self.m_bn_c_beta_f_c + (1 - beta1) * self.dEdbn_c_beta_f_c
            v_t_bn_c_beta_f_c = beta2 * self.v_bn_c_beta_f_c + (1 - beta2) * self.dEdbn_c_beta_f_c ** 2
            step_bn_c_beta_f_c = a_t * m_t_bn_c_beta_f_c / (T.sqrt(v_t_bn_c_beta_f_c) + epsilon)
            updates.append((self.m_bn_c_beta_f_c, m_t_bn_c_beta_f_c))
            updates.append((self.v_bn_c_beta_f_c, v_t_bn_c_beta_f_c))
            updates.append((self.bn_c_beta_f_c, self.bn_c_beta_f_c - step_bn_c_beta_f_c))





            m_t_bn_a_beta_b_c = beta1 * self.m_bn_a_beta_b_c + (1 - beta1) * self.dEdbn_a_beta_b_c
            v_t_bn_a_beta_b_c = beta2 * self.v_bn_a_beta_b_c + (1 - beta2) * self.dEdbn_a_beta_b_c ** 2
            step_bn_a_beta_b_c = a_t * m_t_bn_a_beta_b_c / (T.sqrt(v_t_bn_a_beta_b_c) + epsilon)
            updates.append((self.m_bn_a_beta_b_c, m_t_bn_a_beta_b_c))
            updates.append((self.v_bn_a_beta_b_c, v_t_bn_a_beta_b_c))
            updates.append((self.bn_a_beta_b_c, self.bn_a_beta_b_c - step_bn_a_beta_b_c))

            m_t_bn_c_beta_b_c = beta1 * self.m_bn_c_beta_b_c + (1 - beta1) * self.dEdbn_c_beta_b_c
            v_t_bn_c_beta_b_c = beta2 * self.v_bn_c_beta_b_c + (1 - beta2) * self.dEdbn_c_beta_b_c ** 2
            step_bn_c_beta_b_c = a_t * m_t_bn_c_beta_b_c / (T.sqrt(v_t_bn_c_beta_b_c) + epsilon)
            updates.append((self.m_bn_c_beta_b_c, m_t_bn_c_beta_b_c))
            updates.append((self.v_bn_c_beta_b_c, v_t_bn_c_beta_b_c))
            updates.append((self.bn_c_beta_b_c, self.bn_c_beta_b_c - step_bn_c_beta_b_c))




        m_t_h0_f_q = beta1 * self.m_h0_f_q + (1 - beta1) * self.dEdh0_f_q
        v_t_h0_f_q = beta2 * self.v_h0_f_q + (1 - beta2) * self.dEdh0_f_q ** 2
        step_h0_f_q = a_t * m_t_h0_f_q / (T.sqrt(v_t_h0_f_q) + epsilon)
        updates.append((self.m_h0_f_q, m_t_h0_f_q))
        updates.append((self.v_h0_f_q, v_t_h0_f_q))
        updates.append((self.h0_f_q, self.h0_f_q - step_h0_f_q))


        m_t_c0_f_q = beta1 * self.m_c0_f_q + (1 - beta1) * self.dEdc0_f_q
        v_t_c0_f_q = beta2 * self.v_c0_f_q + (1 - beta2) * self.dEdc0_f_q ** 2
        step_c0_f_q = a_t * m_t_c0_f_q / (T.sqrt(v_t_c0_f_q) + epsilon)
        updates.append((self.m_c0_f_q, m_t_c0_f_q))
        updates.append((self.v_c0_f_q, v_t_c0_f_q))
        updates.append((self.c0_f_q, self.c0_f_q - step_c0_f_q))






        m_t_h0_b_q = beta1 * self.m_h0_b_q + (1 - beta1) * self.dEdh0_b_q
        v_t_h0_b_q = beta2 * self.v_h0_b_q + (1 - beta2) * self.dEdh0_b_q ** 2
        step_h0_b_q = a_t * m_t_h0_b_q / (T.sqrt(v_t_h0_b_q) + epsilon)
        updates.append((self.m_h0_b_q, m_t_h0_b_q))
        updates.append((self.v_h0_b_q, v_t_h0_b_q))
        updates.append((self.h0_b_q, self.h0_b_q - step_h0_b_q))


        m_t_c0_b_q = beta1 * self.m_c0_b_q + (1 - beta1) * self.dEdc0_b_q
        v_t_c0_b_q = beta2 * self.v_c0_b_q + (1 - beta2) * self.dEdc0_b_q ** 2
        step_c0_b_q = a_t * m_t_c0_b_q / (T.sqrt(v_t_c0_b_q) + epsilon)
        updates.append((self.m_c0_b_q, m_t_c0_b_q))
        updates.append((self.v_c0_b_q, v_t_c0_b_q))
        updates.append((self.c0_b_q, self.c0_b_q - step_c0_b_q))







        m_t_h0_f_c = beta1 * self.m_h0_f_c + (1 - beta1) * self.dEdh0_f_c
        v_t_h0_f_c = beta2 * self.v_h0_f_c + (1 - beta2) * self.dEdh0_f_c ** 2
        step_h0_f_c = a_t * m_t_h0_f_c / (T.sqrt(v_t_h0_f_c) + epsilon)
        updates.append((self.m_h0_f_c, m_t_h0_f_c))
        updates.append((self.v_h0_f_c, v_t_h0_f_c))
        updates.append((self.h0_f_c, self.h0_f_c - step_h0_f_c))


        m_t_c0_f_c = beta1 * self.m_c0_f_c + (1 - beta1) * self.dEdc0_f_c
        v_t_c0_f_c = beta2 * self.v_c0_f_c + (1 - beta2) * self.dEdc0_f_c ** 2
        step_c0_f_c = a_t * m_t_c0_f_c / (T.sqrt(v_t_c0_f_c) + epsilon)
        updates.append((self.m_c0_f_c, m_t_c0_f_c))
        updates.append((self.v_c0_f_c, v_t_c0_f_c))
        updates.append((self.c0_f_c, self.c0_f_c - step_c0_f_c))







        m_t_h0_b_c = beta1 * self.m_h0_b_c + (1 - beta1) * self.dEdh0_b_c
        v_t_h0_b_c = beta2 * self.v_h0_b_c + (1 - beta2) * self.dEdh0_b_c ** 2
        step_h0_b_c = a_t * m_t_h0_b_c / (T.sqrt(v_t_h0_b_c) + epsilon)
        updates.append((self.m_h0_b_c, m_t_h0_b_c))
        updates.append((self.v_h0_b_c, v_t_h0_b_c))
        updates.append((self.h0_b_c, self.h0_b_c - step_h0_b_c))


        m_t_c0_b_c = beta1 * self.m_c0_b_c + (1 - beta1) * self.dEdc0_b_c
        v_t_c0_b_c = beta2 * self.v_c0_b_c + (1 - beta2) * self.dEdc0_b_c ** 2
        step_c0_b_c = a_t * m_t_c0_b_c / (T.sqrt(v_t_c0_b_c) + epsilon)
        updates.append((self.m_c0_b_c, m_t_c0_b_c))
        updates.append((self.v_c0_b_c, v_t_c0_b_c))
        updates.append((self.c0_b_c, self.c0_b_c - step_c0_b_c))






        m_t_W = beta1 * self.m_W + (1 - beta1) * self.dEdW
        v_t_W = beta2 * self.v_W + (1 - beta2) * self.dEdW ** 2
        step_W = a_t * m_t_W / (T.sqrt(v_t_W) + epsilon)
        updates.append((self.m_W, m_t_W))
        updates.append((self.v_W, v_t_W))
        updates.append((self.W, self.W - step_W))


        m_t_b = beta1 * self.m_b + (1 - beta1) * self.dEdb
        v_t_b = beta2 * self.v_b + (1 - beta2) * self.dEdb ** 2
        step_b = a_t * m_t_b / (T.sqrt(v_t_b) + epsilon)
        updates.append((self.m_b, m_t_b))
        updates.append((self.v_b, v_t_b))
        updates.append((self.b, self.b - step_b))


        m_t_Wq = beta1 * self.m_Wq + (1 - beta1) * self.dEdWq
        v_t_Wq = beta2 * self.v_Wq + (1 - beta2) * self.dEdWq ** 2
        step_Wq = a_t * m_t_Wq / (T.sqrt(v_t_Wq) + epsilon)
        updates.append((self.m_Wq, m_t_Wq))
        updates.append((self.v_Wq, v_t_Wq))
        updates.append((self.Wq, self.Wq - step_Wq))


        m_t_bq = beta1 * self.m_bq + (1 - beta1) * self.dEdbq
        v_t_bq = beta2 * self.v_bq + (1 - beta2) * self.dEdbq ** 2
        step_bq = a_t * m_t_bq / (T.sqrt(v_t_bq) + epsilon)
        updates.append((self.m_bq, m_t_bq))
        updates.append((self.v_bq, v_t_bq))
        updates.append((self.bq, self.bq - step_bq))



        m_t_Wc = beta1 * self.m_Wc + (1 - beta1) * self.dEdWc
        v_t_Wc = beta2 * self.v_Wc + (1 - beta2) * self.dEdWc ** 2
        step_Wc = a_t * m_t_Wc / (T.sqrt(v_t_Wc) + epsilon)
        updates.append((self.m_Wc, m_t_Wc))
        updates.append((self.v_Wc, v_t_Wc))
        updates.append((self.Wc, self.Wc - step_Wc))






        m_t_W_att = beta1 * self.m_W_att + (1 - beta1) * self.dEdW_att
        v_t_W_att = beta2 * self.v_W_att + (1 - beta2) * self.dEdW_att ** 2
        step_W_att = a_t * m_t_W_att / (T.sqrt(v_t_W_att) + epsilon)
        updates.append((self.m_W_att, m_t_W_att))
        updates.append((self.v_W_att, v_t_W_att))
        updates.append((self.W_att, self.W_att - step_W_att))


        m_t_b_att = beta1 * self.m_b_att + (1 - beta1) * self.dEdb_att
        v_t_b_att = beta2 * self.v_b_att + (1 - beta2) * self.dEdb_att ** 2
        step_b_att = a_t * m_t_b_att / (T.sqrt(v_t_b_att) + epsilon)
        updates.append((self.m_b_att, m_t_b_att))
        updates.append((self.v_b_att, v_t_b_att))
        updates.append((self.b_att, self.b_att - step_b_att))




        updates.append((self.n_samples, t)) 

        return updates
    
        
    def BN_reset(self):
    
        updates = []
        updates.append((self.n_samples, self.n_samples*0.))
        
        return updates



