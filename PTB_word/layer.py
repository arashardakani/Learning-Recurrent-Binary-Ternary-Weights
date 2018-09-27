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
import theano.printing as P 
from theano import pp
import time




class LSTM(object):
    
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
        self.size_we = self.n_units


        def initializer(shape, interval_n, interval_p):
            flat_shape = (shape[0], np.prod(shape[1:]))
            a = np.random.normal(0., 1., flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v
            q = q.reshape(shape)
            return np.random.uniform(interval_n, interval_p, size=shape).astype(theano.config.floatX)




        self.high_a = np.float32(np.sqrt(6. / (self.n_units + 4 * self.n_units)))
        self.W0_a = np.float32(self.high_a/2)

        self.high_x = np.float32(np.sqrt(6. / (self.size_we + 4 * self.n_units)))
        self.W0_x = np.float32(self.high_x/2)    



        self.bn_a_gamma = theano.shared(self.initial_gamma * np.ones((4 * self.n_units,)).astype(theano.config.floatX), name="bn_a_gamma")

        bn_a_beta_value = np.zeros((4 * self.n_units,))
        bn_a_beta_value = bn_a_beta_value.astype(theano.config.floatX)
        self.bn_a_beta = theano.shared(bn_a_beta_value, name="bn_a_beta")


        self.bn_b_gamma = theano.shared(self.initial_gamma * np.ones((4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_gamma")
        self.bn_b_beta = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_beta")

        self.bn_c_gamma = theano.shared(self.initial_gamma * np.ones((self.n_units,)).astype(theano.config.floatX), name="bn_c_gamma")
        self.bn_c_beta = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="bn_c_beta")



        self.bn_a_mean = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_a_mean")
        self.bn_a_var = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_a_var")

        self.bn_b_mean = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_mean")
        self.bn_b_var = theano.shared(np.zeros((self.length,4 * self.n_units,)).astype(theano.config.floatX), name="bn_b_var")

        self.bn_c_mean = theano.shared(np.zeros((self.length,self.n_units,)).astype(theano.config.floatX), name="bn_c_mean")
        self.bn_c_var = theano.shared(np.zeros((self.length,self.n_units,)).astype(theano.config.floatX), name="bn_c_var")

        
        shape0 = (self.size_we, 4 * self.n_units)
        shape1 = (self.n_units , 4 * self.n_units)
        shape2 = (self.n_inputs , self.size_we)
        
        Wa = initializer(shape1, -0.1, 0.1).astype(theano.config.floatX)
        We = np.random.uniform(-0.1, 0.1, size=shape2).astype(theano.config.floatX)

        """if self.identity_hh:
            Wa[:self.n_units, :self.n_units] = np.eye(self.n_units)"""

        self.h0 = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="h0")
        self.c0 = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="c0")
        self.Wa = theano.shared(Wa, name="Wa")
        self.We = theano.shared(We, name="We")
        self.Wx = theano.shared(initializer(shape0, -0.1, 0.1).astype(theano.config.floatX), name="Wx")


        # momentum
        self.m_Wa = theano.shared(np.zeros(shape1).astype(theano.config.floatX), name='m_Wa')
        self.v_Wx = theano.shared(np.zeros(shape0).astype(theano.config.floatX), name='v_Wx')
        self.v_We = theano.shared(np.zeros(shape2).astype(theano.config.floatX), name='v_We')
        self.v_Wa = theano.shared(np.zeros(shape1).astype(theano.config.floatX), name='v_Wa')
        self.m_Wx = theano.shared(np.zeros(shape0).astype(theano.config.floatX), name='m_Wx')
        self.m_We = theano.shared(np.zeros(shape2).astype(theano.config.floatX), name='m_We')

        self.m_bn_a_beta = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='m_bn_a_beta')
        self.m_bn_b_beta = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='m_bn_b_beta')
        self.v_bn_c_beta = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='v_bn_c_beta')
        self.v_bn_a_beta = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='v_bn_a_beta')
        self.v_bn_b_beta = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='v_bn_b_beta')
        self.m_bn_c_beta = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='m_bn_c_beta')


        self.m_bn_a_gamma = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='m_bn_a_gamma')
        self.v_bn_a_gamma = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='v_bn_a_gamma')

        self.m_bn_b_gamma = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='m_bn_b_gamma')
        self.v_bn_b_gamma = theano.shared(np.zeros((4 * self.n_units,)).astype(theano.config.floatX), name='v_bn_b_gamma')

        self.m_bn_c_gamma = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='m_bn_c_gamma')
        self.v_bn_c_gamma = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name='v_bn_c_gamma')


        self.m_h0 = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="m_h0")
        self.m_c0 = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="m_c0")

        self.v_h0 = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="v_h0")
        self.v_c0 = theano.shared(np.zeros((self.n_units,)).astype(theano.config.floatX), name="v_c0")



        self.n_samples = theano.shared(value=np.float32(0),name='n_samples')    

        self.dummy = theano.shared(np.zeros((179, 100, self.n_units)).astype(theano.config.floatX), name='dummy')



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

    def BatchNormalization_a(self, x, t, can_fit = True):

        if can_fit == True:
            mean = T.mean(x,axis=0)
            var  = T.var (x,axis=0)
        else:
            mean = self.bn_a_mean[t]
            var = self.bn_a_var[t]



        if (self.BN == False):
            y = x + self.bn_a_beta
        else:
            y = theano.tensor.nnet.bn.batch_normalization(
                inputs=x,
                gamma=self.bn_a_gamma, beta = self.bn_a_beta,
                mean=T.shape_padleft(mean),
                std=T.shape_padleft(T.sqrt(var + self.BN_epsilon)))
        return y , mean, var

    def BatchNormalization_b(self, x, t, can_fit = True):

        
        if can_fit == True:
            mean = T.mean(x,axis=0)
            var  = T.var (x,axis=0)
        else:
            mean = self.bn_b_mean[t]
            var = self.bn_b_var[t]


        if (self.BN == False):
            y = x + self.bn_b_beta
        else:
            y = theano.tensor.nnet.bn.batch_normalization(
                inputs=x,
                gamma=self.bn_b_gamma, beta=self.bn_b_beta,
                mean=T.shape_padleft(mean),
                std=T.shape_padleft(T.sqrt(var + self.BN_epsilon)))
        return y, mean, var

    def BatchNormalization_c(self, x, t, can_fit = True):


        if can_fit == True:
            mean = T.mean(x,axis=0)
            var  = T.var (x,axis=0)
        else:
            mean = self.bn_c_mean[t]
            var = self.bn_c_var[t]
        


        if (self.BN == False):
            y = x + self.bn_c_beta
        else:
            y = theano.tensor.nnet.bn.batch_normalization(
                inputs=x,
                gamma=self.bn_c_gamma, beta=self.bn_c_beta,
                mean=T.shape_padleft(mean),
                std=T.shape_padleft(T.sqrt(var + self.BN_epsilon)))
        return y, mean, var






    def fprop1(self, x, h0, c0, can_fit, eval):
        

        self.x = x
        symlength = x.shape[0]
        t = T.cast(T.arange(symlength), "int16")

        batch_size = x.shape[1]
        dummy_states = dict(h=T.zeros((symlength, batch_size, self.n_units)),
                            c=T.zeros((symlength, batch_size, self.n_units)))

        output_names = "h c atilde btilde".split()
        for key in "abc":
            for stat in "mean var".split():
                output_names.append("%s_%s" % (key, stat))

        # binarize the weights
        self.Wbx = self.binarize_weights_x(self.Wx,eval)
        self.Wba = self.binarize_weights_a(self.Wa,eval)


        a = self.We[T.cast(x,"int16"),:] 
        if (self.dropout < 1.):
            if eval == False:
                srng = theano.sandbox.rng_mrg.MRG_RandomStreams(self.rng.randint(999999))
                mask = T.cast(srng.binomial(n=1, p=self.dropout, size=T.shape(a) ), theano.config.floatX)
                a = (1./self.dropout) * mask * a
            else:
                a = a 



        def stepfn(t, x, dummy_h, dummy_c, h, c, Wba, Wbx):
            


            btilde = T.dot(x, Wbx)
            atilde = T.dot(h, Wba)


            a_normal, a_mean, a_var = self.BatchNormalization_a(atilde, t, can_fit)    
            b_normal, b_mean, b_var = self.BatchNormalization_b(btilde, t, can_fit)





            ab = a_normal + b_normal
            i, f, g, o = [fn(ab[:, j * self.n_units:(j + 1) * self.n_units])
                          for j, fn in enumerate(2 * [T.nnet.sigmoid] + [T.tanh] + [T.nnet.sigmoid])]

            c = dummy_c + f * c + i * g

            c_normal, c_mean, c_var = self.BatchNormalization_c(c, t, can_fit)

            h = dummy_h + o * T.tanh(c_normal)


            return (h, c, atilde, btilde, a_mean, a_var, b_mean, b_var, c_mean, c_var)




        sequences = [t, a, dummy_states["h"], dummy_states["c"]] 
        non_sequences = [self.Wba, self.Wbx]
        outputs_info = [
            h0,
            c0,
        ]
        outputs_info.extend([None] * (len(output_names) - len(outputs_info)))


        outputs, updates = theano.scan(
            stepfn,
            sequences=sequences,
            non_sequences=non_sequences,
            outputs_info=outputs_info)


        outputs = dict(zip(output_names, outputs))
        

        
        return outputs["h"], outputs["h"][-1], outputs["c"][-1]
    

    def bprop(self, cost):
        clip_ = 20000.
        if self.binary_training == True:
            self.dEdWa = T.clip(T.grad(cost=cost, wrt = self.Wba), -clip_, clip_) 
            norm = theano.tensor._tensor_py_operators.norm (self.dEdWa,2)**2
            self.dEdWx = T.clip(T.grad(cost=cost, wrt = self.Wbx), -clip_, clip_) 
            norm = norm + theano.tensor._tensor_py_operators.norm (self.dEdWx,2)**2
        else:
            self.dEdWa = T.clip(T.grad(cost=cost, wrt = self.Wa), -clip_, clip_) 
            norm = theano.tensor._tensor_py_operators.norm (self.dEdWa,2)**2
            self.dEdWx = T.clip(T.grad(cost=cost, wrt = self.Wx), -clip_, clip_) 
            norm = norm + theano.tensor._tensor_py_operators.norm (self.dEdWx,2)**2



        if self.BN == True:
            self.dEdbn_a_gamma = T.clip(T.grad(cost=cost, wrt = self.bn_a_gamma), -clip_, clip_) 
            norm = norm + theano.tensor._tensor_py_operators.norm (self.dEdbn_a_gamma,2)**2
            self.dEdbn_a_beta = T.clip(T.grad(cost=cost, wrt = self.bn_a_beta), -clip_, clip_) 
            norm = norm +  theano.tensor._tensor_py_operators.norm (self.dEdbn_a_beta,2)**2
            self.dEdbn_b_gamma = T.clip(T.grad(cost=cost, wrt = self.bn_b_gamma), -clip_, clip_) 
            norm = norm + theano.tensor._tensor_py_operators.norm (self.dEdbn_b_gamma,2)**2
            self.dEdbn_c_gamma = T.clip(T.grad(cost=cost, wrt = self.bn_c_gamma), -clip_, clip_) 
            norm = norm + theano.tensor._tensor_py_operators.norm (self.dEdbn_c_gamma,2)**2
            self.dEdbn_c_beta = T.clip(T.grad(cost=cost, wrt = self.bn_c_beta), -clip_, clip_) 
            norm = norm + theano.tensor._tensor_py_operators.norm (self.dEdbn_c_beta,2) **2
        else:
            self.dEdbn_a_beta = 0.5 * T.clip(T.grad(cost=cost, wrt = self.bn_a_beta), -clip_, clip_) 
            norm = norm + theano.tensor._tensor_py_operators.norm (self.dEdbn_a_beta,2)**2

            self.dEdbn_b_beta = 0.5 * T.clip(T.grad(cost=cost, wrt = self.bn_b_beta), -clip_, clip_ ) 
            norm = norm + theano.tensor._tensor_py_operators.norm (self.dEdbn_b_beta,2)**2


        self.dEdWe = T.clip(T.grad(cost=cost, wrt = self.We), -clip_, clip_) 
        norm = norm + theano.tensor._tensor_py_operators.norm (self.dEdWe,2)**2


        return norm
        
    def parameters_updates(self, LR, norm_factor):    
        
        updates = []
        

        beta1 = 0.9
        beta2 = 0.
        epsilon = 1e-6
        alpha = 0.05
        clip_ = 20000.
        t = self.n_samples + 1





        v_t_Wa = beta2 * self.v_Wa - LR * self.dEdWa * norm_factor
        step_Wa = self.Wa + v_t_Wa 


        updates.append((self.v_Wa, v_t_Wa))
        step_Wa = T.clip(step_Wa, -clip_, clip_)
        updates.append((self.Wa, step_Wa))





        v_t_Wx = beta2 * self.v_Wx - LR * self.dEdWx * norm_factor
        step_Wx = self.Wx + v_t_Wx

        updates.append((self.v_Wx, v_t_Wx))
        step_Wx = T.clip(step_Wx, -clip_, clip_)
        updates.append((self.Wx, step_Wx))





        v_t_We = beta2 * self.v_We - LR * self.dEdWe * norm_factor
        step_We = self.We + v_t_We


        updates.append((self.v_We, v_t_We))
        step_We = T.clip(step_We, -clip_, clip_)
        updates.append((self.We,  step_We))



        
        

        if self.BN == True:


            v_t_bn_a_beta = beta2 * self.v_bn_a_beta - LR * self.dEdbn_a_beta * norm_factor
            step_bn_a_beta = self.bn_a_beta + v_t_bn_a_beta
            updates.append((self.v_bn_a_beta, v_t_bn_a_beta))
            step_bn_a_beta = T.clip(step_bn_a_beta, -clip_, clip_)
            updates.append((self.bn_a_beta, step_bn_a_beta))




            v_t_bn_a_gamma = beta2 * self.v_bn_a_gamma - LR * self.dEdbn_a_gamma * norm_factor
            step_bn_a_gamma = self.bn_a_gamma + v_t_bn_a_gamma
            updates.append((self.v_bn_a_gamma, v_t_bn_a_gamma))
            step_bn_a_gamma = T.clip(step_bn_a_gamma, -clip_, clip_)
            updates.append((self.bn_a_gamma, step_bn_a_gamma))




            v_t_bn_b_gamma = beta2 * self.v_bn_b_gamma - LR * self.dEdbn_b_gamma * norm_factor
            step_bn_b_gamma = self.bn_b_gamma + v_t_bn_b_gamma
            updates.append((self.v_bn_b_gamma, v_t_bn_b_gamma))
            step_bn_b_gamma = T.clip(step_bn_b_gamma, -clip_, clip_)
            updates.append((self.bn_b_gamma, step_bn_b_gamma))




			
			
            v_t_bn_c_beta = beta2 * self.v_bn_c_beta -LR * self.dEdbn_c_beta  * norm_factor
            step_bn_c_beta = self.bn_c_beta  + v_t_bn_c_beta
            updates.append((self.v_bn_c_beta, v_t_bn_c_beta))
            step_bn_c_beta = T.clip(step_bn_c_beta, -clip_, clip_)
            updates.append((self.bn_c_beta, step_bn_c_beta))




            v_t_bn_c_gamma = beta2 * self.v_bn_c_gamma - LR *  self.dEdbn_c_gamma * norm_factor
            step_bn_c_gamma = self.bn_c_gamma + v_t_bn_c_gamma
            updates.append((self.v_bn_c_gamma, v_t_bn_c_gamma))
            step_bn_c_gamma = T.clip(step_bn_c_gamma, -clip_, clip_)
            updates.append((self.bn_c_gamma, step_bn_c_gamma))



        else:



            v_t_bn_a_beta = beta2 * self.v_bn_a_beta - LR * self.dEdbn_a_beta * norm_factor
            step_bn_a_beta = self.bn_a_beta + v_t_bn_a_beta
            updates.append((self.v_bn_a_beta, v_t_bn_a_beta))
            step_bn_a_beta = T.clip(step_bn_a_beta, -clip_, clip_)
            updates.append((self.bn_a_beta, step_bn_a_beta))


            v_t_bn_b_beta = beta2 * self.v_bn_b_beta - LR * self.dEdbn_b_beta * norm_factor
            step_bn_b_beta = self.bn_b_beta + v_t_bn_b_beta
            updates.append((self.v_bn_b_beta, v_t_bn_b_beta))
            step_bn_b_beta = T.clip(step_bn_b_beta, -clip_, clip_)
            updates.append((self.bn_b_beta, step_bn_b_beta))




        updates.append((self.n_samples, t)) 

        return updates
    
        
    def BN_reset(self):
    
        updates = []
        
        updates.append((self.m_Wa, self.m_Wa*0.))
        updates.append((self.v_Wa, self.v_Wa*0.))
        updates.append((self.m_We, self.m_We*0.))
        updates.append((self.v_We, self.v_We*0.))
		
        updates.append((self.m_Wx, self.m_Wx*0.))
        updates.append((self.v_Wx, self.v_Wx*0.))

        updates.append((self.m_bn_a_beta, self.m_bn_a_beta*0.))
        updates.append((self.v_bn_a_beta, self.v_bn_a_beta*0.))

        updates.append((self.m_bn_c_beta, self.m_bn_c_beta*0.))
        updates.append((self.v_bn_c_beta, self.v_bn_c_beta*0.))



        updates.append((self.m_h0, self.m_h0*0.))
        updates.append((self.v_h0, self.v_h0*0.))

        updates.append((self.m_c0, self.m_c0*0.))
        updates.append((self.v_c0, self.v_c0*0.))

        if self.BN == True:
            updates.append((self.bn_a_mean, self.bn_a_mean*0.))
            updates.append((self.bn_a_var, self.bn_a_var*0.))

            updates.append((self.bn_b_mean, self.bn_b_mean*0.))
            updates.append((self.bn_b_var, self.bn_b_var*0.))

            updates.append((self.bn_c_mean, self.bn_c_mean*0.))
            updates.append((self.bn_c_var, self.bn_c_var*0.))

            updates.append((self.m_bn_c_gamma, self.m_bn_c_gamma*0.))
            updates.append((self.v_bn_c_gamma, self.v_bn_c_gamma*0.))

            updates.append((self.m_bn_a_gamma, self.m_bn_a_gamma*0.))
            updates.append((self.v_bn_a_gamma, self.v_bn_a_gamma*0.))

            updates.append((self.m_bn_b_gamma, self.m_bn_b_gamma*0.))
            updates.append((self.v_bn_b_gamma, self.v_bn_b_gamma*0.))
        

        updates.append((self.n_samples, self.n_samples*0.))
        
        return updates






class linear_layer(object):
    
    def __init__(self, rng, n_inputs, n_units,
        BN=False, BN_epsilon=1e-4,
        dropout=1.,
        binary_training=False, stochastic_training=False,
        binary_test=False, stochastic_test=0):
        
        self.rng = rng
        
        self.n_units = n_units
        print ("        n_units = "+str(n_units))
        self.n_inputs = n_inputs
        print ("        n_inputs = "+str(n_inputs))

        self.dropout = dropout
        print ("        dropout = "+str(dropout))
        


        def initializer(shape, interval_n, interval_p):
            flat_shape = (shape[0], np.prod(shape[1:]))
            a = np.random.normal(0., 1., flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v  # pick the one with the correct shape
            q = q.reshape(shape)
            return q[:shape[0], :shape[1]].astype(theano.config.floatX)

        
        W_values = np.asarray(initializer((n_inputs, n_units),0.,1.),dtype=theano.config.floatX)
        b_values = np.zeros((n_units), dtype=theano.config.floatX)
        
        self.W = theano.shared(value=W_values, name='W')
        self.b = theano.shared(value=b_values, name='b')
        
        self.n_samples = theano.shared(value=np.float32(0),name='n_samples')
        
        # momentum
        self.m_W = theano.shared(value=np.zeros((n_inputs, n_units), dtype=theano.config.floatX), name='m_t_W')
        self.v_W = theano.shared(value=np.zeros((n_inputs, n_units), dtype=theano.config.floatX), name='v_t_W')
        self.m_b = theano.shared(value=b_values, name='m_t_b')
        self.v_b = theano.shared(value=b_values, name='v_t_b')


    def activation(self, z):
        return z
    
    
    def fprop(self, x, can_fit, eval):
        self.x = x 
        if self.dropout < 1.:
            if eval == False:
                srng = theano.sandbox.rng_mrg.MRG_RandomStreams(self.rng.randint(999999))
                mask = T.cast(srng.binomial(n=1, p=self.dropout, size=T.shape(self.x)), theano.config.floatX)
                
                # apply the mask
                self.x = (1./self.dropout) * self.x * mask
            else:
                self.x = self.x     
        
        z = T.dot(self.x, self.W)
        
        # for BN updates
        self.z = z + self.b
        
        y = self.activation(self.z)
        
        return y
    


    def bprop(self, cost):
        clip_ = 20000.
        self.dEdW =  T.clip(T.grad(cost=cost, wrt = self.W), -clip_, clip_)
        norm = theano.tensor._tensor_py_operators.norm(self.dEdW,2)**2
        
        self.dEdb = T.clip(T.grad(cost=cost, wrt = self.b), -clip_, clip_) 
        norm = norm + theano.tensor._tensor_py_operators.norm(self.dEdb,2)**2
        

        return norm
        
    def parameters_updates(self, LR, norm_factor):    
        
        updates = []
        beta1 = 0.9
        beta2 = 0.
        epsilon = 1e-6
        clip_ = 20000.
        t = self.n_samples + 1



        v_t_W = beta2 * self.v_W - LR * self.dEdW * norm_factor
        step_W = self.W + v_t_W
        updates.append((self.v_W, v_t_W))
        step_W = T.clip(step_W, -clip_, clip_)
        updates.append((self.W, step_W))





        v_t_b = beta2 * self.v_b - LR * self.dEdb * norm_factor
        step_b = self.b + v_t_b
        updates.append((self.v_b, v_t_b))
        step_b = T.clip(step_b, -clip_, clip_)
        updates.append((self.b, step_b))
        

        updates.append((self.n_samples, t)) 

        return updates
    

        
    def BN_reset(self):
    
        updates = []
        
        updates.append((self.m_b, self.m_b*0.))
        updates.append((self.v_b, self.v_b*0.))

        updates.append((self.m_W, self.m_W*0.))
        updates.append((self.v_W, self.v_W*0.))

        


        
        updates.append((self.n_samples, self.n_samples*0.))

        return updates


