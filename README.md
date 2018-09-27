# Learning Recurrent Binary/Ternary Weights


This file contains the code for the proposed training method to replicate the obtained results in the paper and help hardware groups to implement the idea on custom hardware.


Dependencies: 
Theano, Pytorch, Blocks and Fuel.


To run this model, open ./Codes/dataset-name/dataset-name.py and set the training parameters as follows:


Baseline:
    BN_LSTM = True
    binary_training = False
    ternary_training = False
    stochastic_training = False

Learning binary weights using the stochastic training approach:
    BN_LSTM = True
    binary_training = True
    ternary_training = False
    stochastic_training = True

Learning ternary weights using the stochastic training approach:
    BN_LSTM = True
    binary_training = False
    ternary_training = True
    stochastic_training = True



