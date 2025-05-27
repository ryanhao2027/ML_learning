'''
Activation Function provide a very important non-linear transformation at each step of a neural network (usually)
1) Binary Step function
2) Sigmoid
3) TanH
4) ReLU
5) Leaky ReLU
6) softmax


1) Step Function: Not used in practice: but returns (x > 0 ? 1 : 0)

2)Sigmoid f(x) = 1/(1+e^-x) : activation increases as x increases, goes from 0 to 1
----> Last layer of a binary classification problem

3) TanH f(x) = 2/(1+e^-2x) - 1: scaled and shifted version of the sigmoid function
----> Used in hidden layers

4) ReLU f(x) = MAX(0, x), AKA returns (x > 0 ? x : 0)...
----> Most popular. When in doubt, use ReLU for hidden layers or linear

5) Leaky ReLU f(x) = (x > 0 ? x : x * a) where a is very very small. 
----> Improved version of ReLU(), tries to solve the vanishing gradient problem where gradients just die because the activation is just eternally zero. 
----> If you notice that your weights are just not changing, try changing ReLU to Leaky ReLU

6) Softmax normalization: Vectorized function: returns a vector of probabilities (while sigmoid is just a scalar)
S(y[i]) = e^(y[i]) / summation of all e^(y[i])
----> Good in the last layer of multi-classification problems

'''

import torch
import torch.nn as nn
#Sometimes, activation functions like leaky_relu aren't included in torch, so we need to import functional
import torch.nn.functional as F

#How to implement activation functions in a NN class? 
#2 options

#Option 1: create nn modules for each activation func
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)'''
        hidden = self.linear1(x)
        activated = self.relu(hidden)
        final = self.linear2(activated)
        out = self.sigmoid(final)
        return out
    
#option 2 (apply activation functions directly in the forward pass)

class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        '''
        out = self.linear1(x)
        out = torch.ReLU(out)
        out = self.linear2(out)
        out = torch.Sigmoid(out)
        '''
        '''
        hidden = self.linear1(x)
        activated = torch.ReLU(hidden)
        final = self.linear2(activated)
        out = torch.Sigmoid(final)'''
        activated = torch.ReLU(self.linear1(x))
        out =  torch.Sigmoid(self.linear2(activated))
        return out