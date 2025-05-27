'''
Conversely to usage of softmax and crossEntropyLoss(), BCELoss is different and implements the sigmoid function

'''

import torch
import torch.nn as nn

#Binary Classification problem: Is this picture of a dog? Yes or no
class BinNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes): #Hidden size is a custom thing in the hidden layer that could be convolutional // activation function or other
        super(BinNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1) #Num classes is 1 because it's either yes or no

    def forward(self, x):
        '''
        out = self.linear(x)
        out = self.relu(out)
        out = self.linear2(x)'''
        hidden = self.linear1(x)
        activated = self.relu(hidden)
        out = self.linear2(activated)
        y_pred = torch.sigmoid(out) #Unlike with crossEntropyLoss() where softmax is auto, we must implement sigmoid outselves here.
        return y_pred 
    
model = BinNN(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.BCELoss() #Applies softmax

#Insert training loop here