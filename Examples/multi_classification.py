'''
The softmax and crossEntropy comes into play here in the form of a neural network

'''

import torch
import torch.nn as nn

#Multiclass problem
class MultiNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes): #Hidden size is a custom thing in the hidden layer that could be convolutional // activation function or other
        super(MultiNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes) #Num classes is how many categories the input could be classified as

    def forward(self, x):
        '''
        out = self.linear(x)
        out = self.relu(out)
        out = self.linear2(x)'''
        hidden = self.linear1(x)
        activated = self.relu(hidden)
        out = self.linear2(activated)
        return out #NO SOFTMAX AT THE END bc cross entropy loss does it auto
    
model = MultiNN(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss() #Applies softmax

#Insert training loop here