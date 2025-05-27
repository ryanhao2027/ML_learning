#Using softmax and crossentropy in NN
import torch
import torch.nn as nn
import numpy as np

def softmax(x): #Softmax(x sub i) = e^(x sub i) / summation of all e^(x sub i)
    return np.exp(x) / np.sum(np.exp(x), axis=0) #Exp() takes e^x, so basically it outputs a probability 

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x) #I did the math: e^2 + e^1 + e^0.1 is about 11.212 = summation of all e^(x[i])
#e^2 // 11.212 is indeed 0.659, and everything else also checks out.
print('softmax np: ', outputs)

#x is same
x = torch.from_numpy(x)
outputs = torch.softmax(x, dim=0)
print(outputs) #Basically does the same thing as my manual function

'''
Softmax is often used with cross entropy loss
Cross Entropy, often defined as D(Y hat, Y), is defined as the following.
where Y hat equals the probability vector of classifications
and where Y equals the vector where the correct classification is 1 and all other vals are 0
(i.e. ONE HOT ENCODING)
and where N equals the amount of possible categories (i.e. size of vector)

VERY IMPORTANT: PLEASE_NOTE ------ HERE LOG means LN() (i.e. Log base e), which took me so long to figure out because I thought it was multiplying by some constant


D(Y hat, Y) = -1/N * summation of Y[i] * log(Y hat[i])       ------> -1/N is a constant (might be batch normalization)

EX: Y = [1,0,0] and Y hat = [0.7,0.2,0.1] -> D() = -1/N * (1 * log(0.7) + 0 + 0) = -1/3 * 1 * log(.7) = 0.35 -------> IGNORE 1/N, it's just for normalizing multiple batches
'''

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

#y must be one hot encoded:
#If correct classification equals
#0: [1 0 0] 
#1: [0 1 0]
#2: [0 0 1]
Y = np.array([1,0,0])

#y_pred is a vector of probabilities
Y_pred_good = np.array([0.7, 0.2, 0.1]) #Good prediction because 70% estimate of right answer
Y_pred_bad = np.array([0.1, 0.3, 0.6])
lGood = cross_entropy(Y, Y_pred_good)
lBad = cross_entropy(Y, Y_pred_bad)
print(f'Loss Good numpy: {lGood:.4f}')
print(f'Loss Bad  numpy: {lBad:.4f}')

#Let's do it with nn.CrossEntropyLoss: 
#BE CAREFUL HERE: nn.CrossEntropyLoss already applies nn.LogSoftmax + nn.NLLLoss (negative log likelihood loss)
#SO DO NOT IMPLEMENT SOFTMAX OUTSELVES
#PLUS, Y has class labels, not One-Hot encoded
#Y pred has raw scores (logits), not normalized via softmax

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0]) #Instead of making a full array w/ ones and zeros, just put the correct class
#nsampples * nclasses = 1x3
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]]) #It must be an array inside another array because the array is all of the classes, and we're only testing one sample
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]]) #If we add more samples, then we add more elements to the overall array

lGood = loss(Y_pred_good, Y)
lBad = loss(Y_pred_bad, Y)

print(f'Good: {lGood.item()}')
print(f'Bad : {lBad.item()}')

_, predictionsGood = torch.max(Y_pred_good, 1) #Get the highest value of Y_pred_good along the first dimension
_,  predictionsBad = torch.max(Y_pred_bad, 1)
print(f"Good Prediction:{predictionsGood}")
print(f"Bad Prediction: {predictionsBad}")


print("More Samples -------------------------------------------")
#Let's try some more samples
Y = torch.tensor([2, 0, 1]) #Multiple samples means multiple targets
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
Y_pred_bad =  torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [1.1, 0.3, 1.6]])

lGood = loss(Y_pred_good, Y)
lBad =  loss(Y_pred_bad, Y)
print(f"Good Loss: {lGood}")
print(f"Bad  Loss: {lBad}")
#Printing the actual  predictions:

_, predictionsG = torch.max(Y_pred_good, 1)
_, predictionsB = torch.max(Y_pred_bad, 1)
print(f"Correct Values  : {Y}")
print(f"Good predictions: {predictionsG}")
print(f"Bad  predictions: {predictionsB}")

'''
The neural network use of this is shown in the multi_classification.py file
'''