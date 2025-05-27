#import numpy as np




#**********************************************************
#So basically this is the manual model but uses autograd to calculate the gradients
#**********************************************************

import torch

#The regression will be f = w * x ---> don't care about bias which is the (b) in mx+b

#Sample regression function f = 2 * w
X = torch.tensor([1,2,3,4,5], dtype=torch.float32)
Y = torch.tensor([2,4,6,8,10], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) #initialize weights to not very good

#model prediction: forward pass
def forward(x):
    return w * x

#Calculate the loss: mean_squared error
def loss(y, yPred):
    return ((yPred-y)**2).mean()

#Calculate the gradients to do the backward pass
#Minimize loss : MSE (Mean squared error)
#Formula is MSE = 1/N * (w*x - y) ** 2
#Where N is the numer of inputs, w*x is the predicted Y, and y is the actual Y
#So dJ/dw = 1/N * 2 * (w*x - y) * x
#Where J is the objective loss function
#1/N is a coefficient, (w*x-y) is the thing that is being squared, so we do power rule on that
#multiply by x because of chain rule, y is a constant, but x is the coefficient of w
#dJ/dw = 2x(w*x-y).mean() = 2x(yPred-y).mean()
#def gradient(x,y,yPred):
#    return np.dot(2*x, yPred-y).mean() 
#We can pass in an array

print(f'Prediction before training: f(6) = {forward(6):.3f}')

#Training the Model
learning_rate = 0.01
iters = 100

for epoch in range(iters):
    #prediction: forward pass
    yPred = forward(X) #Ypred is a tensor, forward() is vectorized here
    #print(f"yPred: {yPred}")
    #Loss calculation
    l = loss(Y, yPred) #Although it's vectorized, the .mean() in loss converts to scalar

    #gradients - backward pass
    #dw = gradient(X,Y,yPred) #pass predictions for all values
    #and get back how much we should change the weights
    l.backward() #Will take more epochs because backward() isn't quite as accurate

    #update weights
    with torch.no_grad(): #We don't want updating the weights affecting the gradient
        w -= learning_rate * w.grad
    
    #zero gradients-- because they always accumulate
    w.grad.zero_()

    print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.3f}')


print(f'Prediction after training: f(6) = {forward(6):.3f}')
