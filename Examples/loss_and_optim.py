import torch

#Here, the loss and optimizers will come from torch and torch.nn modules
#Whereas we were manually calculating MSE loss before via a function, we now use torch
#Updating the gradients is also done through torch

import torch.nn as nn

#The regression will be f = w * x ---> don't care about bias which is the (b) in mx+b

#Sample regression function f = 2 * w
X = torch.tensor([1,2,3,4,5], dtype=torch.float32)
Y = torch.tensor([2,4,6,8,10], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) #initialize weights to not very good

#model prediction: forward pass
def forward(x):
    return w * x

#Calculate the loss: mean_squared error
#def loss(y, yPred):
#    return ((yPred-y)**2).mean()

#The Gradient was already automated in the last step: here we use autograd

print(f'Prediction before training: f(6) = {forward(6):.3f}')

#Training the Model
learning_rate = 0.01
iters = 100

loss = nn.MSELoss() #this is a callable function now
optimizer = torch.optim.SGD([w], lr=learning_rate) #Stochastic gradient descent

for epoch in range(iters):
    #prediction: forward pass
    yPred = forward(X) #Ypred is a tensor, forward() is vectorized here
    #print(f"yPred: {yPred}")

    #Loss calculation
    l = loss(Y, yPred) #Uses MSELoss() defined earlier

    #gradients - backward pass
    #dw = gradient(X,Y,yPred) #pass predictions for all values
    #and get back how much we should change the weights
    l.backward() #Will take more epochs because backward() isn't quite as accurate

    #update weights
    #with torch.no_grad(): #We don't want updating the weights affecting the gradient
    #    w -= learning_rate * w.grad
    optimizer.step() #optimization step 


    #zero gradients-- because they always accumulate
    #w.grad.zero_()
    optimizer.zero_grad()

    print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.3f}')


print(f'Prediction after training: f(6) = {forward(6):.3f}')
