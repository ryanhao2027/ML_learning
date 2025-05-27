import torch
import torch.nn as nn

#We've already done the backward pass by automatically calculating gradients using autograd
#We've also used the optimizer module and the MSELoss() from Torch

#Final Step is to do the forward pass by using the pipeline for a model from torch

#The regression will be f = w * x ---> don't care about bias which is the (b) in mx+b

#Sample regression function f = 2 * w
X = torch.tensor([[1],[2],[3],[4],[5]], dtype=torch.float32) #Sample input
Y = torch.tensor([[2],[4],[6],[8],[10]], dtype=torch.float32) #Sample output

X_test = torch.tensor([[6],[7]], dtype=torch.float32)

n_samples, n_features = X.shape #5 samples --> 5 test cases
print(n_samples, n_features) #1 feature per sample --> param used to do regression

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size) #this is the only layer that we have
#because it's a very simple linear regression

#w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) #initialize weights to not very good

#model prediction: forward pass
#def forward(x):
#    return w * x

#The loss has already been optimized so we need not manually calc
#The Gradient was already automated in the last step: here we use autograd

print(f'Prediction before training: f(6) = {model(X_test)}')

#Training the Model
learning_rate = 0.01
iters = 1000

loss = nn.MSELoss() #this is a callable function now
#optimizer = torch.optim.SGD([w], lr=learning_rate) #Stochastic gradient descent
#Change to: 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(iters):
    #prediction: forward pass
    yPred = model(X) #Ypred is a tensor, forward() is vectorized here
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

    #For visualization of the weights
    [w,b] = model.parameters()
    print(f'epoch {epoch+1}: w = {w[0][0]:.3f}, loss = {l:.3f}')


print(f'Prediction after training: f(6) = {model(X_test)}')
