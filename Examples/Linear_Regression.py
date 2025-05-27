#Similar to the full_pipeline
#1) Design model (input size, output size, forward pass)
#2) Loss and optimizier
#3) Training loop
# -forward pass: compute prediction and loss
# -backward pass : calculate gradients
# -update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#Step 0: prepare data
X_np, Y_np = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_np.astype(np.float32))
Y = torch.from_numpy(Y_np.astype(np.float32))
print("X shape: " + str(X.shape))
print("Y shape: " + str(Y.shape))
#print(Y)
Y = Y.view(Y.shape[0], 1)
print("Y shape: " + str(Y.shape))

n_samples, n_features = X.shape
#Step 1: Define model
input_size = n_features
output_size = 1 #Doesn't have anything to do with samples/features, but instead is hard coded in

#Step 2: Loss and optimizer definition
learning_rate = 0.01
loss = nn.MSELoss()
model = nn.Linear(input_size, output_size)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Step 3: training loop
num_epochs = 100
for epoch in range(num_epochs):
    #forward pass and prediction
    y_pred = model(X)
    l = loss(y_pred, Y)
    #backward pass
    l.backward()
    #update weights
    optimizer.step()

    optimizer.zero_grad()

    if(epoch +1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {l.item():.4f}')
        [x,b] = model.parameters()
        #print(str(x) + str(b))
        

#plot results
print(model(X))
print(model(X).detach()) #Returns a new copy of the tensor with no grad
predicted = model(X).detach().numpy()
plt.plot(X_np, Y_np, 'ro')
plt.plot(X_np, predicted, 'b')
plt.show()