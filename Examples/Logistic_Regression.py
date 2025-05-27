import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#0)Prep Data
bc = datasets.load_breast_cancer()
X, Y = bc.data, bc.target
n_samples, n_features = X.shape
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=1234) #Random state is basically a seed

#Scale data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))
print("X_train::" + str(X_train))
print("X_test::" + str(X_test))
print("Y_train::" + str(Y_train))
print("Y_test::" + str(Y_test))

Y_train = Y_train.view(Y_train.shape[0], 1) #Make it into a column vector
Y_test = Y_test.view(Y_test.shape[0], 1)

#1) Set up the model
#f = wx + b, sigmoid function at the end

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__() #Pass the class name and the object for some reason?
        self.linear = nn.Linear(n_input_features, 1)
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegression(n_features)

#2) Loss and optimizer
learning_rate = 0.01
crit = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#3)Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    #forward pass & prediction
    y_pred = model(X_train)
    loss = crit(y_pred, Y_train)

    #backward pass to calc gradients
    loss.backward()

    #update weights
    optimizer.step()

    optimizer.zero_grad() #Empty gradients

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

#Model Testing
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cls = y_pred.round() #Don't want to calculate gradient on this step
    #Round to the nearest whole number because it's a prediction: 0 = no cancer 1 = cancer
    acc = y_pred_cls.eq(Y_test).sum() / float(Y_test.shape[0])
    print(y_pred_cls)
    print(Y_test)
    print(y_pred_cls.eq(Y_test))
    print(Y_test.shape)
    print(f'accuracy = {acc:.4f}')
