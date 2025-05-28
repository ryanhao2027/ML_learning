#Use MNIST dataset
#DataLoader, Transformation to setup data
#Multilayer Neural Net, activation function
#Loss and optimizer implemented in training loop
#Training loop (batch training)
#Model evaluation
#GPU support

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#device config: See if we can use cuda or only CPU
print("Is cuda available? : ", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper parameters for model
#input size: 28x28 image
input_size = 784 #28*28
hidden_size = 100 #slightly arbitrary
num_classes = 10 #All of the numbers 0-10
num_epochs = 1
batch_size = 100
learning_rate = 0.001

#import the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                          transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                          transform=transforms.ToTensor(),download=True)
#Set the dataloader classes
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                           shuffle=False)

examples=iter(train_loader)
samples,labels=next(examples) #Get the first sample, first label so we can see size
print("Sample shape, label shape: ", samples.shape, ", ",  labels.shape)
#Sample size = [100, 1, 28, 28] bc batch_size = 100, only 1 channel/feature, 28x28 is the image
#Labels size = [100] because batch_size = 100 and it's just the correct classes
print(samples, labels)

for i in range(6): #Visualizing the first 6 samples
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0], cmap='gray')
#plt.show()

class FeedForward_NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForward_NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, 10)
        #self.softmax = nn.Softmax() This gets auto implemented with CrossEntropyLoss

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = FeedForward_NeuralNet(input_size, hidden_size, 10)

#Loss and optimizer definition
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs): 
    for i, (images, labels) in enumerate(train_loader): #The only reason to enumerate it is so that we get i
        #Use i to track how many iterations into the batch we are, so we can print the accuracy at every 100 iter's

        #Flatten images
        #[100, 1, 28, 28] to [784,1] or the other way
        images = images.reshape(-1, 28*28).to(device) 
        #technically "to(device)" this does nothing if on cpu right?
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criteria(outputs, labels) #outputs are program outputs, labels are the correct labels (one hot encoded)

        #Backward pass and gradient update
        optimizer.zero_grad()
        loss.backward() #Calc gradients
        optimizer.step() #update weights

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs} step {i+1}/{n_total_steps} loss = {loss.item()}')

#Testing the fininshed model

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device) 
        outputs = model(images)

        #Counting whether the output is correct
        #value, index
        #print("Output Probabilities: ", outputs[0])
        sup, predictions = torch.max(outputs, 1)
        #Max basically returns a tensor containing the max value and max index of the input tensor
        #along a certain dimension, which is the second arg specified. It's not max of outputs and 1. 
        #print("sup,", sup[0])
        #print("Predict:", predictions[0])
        n_samples += labels.shape[0] #Adds 100 for every batch of test data
        #lables.shape[0] basically is the batch size
        n_correct += (predictions == labels).sum().item() #Add the number of correct predictions per batch

    acc = 100.0 * n_correct / n_samples #percentage of correct in all batches
    print("Overall Model Accuracy: ", acc)
