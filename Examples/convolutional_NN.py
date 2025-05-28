#Using the CIFAR-10 dataset, with many images and 10 categories, to train an example of a CNN

import torch
import torch.nn as nn
import torch.nn.functional as F #technically we already had this but it's more convinient this way
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Subsequently, convert everything to GPU in case cuda is available

#Hyper-parameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

#dataset has PIL images with pixels of range [0,1] or 0 to 255
#We transform them into Tensors of normalized range -1 to 1
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,.5,.5))])
#Normalize() does converts every value if given a tensor of means and a tensor of standard deviations
#Basically it uses (input[channel] - mean[channel]) / std[channel]
#I'm not sure why, but I do know that the images have 3 channels, RGB

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                             download=True)#, transform=transform)
first_image = train_dataset[0][0]
print("Img obj: ", first_image)
first_image.show()
image_npArray = np.asarray(first_image)
print("Image Array: ", image_npArray)

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                             download=True, transform=transform)
print("Transformed First Data: Normalized and ToTensor()", train_dataset[0][0])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                             download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ConvNeuralNet(nn.Module):
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        #The first 2 means [2,2] and the second is stride, which means how far you move.
        #So moving 2 every time means there is no overlap (horizontally)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120) #See below why...
        self.fc2 = nn.Linear(120, 84) #120 and 84 can be changed but the first and last are fixed
        self.fc3 = nn.Linear(84, 10)

        #Why is fc1 16*5*5? Well, 16 is the number of input channels after the second Conv2d layer,
        #But the final image size is 5x5, which we now want to flatten.

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #First conv and pooling layer
        x = self.pool(F.relu(self.conv2(x))) #Second one
        x = x.view(-1, 16*5*5) #Flatten into 1 dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) #No softmax or other activation because CrossEntropyLoss() auto does it
        return x
        
model = ConvNeuralNet().to(device)

#Define loss and optim
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_dataloader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        #Original shape: [4,3,32,32] = 4, 3, 1024
        #input layer = 3 input channels, 6 output channels, 5 kernel size
        images.to(device)
        labels.to(device)

        outputs = model(images)
        loss = criteria(outputs,labels)

        #Backward gradient calc and optimize weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % 2000 == 0:
            print(f'epoch {epoch+1}, Step {i+1}/{n_total_steps}, Loss:{loss.item():.4f}')

print("Finished Training")

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images,labels in test_dataloader: #Don't enumerate because we don't need to know what iteration we're on
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        #Did it predict correct? Take max of outputs, we don't care about the max val but instead the index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predictions == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predictions[i]
            if(label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = n_correct/n_samples * 100 #Percentage of correct predictions
    print(f'Overall Accuracy Test: {acc}%')

    print("Accuracy per class")
    for i in range(10): #Iterate through all clases: 10 is num classes
        class_acc = n_class_correct[i] / n_class_samples[i] * 100
        print(f'Accuracy of {classes[i]} : {class_acc}%')