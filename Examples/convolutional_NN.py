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

#Hyper-parameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

#dataset has PIL images with pixels of range [0,1] or 0 to 255
#We transform them into Tensors of normalized range -1 to 1
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((1,1,1), (1,1,1))])

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
train_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

