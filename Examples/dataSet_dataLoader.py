#Use dataset/dataloader from torch to do batch training to optimize code
#This file may only work if run from terminal.
#OHHHHH, in VS Code if the folder you're working in is too big then it will look for all files in the biggest foldere
#For example I had let VS open ML and then I clicked into Learning, but then this file did ./, so it was looking for the "data" folder in 
#the ML instead of learning lol
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
print(os.cpu_count())

#torch.cuda.device('cuda')

class WineDataset(Dataset): #Parent class is torch.utils.data.Dataset
    def __init__(self):
        xy = np.loadtxt('./data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:]) #All the samples without the first column because that is for the wine category
        self.y = torch.from_numpy(xy[:, [0]]) #n_samples, 1
        self.n_samples = xy.shape[0]
        #I'm guessing x,y, and n_samples are all part of the Dataset superclass

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.n_samples

dataset = WineDataset()
first_data = dataset[0] #First sample, the index calls the __getitem__ func 
first_features, first_label = first_data
print(f"Seeing the first Row: features {first_features} , label: {first_label}" )

dataloader = DataLoader(dataset=dataset,batch_size=4, shuffle=True) 
#dataloader = DataLoader(dataset=dataset,batch_size=4, shuffle=True, num_workers=2) 
#num workers manages sub processors, but if i put 2 workers then error hits

print("Seeing first batch? -----------------------------------")
dataiter = iter(dataloader) #Create an iterator object
data = next(dataiter) #In the example, it's dataiter.next() but that no work
features, labels = data
print(features, labels)

#With the iterator object, let's practice iterating the entire dataset
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4) #divide by batch size
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        #forward
        if (i + 1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations} inputs {inputs.shape}')