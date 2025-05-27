'''
Transforms can be applied to many things: PIL images, tensors, ndarrays, or custom data during the creation of the dataset

On Images: 
CenterCrop, Grayscale, Pad, RandomAffine, RandomCrop, 
RandomHorizontalFlip, RandomRotation, Resize, Scale

On Tensors:
LinearTransformation, Normalize, RandomErasing

Conversion:
ToPILImage: From tensor to np.ndarray
ToTensor: from numpy.ndarray to PILImage

Generic: Use lambda
Custom: Write own class

Compose multiple Transforms: comp = transforms.Compose(...)
'''
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

#Copy the same dataSet class from the other dataset file: Wine.csv

class WineDataset(Dataset): #Parent class is torch.utils.data.Dataset
    def __init__(self, transform): #Pass in a new transform function
        xy = np.loadtxt('./data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)

        #DIDN'T CALL torch.from_numpy on self.x and self.y BECAUSE THAT'S THE TRANSFORM: ToTensor()
        self.x = xy[:, 1:] #All the samples without the first column because that is for the wine category
        self.y = xy[:, [0]] #n_samples, 1
        
        
        self.n_samples = xy.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform: #if there is a transform is available
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

class ToTensor: #Standard transform that could be implemented before hand but whatever
    def __call__(self, sample): #Make the class a callable function, useful below
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    
class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

print("\n\nPrinting First Data with No transform: --------------------------------------------")
dataset = WineDataset(transform=None) #Without the transform, it would still be a np array
first_data = dataset[0]
first_feature, first_label = first_data
print(first_feature, first_label)
print(type(first_feature), type(first_label))



print("\n\nPrinting First Data with ToTensor() transform: --------------------------------------------")
dataset = WineDataset(transform=ToTensor()) #Without the transform, it would still be a np array
first_data = dataset[0]
first_feature, first_label = first_data
print(first_feature, first_label)
print(type(first_feature), type(first_label))

print("\n\nPrinting First Data with ToTensor() and MulTransform() transform: --------------------------------------------")
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)]) #Here, factor equals 2, so all values should be doubled (and converted to tensor)
dataset = WineDataset(transform=composed) #Multiplies the dataset by a factor of 'factor' then converts to Tensor form
first_data = dataset[0]
first_feature, first_label = first_data
print(first_feature, first_label)
print(type(first_feature), type(first_label))


#Same except everything is multiplied by 4?
print("\n\nSame except everything is multiplied by 4: -------------------------------------------")
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)]) #Here, factor equals 2, so all values should be doubled (and converted to tensor)
dataset = WineDataset(transform=composed) #Multiplies the dataset by a factor of 'factor' then converts to Tensor form
first_data = dataset[0]
first_feature, first_label = first_data
print(first_feature, first_label)
print(type(first_feature), type(first_label))

