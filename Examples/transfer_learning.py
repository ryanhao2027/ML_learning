#Transfer learning is modifying an existing machine learning model to classify new stuff
#Because CNN's can be extremely time consuming to train, training a new model for each new task is bad
#Here we use the existing ResNet-18 pre-trained pytorch CNN (trained on over 1M images,
#can classify objects into 1000+ categories)

#New stuff: ImageFolder, Scheduler, Transfer Learning
import torch
import torch.nn as nn 
from torch.optim import lr_scheduler
import numpy as np
import torchvision #for computer vision stuffs
from torchvision import datasets, models, transforms #use pretrained dataset, model, and apply transforms
import matplotlib.pyplot as plt #visualize data
import time #for getting the speed/time_elapsed of training
import os
import copy

#device config
torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#The video has these weird values that aren't the same which I commented out... 
#mean =  np.array([0.485, 0.456, 0.406])
mean = np.array([0.5, 0.5, 0.5]) #3 channels. RGB
#std = np.array([0.229, 0.224, 0.225])
std = np.array([0.25, 0.25, 0.25])

data_transforms = { #dictionary with train and test transforms
    'train': transforms.Compose([transforms.RandomResizedCrop(224), #random error
                                      transforms.RandomHorizontalFlip(), #random error
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean,std)]), #normalize data
    'eval': transforms.Compose([transforms.Resize(256), #In the video it says 'val' but it's the same as test or eval
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean,std)])
}



#import data -- I downloaded the data from the video
data_dir = 'data/hymenoptera_data'
phases = ['train', 'eval']

#Define datasets, dataloaders, and dataset sizes as dictionaries where keys are 'train' or 'eval'
#So os.path.join() basically just makes a string that concatenates the two paths. 
#--It doesn't make new files/folder
image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                    for x in phases} #for x in ['train', 'eval']
dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0)
               for x in phases}

dataset_sizes = {x:len(image_datasets[x]) for x in phases}
class_names = image_datasets['train'].classes 
print("Class Names: ", class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criteria, optimizer, scheduler, num_epochs=25):
    #Slightly different training function
    since = time.time() #For marking the time_elapsed of training

    best_model = copy.deepcopy(model.state_dict()) #Copy the best model --> this will be updated
    best_acc = 0.0 #initialize best accuracy to 0

    cur_time = since
    for epoch in range(num_epochs): 
        print(f'Epoch {epoch+1}/{num_epochs}') #Formatting (good for visualization)
        print('-' * 50)

        #Each epoch has a training and evaluation phase:
        for phase in phases:
            isTrain = (phase == 'train') #because we must check this multiple times, make it a variable
            #In the tutorial it just calls phase == 'train' a bunch of times but I changed it 
            if isTrain:
                model.train() #sets model to training mode
            else: 
                model.eval()
            running_loss = 0.0
            running_correct_predicts = 0.0

            #Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                #forward pass (only calc gradients if in train)
                with torch.set_grad_enabled(isTrain):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)  #Get max of outputs along dimension 1
                    loss = criteria(outputs, labels)

                    #backward+optimize(change weights) only if in training mode
                    if isTrain:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                #Stats
                running_loss += loss.item()
                running_correct_predicts += torch.sum(predictions==labels)
            if isTrain:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct_predicts / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f}, Acc:{epoch_acc:.4f}')

            #if in testing and the model is better than the previous best, save it
            if phase == 'eval' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())
            epoch_time = time.time() - cur_time
            cur_time = time.time()
            print(f"Epoch #{epoch+1} {phase} Time: {epoch_time}")
        print()

    time_elapsed = time.time() - since
    print(f"Training Complete in {time_elapsed//60:0.0f} minutes, {time_elapsed%60:0.0f} seconds")
    
    #Now, the last one isn't necessarily the best one, so we load the best one
    model.load_state_dict(best_model)
    return model

#NOW, FINALLY, we get to use transfer learning-------------------------------------
model = models.resnet18(pretrained=True)
#If we don't freeze all layers, the model trains all 18 conv2d layers as well so will be very slow
#Better to only train the last fc layer
for param in model.parameters():
    param.requires_grad = False

#Re-define the fully connected layer. As a standard, it's called fc.
num_ftrs = model.fc.in_features #get num features so we can know what size of input is when we re-define
print("FC Features:", num_ftrs)
model.fc = nn.Linear(num_ftrs,2) #2 because now we just have 2 classes: ants/bees
model.to(device) #Do this after we've modified the layer bc now we are sending it to GPU
#define loss and optim
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

#Hyper params
num_epochs = 20

#Define the scheduler
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) 
#Every (step_size) epochs, lr is multiplied by (gamma)

model = train_model(model, criteria, optimizer, step_lr_scheduler, num_epochs=20) #get the new best model







        


