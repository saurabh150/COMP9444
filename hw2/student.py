#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.
x
a. choice of architecture, algorithms and enhancements (if any)  

The architecture we decided to use is composed of 6 convolutional layers since they are commonly used to extract features from images. 
Since we decided not to use Grayscale, the input has 3 channels (RGB) given to the first layer. 
The number of channels grows as the input is passed to more layers to extract more features, but a max-pooling layer is applied at the output of each 2 convolutional layers to output the most notable features. 
To do the training faster and make the model more stable, 3 batch normalization layers were applied after every 2 convolutional layers. 
We also used LSTM as we found out that it is a better predictor of the CNN output over adding more linear layers. 
Lastly, To make the model generalize better, we used the dropout technique on LSTM and linear layers. 

b. choice of loss function and optimiser  

For implementing an image classification model with dataset not a 100% correct, the best and the most flexible optimizer is the Adam optimizer. 
Adam optimizer is used as an all-purpose optimizer with good documentation of all its features. 
Whereas for the loss function, after researching the best loss function for image classification, it was recommended by most experienced programmers to use the NLLLoss function. 
It is a simple loss function which would help maintain the simplicity of model for the size limitations. 

c. choice of image transformations  

We first tried to implement the model with no image transformation, but noticed that the model would generalize and overfit quickly. 
Then we choose to perform transformations. We firstly performed a random horizontal flip followed by a random rotation by 45 degrees. 
This was done to not generalize features in the same segment of the image. 
We then normalized the data set by a mean of 0 and standard deviation of 1, followed by converting to tensor.
 We had initially tried to also run grayscale but discovered that all it did was converge to the same accuracy but only slower. 
 We choose to resize the image to 160 by 160 after trial and error.  

d. tuning of metaparameters  

We modified the initial learning rate to be 0.001 with a weight decay of 0.001 too. 
We noticed the model would avoid quickly learning and overfitting by these values. 
We used the batch size of 25 because not only it’s how our model will be ran when marked, but it updates the gradient more often and extracts the correct features without over fitting.
We ran the model with 370 epochs to train with a train_val_split of 0.99 so we would have about 80 images to verify if the model still worked right.  

e. use of validation set, and any other steps taken to improve generalization and avoid overfitting 

We used a scheduler, the StepLR scheduler multiplies the learning rate with a factor of 0.9 every 20 epochs.
We noticed that doing this slowed down the learning process of the model towards the convergence part, and prevented early overfitting of data and improved the generalization.  

"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    train_transform = transforms.Compose([
        transforms.Resize((160,160)),
        #transforms.Grayscale(1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0], std  = [1])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((160,160)),
        #transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0], std  = [1])
    ])

    if mode == 'train':
        return train_transform
    elif mode == 'test':
        return test_transform


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################

# 85% Test Accuracy
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, padding=0)        
        self.conv2 = nn.Conv2d(16, 16, 3, padding=0)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 3, padding=0)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=0)     
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, 3, padding=0)
        self.conv6 = nn.Conv2d(128, 256, 3, padding=0)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.maxpool = nn.MaxPool2d((3,3))
        
        self.lstm = nn.LSTM(16, 16, 2, dropout=0.3)
        
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 8)
        
    def forward(self, input):
        
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = self.maxpool(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = self.maxpool(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.bn3(x)
        x = self.maxpool(x)
        
        x = torch.flatten(x, 2)
        x,_ = self.lstm(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        
        x = F.log_softmax(self.fc2(x), 1)
        return x

net = Network()

############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)

loss_func = nn.NLLLoss()

############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    return

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9, verbose=True)
#scheduler = None

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.99
batch_size = 25
epochs = 500
