# 6998 HPML Final Project
# Hanna Gao & Junqi Zou
# This file utilized Weights & Biases to perform hyperparameter turning to find the 
# optimal set of parameters for the model

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import model
import csv
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import time
import os
import argparse
import wandb
import pprint

if not torch.cuda.is_available():
    from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
shape = (44, 44)

# Login to Weights and Biases
wandb.login()

# Define Sweep Configuration
sweep_config = {
    'method': 'random'
    }

metric = {
    'name': 'loss',
    'goal': 'minimize'
    }

sweep_config['metric'] = metric

parameters_dict = {
    'learning_rate': {
        'values': [0.1, 0.05, 0.01, 0.005, 0.001]
        },
    'optimizer': {
        'values': ['adam', 'sgd']
        },
    'learning_rate_decay_start': {
          'values': [80, 100, 120, 150]
        },
    'learning_rate_decay_every': {
          'values': [5, 10, 15]
        },
    'learning_rate_decay_rate': {
          'values': [0.8, 0.85, 0.9, 0.95]
        },
    }

sweep_config['parameters'] = parameters_dict
pprint.pprint(sweep_config)
sweep_id = wandb.sweep(sweep_config, project="6998-proj2")

# Process Dataset
class DataSetFactory:

    def __init__(self):
        images = []
        emotions = []
        private_images = []
        private_emotions = []
        public_images = []
        public_emotions = []

        with open('../dataset/fer2013_balanced.csv', 'r') as csvin:
            data = csv.reader(csvin)
            next(data)
            for row in data:
                face = [int(pixel) for pixel in row[1].split()]
                face = np.asarray(face).reshape(48, 48)
                face = face.astype('uint8')

                if row[-1] == 'Training':
                    emotions.append(int(row[0]))
                    images.append(Image.fromarray(face))
                elif row[-1] == "PrivateTest":
                    private_emotions.append(int(row[0]))
                    private_images.append(Image.fromarray(face))
                elif row[-1] == "PublicTest":
                    public_emotions.append(int(row[0]))
                    public_images.append(Image.fromarray(face))

        print('training size %d : private val size %d : public val size %d' % (
            len(images), len(private_images), len(public_images)))
        train_transform = transforms.Compose([
            transforms.RandomCrop(shape[0]),
            transforms.RandomHorizontalFlip(),
            ToTensor(),
        ])
        val_transform = transforms.Compose([
            transforms.CenterCrop(shape[0]),
            ToTensor(),
        ])

        self.training = DataSet(transform=train_transform, images=images, emotions=emotions)
        self.private = DataSet(transform=val_transform, images=private_images, emotions=private_emotions)
        self.public = DataSet(transform=val_transform, images=public_images, emotions=public_emotions)

# Define PyTorch Dataset
class DataSet(torch.utils.data.Dataset):

    def __init__(self, transform=None, images=None, emotions=None):
        self.transform = transform
        self.images = images
        self.emotions = emotions

    def __getitem__(self, index):
        image = self.images[index]
        emotion = self.emotions[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, emotion

    def __len__(self):
        return len(self.images)

# Define optimizer
def build_optimizer(network, optimizer, learning_rate):
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-3)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=5e-3)
    return optimizer

# Training with Hyperparameter Tuning
def trainWB(config = None):
    with wandb.init(config=config):
        # Initiliaze variables
        classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        network = model.Model(num_classes=len(classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        factory = DataSetFactory()
        batch_size = 128

        # Load the dataset
        training_loader = DataLoader(factory.training, batch_size=batch_size, shuffle=True, num_workers=20)
        validation_loader = {
            'private': DataLoader(factory.private, batch_size=batch_size, shuffle=True, num_workers=20),
            'public': DataLoader(factory.public, batch_size=batch_size, shuffle=True, num_workers=20)
        }
        min_validation_loss = {
            'private': 10000,
            'public': 10000,
            }
        
        config = wandb.config
        lr = config.learning_rate
        epochs = 300
        learning_rate_decay_start = config.learning_rate_decay_start
        learning_rate_decay_every = config.learning_rate_decay_every
        learning_rate_decay_rate = config.learning_rate_decay_rate

        if not torch.cuda.is_available():
            summary(network, (1, shape[0], shape[1]))
        
        # Build Optimizer
        optimizer = build_optimizer(network, config.optimizer, lr)
        
        for epoch in range(epochs):
            network.train()
            total = 0
            correct = 0
            total_train_loss = 0

            # Learning Rate Decay
            if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
                frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
                decay_factor = learning_rate_decay_rate ** frac
                current_lr = lr * decay_factor
                for group in optimizer.param_groups:
                    group['lr'] = current_lr
            else:
                current_lr = lr
            
            print('learning_rate: %s' % str(current_lr))

            it = iter(training_loader)
            total_batch = len(training_loader)

            # Batch Training
            for i in range(total_batch):
                # Load sample from DataLoader
                x_train, y_train = next(it)
                x_train = x_train.to(device)
                y_train = y_train.to(device)
                optimizer.zero_grad()
                y_predicted = network(x_train)
                loss = criterion(y_predicted, y_train)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(y_predicted.data, 1)
                total_train_loss += loss.data
                total += y_train.size(0)
                correct += predicted.eq(y_train.data).sum()
            
            # Calculate Accuracy
            accuracy = 100. * float(correct) / total
            print('----------------------------------------------------------------------')
            print('Epoch [%d/%d] Training Loss: %.4f, Accuracy: %.4f' % (epoch + 1, epochs, total_train_loss / (i + 1), accuracy))
            wandb.log({"loss": total_train_loss / (i + 1), "epoch": epoch + 1})

if __name__ == "__main__":
    wandb.agent(sweep_id, trainWB, count=10)
    
