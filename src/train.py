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

if not torch.cuda.is_available():
    from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

shape = (44, 44)

# Process the dataset
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


def main():
    # Define Command line argument parser
    parser = argparse.ArgumentParser(description='Facial Expression Recognition')
    parser.add_argument('--n', default=1, type=int, help='number of dataloader workers')
    args = parser.parse_args()

    for arg, value in vars(args).items():
        print(arg, ":", value)

    # variables  -------------
    batch_size = 128
    lr = 0.005
    epochs = 300
    learning_rate_decay_start = 100
    learning_rate_decay_every = 15
    learning_rate_decay_rate = 0.85
    # ------------------------

    # Initiazlie variable for time profiling 
    data_loading_time = []
    training_time = []
    total_epoch_time = []
    total_validation_time = []


    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    network = model.Model(num_classes=len(classes)).to(device)
    if not torch.cuda.is_available():
        summary(network, (1, shape[0], shape[1]))

    optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=5e-3)
    criterion = nn.CrossEntropyLoss()
    factory = DataSetFactory()

    # Initialize dataloader for training and validation set
    training_loader = DataLoader(factory.training, batch_size=batch_size, shuffle=True, num_workers=args.n)
    validation_loader = {
        'private': DataLoader(factory.private, batch_size=batch_size, shuffle=True, num_workers=args.n),
        'public': DataLoader(factory.public, batch_size=batch_size, shuffle=True, num_workers=args.n)
    }

    min_validation_loss = {
        'private': 10000,
        'public': 10000,
    }

    # Start Iterative Training
    for epoch in range(epochs):
        # Profiliing time for each epoch
        epoch_start = time.perf_counter()
        network.train()
        total = 0
        correct = 0
        total_train_loss = 0

        # Learning Rate decay
        if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
            frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
            decay_factor = learning_rate_decay_rate ** frac
            current_lr = lr * decay_factor
            for group in optimizer.param_groups:
                group['lr'] = current_lr
        else:
            current_lr = lr

        it = iter(training_loader)
        total_batch = len(training_loader)
        total_train_time = 0
        total_dataload_time = 0

        for i in range(total_batch):
            # Profiling data loading time
            start = time.perf_counter()
            x_train, y_train = next(it)
            end = time.perf_counter()
            total_dataload_time += end - start
            
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            # Profiling training time
            start = time.perf_counter()
            optimizer.zero_grad()
            y_predicted = network(x_train)
            loss = criterion(y_predicted, y_train)
            loss.backward()
            optimizer.step()
            end = time.perf_counter()
            total_train_time += end - start

            _, predicted = torch.max(y_predicted.data, 1)
            total_train_loss += loss.data
            total += y_train.size(0)
            correct += predicted.eq(y_train.data).sum()

        accuracy = 100. * float(correct) / total
        print('----------------------------------------------------------------------')
        print('Epoch [%d/%d] Training Loss: %.4f, Accuracy: %.4f' % (
            epoch + 1, epochs, total_train_loss / (i + 1), accuracy))

        data_loading_time.append(total_dataload_time)
        training_time.append(total_train_time)

        # Switch to model Evaluation mode
        network.eval()

        # Apply Dynamic Post Training QUantization 
        quantized_model = torch.quantization.quantize_dynamic(
            network, 
            {nn.Linear, nn.Conv2d}, 
            dtype=torch.qint8
            )
        # quantized_model = network

        # model inference with validation sets
        validation_time = 0
        with torch.no_grad():
            for name in ['private', 'public']:
                total = 0
                correct = 0
                total_validation_loss = 0
                
                for j, (x_val, y_val) in enumerate(validation_loader[name]):
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    # Profiling validation time
                    inference_start = time.perf_counter()
                    y_val_predicted = quantized_model(x_val)
                    inference_end = time.perf_counter()
                    inference_time = inference_end - inference_start
                    validation_time += inference_time
                    val_loss = criterion(y_val_predicted, y_val)
                    _, predicted = torch.max(y_val_predicted.data, 1)
                    total_validation_loss += val_loss.data
                    total += y_val.size(0)
                    correct += predicted.eq(y_val.data).sum()

                accuracy = 100. * float(correct) / total
                # Save model
                if total_validation_loss <= min_validation_loss[name]:
                    if epoch >= 10:
                        print('saving new model')
                        state = {'net': quantized_model.state_dict()}
                        torch.save(state, '../trained/%s_model_%d_%d.t7' % (name, epoch + 1, accuracy))
                    min_validation_loss[name] = total_validation_loss

                print('Epoch [%d/%d] %s validation Loss: %.4f, Accuracy: %.4f' % (
                    epoch + 1, epochs, name, total_validation_loss / (j + 1), accuracy))
        
        total_validation_time.append(validation_time)
        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        total_epoch_time.append(epoch_time)

    
    data_loading_time = np.array(data_loading_time)
    training_time = np.array(training_time)
    total_epoch_time = np.array(total_epoch_time)
    total_validation_time = np.array(total_validation_time)

    print()
    print('Avergae Data Loading Time per Epoch: ', np.mean(data_loading_time))
    print('Avergae Training Time per Epoch: ', np.mean(training_time))
    print('Avergae Validation Time per Epoch: ', np.mean(total_validation_time))
    print('Avergae Time per Epoch: ', np.mean(total_epoch_time))

    # np.save('/home/qg2205/HPML-project/data_loading_time' + str(args.n) + '.npy', data_loading_time)
    # np.save('/home/qg2205/HPML-project/training_time.npy', training_time)
    # np.save('/home/qg2205/HPML-project/total_epoch_time.npy', total_epoch_time)
    

if __name__ == "__main__":
    main()
