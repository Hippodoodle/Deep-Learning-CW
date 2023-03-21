
import collections
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import (ConcatDataset, DataLoader, Dataset, Subset,
                              TensorDataset, random_split)
from torchvision import datasets
from torchvision.transforms import ToTensor


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # input: 3 x 100 x 100
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 100 x 100 x 32
            nn.Conv2d(32, 64, kernel_size=3, stride=1,
                      padding=1),  # 100 x 100 x 64
            nn.MaxPool2d(2, 2),  # 50 x 50 x 64
            nn.Flatten(),

            nn.Linear(50 * 50 * 64, 128),
            nn.Linear(128, 32),
            nn.Linear(32, 12)
        )

    def forward(self, x):
        return self.network(x)


class CustomImageDataset(Dataset):
    def __init__(self, labels, images, transform=None, target_transform=None):
        self.img_labels = labels
        self.img_dir = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return self.img_dir[idx], self.img_labels[idx]


def load_data(data):

    original_dataset = CustomImageDataset(
        data['labels'].long(), data['images'], transform=transforms.ToTensor())

    transform = transforms.Compose(
        [
            transforms.RandomCrop(size=(50, 50)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation((0, 180)),
            transforms.ToTensor(),
        ]
    )

    transformed_dataset1 = CustomImageDataset(
        data['labels'].long(), data['images'], transform=transform)
    transformed_dataset2 = CustomImageDataset(
        data['labels'].long(), data['images'], transform=transform)

    concat_dataset = ConcatDataset(
        [original_dataset, transformed_dataset1, transformed_dataset2])
    dataset_labels = torch.cat(
        [original_dataset.img_labels, transformed_dataset1.img_labels, transformed_dataset2.img_labels], axis=0)
    dataset_images = torch.cat(
        [original_dataset.img_dir, transformed_dataset1.img_dir, transformed_dataset2.img_dir], axis=0)

    # Split twice to get train, val and test splits
    train_indices, val_test_indices, _, val_test_labels = train_test_split(
        range(len(dataset_labels)),
        dataset_labels,
        stratify=dataset_labels,
        test_size=0.4,
        random_state=42
    )

    val_indices, test_indices, _, _ = train_test_split(
        val_test_indices,
        val_test_labels,
        stratify=val_test_labels,
        test_size=0.5,
        random_state=42
    )

    train_split = Subset(concat_dataset, train_indices)

    val_split = Subset(concat_dataset, val_indices)

    test_split = Subset(concat_dataset, test_indices)

    return train_split, val_split, test_split


def main():

    data = torch.load('plankton.pt')

    # get the number of different classes
    classes = data['labels'].unique()
    print('The classes in this dataset are: ')
    print(classes)

    # display the number of instances per class:
    print('\nAnd the numbers of examples per class are: ')
    print(pd.Series(data['labels']).value_counts())

    train_set, val_set, test_set = load_data(data)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    net = Net()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.parallel.DataParallel(net)
    net.to(device)

    # define loss function and optimizer
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train the network
    for epoch in range(3):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data2 in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data2
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')


    def test_accuracy(net, data, device: str = "cpu"):
        _, _, testset = load_data(data)

        testloader = DataLoader(testset, batch_size=4,
                                shuffle=False, num_workers=2)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    print(test_accuracy(net, data, device))


    # Save
    torch.save((net.state_dict(), optimizer.state_dict()), "checkpoint")


    # Evaluate
    model = Net()
    optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Load state dicts
    model_state, optimizer_state = torch.load("checkpoint")
    model.load_state_dict(model_state)
    optimiser.load_state_dict(optimizer_state)

    model.eval()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    model.to(device)

    print(test_accuracy(model, data, device))


if __name__ == "__main__":
    main()
