#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

import os
import copy

def model_ftrs(model):
    # ResNet18
    # Params: 11.7M
    # GFLOPs: 1.814
   if model == 'resnet':
        model_ft = models.resnet18(weights='DEFAULT')
        num_ftrs = model_ft.fc.in_features
    # Convnext Tiny
    # Params: 28.6M
    # GFLOPs: 4.456
   elif model == 'convnext':
        model_ft = models.convnext_tiny(weights='DEFAULT')
        num_ftrs = model_ft.classifier[2].in_features
    # EfficientNet B4
    # Params: 19.3M
    # GFLOPs: 4.394
   elif model == 'efficientnet':
        model_ft = models.efficientnet_b4(weights='DEFAULT')
        num_ftrs = model_ft.classifier[1].in_features
    # MobileNet v3
    # Params: 28.6M
    # GFLOPs: 4.456
   elif model == 'mobilenet':
        model_ft = models.mobilenet_v8_small(weights='DEFAULT')
        num_ftrs = model_ft.classifier[3].in_features
    # ShuffleNet
    # Params: 7.4M
    # GFLOPs: 0.583
   elif model == 'shufflenet':
        model_ft = models.shufflenet_v2_x2_0(weights='DEFAULT')
        num_ftrs = model_ft.fc.in_features
    # Swin_T
    # Params: 28.3M
    # GFLOPs: 4.491
   elif model == 'swin':
        model_ft = models.swin_s(weights='DEFAULT')
        num_ftrs = model_ft.head.in_features
   else:
        print("Available models:\n")
        print("resnet, convnext, efficientnet, mobilenet, shufflenet, swin\n")
        print("please pass correct model")

    return model_ft, num_ftrs

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders["val"]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(f"predicted: {class_names[preds[j]]}")
                plt.imsave(f'res_{i}.png', inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

if __name__ == '__main__':
# Load Data
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    data_dir = "../data/"
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=32, shuffle=True, num_workers=4
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft, num_ftrs = model_ftrs("resnet")
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(
        model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50
    )
    model.save(model_ft)
    visualize_model(model_ft)
