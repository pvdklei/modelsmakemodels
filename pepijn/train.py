

import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import utils

def train(model, 
          optimizer, 
          trainloader, 
          testloader, 
          criterion=nn.CrossEntropyLoss(), 
          epochs=5, 
          loaders=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Training on: ", device)

    train_losses, test_losses = [], []

    start_time = time.time()

    for epoch in range(epochs):

        # validation
        testloss = []
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                out = model(images)
                loss = criterion(out, labels)
                testloss.append(loss.item())
        testloss = np.mean(testloss)
        print(f"Epoch {epoch}, Validation loss: {testloss}")
        test_losses.append(testloss)


        # training
        trainloss = []
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            trainloss.append(loss.item())
        trainloss = np.mean(trainloss)
        print(f"Epoch {epoch}, Training loss: {trainloss}")
        train_losses.append(trainloss)

        if loaders:
            trainloader, testloader = loaders()

    end_time = time.time()

    # final accuracy
    totacc = 0
    n = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            out = model(images)
            topv, topi = torch.topk(out, 1, dim=1)
            labels.resize_(*topi.shape)
            eq = topi == labels
            acc = torch.mean(eq.type(torch.FloatTensor))
            totacc += acc.item()
            n += 1
    totacc /= n
    print(f"The final total accuracy is: {totacc * 100}")
    return train_losses, test_losses, totacc, end_time - start_time


def autotrain(model, optimizer, trainloader, testloader, criterion=nn.CrossEntropyLoss(), epochs=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(device)

    lowest_testloss = np.Inf

    for epoch in range(epochs):

        # training
        trainloss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, images)
            loss.backward()
            optimizer.step()
            trainloss += loss.item()
        print(f"Epoch {epoch}, Training loss: {trainloss}")

        # validation
        testloss = 0
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                out = model(images)
                loss = criterion(out, images)
                testloss += loss
        print(f"Epoch {epoch}, Validation loss: {testloss}")
        lowest_testloss = testloss

def training_eval(save_as="evaluation", 
                  descr="An evaluation of the training of a model",
                  **train_params):

    if "optimizer" not in train_params.keys():
        train_params["optimizer"] = optim.Adam(params=train_params["model"].parameters(), lr=0.001)

    train_losses, test_losses, accuracy, time = train(**train_params)
    stats = dict(train_losses=train_losses,
                 test_losses=test_losses,
                 accuracy=accuracy,
                 description=decr,
                 time=time)

    utils.save_pickle(stats, save_as)

def show_training_eval(path):
    data = utils.load_pickle(path)
    path = path.split("/")[-1]
    print(f"Evaluation of {path}\n")
    if "description" in data.keys():
        print(f"Description:\n\n{data['description']}\n")
    print(f"The final accuracy after training was {round(data['accuracy'], 1)}")
    print(f"Training took {round(data['time'], 1)} seconds\n")
    x = range(len(data["train_losses"]))
    plt.plot(x, data["train_losses"], label="Training Loss")
    plt.plot(x, data["test_losses"], label="Evaluation Loss")
    plt.legend()
    plt.show()

def compare_training_evals(*paths):
    for path in paths:
        data = utils.load_pickle(path)
        name = path.split("/")[-1]
        x = range(len(data["test_losses"]))
        acc = round(data['accuracy']*100, 1)
        time = round(data["time"], 1)
        label = f"{name} ({acc}% in {time}s"
        plt.plot(x, data["test_losses"], label=label)
    plt.legend()
    plt.show()


class TrainingStats:
    def __init__(self, train_losses, test_losses, accuracy, time, title="", descr="", model=None):
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.time = time
        self.title = title
        self.description = descr
        self.accuracy = accuracy
        self.model = model
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(train_losses=data["train_losses"],
                   test_losses=data["test_losses"],
                   time=data["time"],
                   accuracy=data["accuracy"],
                   descr=data["description"])

    @classmethod
    def from_tuple(cls, data: tuple):
        return cls(train_losses=data[0],
                   test_losses=data[1],
                   time=data[2],
                   accuracy=data[3])
 

    def summary(self):
        print(f"Title: {self.title}")
        print(self.description, "\n")
        print(f"Done training after {round(self.time, 1)} seconds")
        print(f"A final accuracy of {round(100 * self.accuracy, 1)}%\n\n")
        x0, x1 = range(len(self.train_losses)), range(len(self.test_losses))
        plt.plot(x0, self.train_losses, label="Training Loss")
        plt.plot(x1, self.test_losses, label="Validation Loss")
        plt.title(self.title)
        plt.legend()
        plt.show()

    def add(self, train_losses, test_losses, accuracy, time): 
        self.train_losses.extend(train_losses)
        self.test_losses.extend(test_losses)
        self.time += time
        self.accuracy = accuracy
