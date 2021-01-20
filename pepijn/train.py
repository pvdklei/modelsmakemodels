

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


class Training:
    def __init__(self, 
                 autotrain=False,
                 train_losses=[], 
                 test_losses=[], 
                 accuracy=0.0, 
                 time=0.0, 
                 title="", 
                 descr=""):
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.time = time
        self.title = title
        self.description = descr
        self.accuracy = accuracy
        self.autotrain = autotrain
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(train_losses=data["train_losses"],
                   test_losses=data["test_losses"],
                   time=data["time"],
                   accuracy=data["accuracy"],
                   descr=data["description"])

    @classmethod
    def load(cls, path):
        data = utils.load_pickle(path)
        if type(data) == dict:
            return cls.from_dict(data)
        elif type(data) == cls:
            return data
        else:
            raise Exception("Not a valid training stats format found in pickle")

    def save(self, path):
        utils.save_pickle(self, path)
 
    @staticmethod
    def compare(*trainings):
        for training in trainings:
            x = range(len(training.test_losses))
            acc = round(training.accuracy*100, 1)
            time = round(training.time, 1)
            label = f"{training.title} ({acc}% in {time}s"
            plt.plot(x, training.testlosses, label=label)
        plt.legend()
        plt.show()
    

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

    def train(self, 
              model, 
              trainloader, 
              testloader, 
              optimizer=None,
              criterion=nn.CrossEntropyLoss(), 
              epochs=5, 
              loaders=None):
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    
        if optimizer == None:
            optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    
    
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
            print(f"Epoch {epoch}, Validation loss: {round(testloss, 3)}")
            test_losses.append(testloss)
    
    
            # training
            trainloss = []
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                out = model(images)
                if self.autotrain:
                    loss = criterion(out, images)
                else:
                    loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                trainloss.append(loss.item())
            trainloss = np.mean(trainloss)
            print(f"Epoch {epoch}, Training loss: {round(trainloss, 3)}")
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
        self.add(train_losses, test_losses, totacc, end_time - start_time)










