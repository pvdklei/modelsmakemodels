# import libraries and packages

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
from matplotlib.lines import Line2D
from collections import OrderedDict

# create training class

class Training:
    def __init__(self, 
                 autotrain=False,
                 train_losses=[], 
                 test_losses=[], 
                 accuracies=[], 
                 time=0.0, 
                 title="", 
                 descr=""):
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.time = time
        self.title = title
        self.description = descr
        self.accuracies = accuracies
        self.autotrain = autotrain

    @classmethod
    def from_dict(cls, data: dict):
        if "accuracy" in data.keys():
            acc = data.pop("accuracy")
            data["accuracies"] = [acc]
        return cls(**data)
    
    def to_dict(self) -> dict:
        return dict(test_losses=self.test_losses,
                    train_losses=self.train_losses,
                    time=self.time,
                    accuracies=self.accuracies,
                    title=self.title,
                    descr=self.description)

    @classmethod
    def load(cls, path):
        """Loads a saved Training"""
        if path.endswith(".pkl"):
            path = path[:-4]
        data = utils.load_pickle(path)
        if type(data) == dict:
            return cls.from_dict(data)
        elif type(data) == cls:
            return data
        else:
            raise Exception("Not a valid training stats format found in pickle")

    def save(self, path):
        """Saved a Training instance, so it can be loaded later"""
        utils.save_pickle(self.to_dict(), path)
 
    @staticmethod
    def _plot_training(training, color=None, stroke="-"):
        x = range(len(training.test_losses))
        acc = round(max(training.accuracies)*100, 1)
        time = round(training.time, 1)
        label = f"{training.title}"
        plt.plot(x, training.test_losses, label=label, color=color, linestyle=stroke)

    @classmethod
    def compare(cls, *trainings, max_epoch=10):
        """Similar to summary, but lets you compare multiple training
        sessions based on their test-loss through visuals.

        If the trainings are Training instances, they will all be 
        plotted with a different color. 

        If the trainings are lists of Training, every list will have 
        the same color in the plot.

        If the trainings are lists of lists of Trainings, the first
        list division will give different colors, and the second
        a different type of stroke.

        Example: 
            Training.compare(t1, t2, t3, ...) # all different colors
            Training.compare([t1, t2], [t3, t4], ...) # t1 and t2 have same color 
            Training.compare([[t1, t2], [t1, t2]], ...) # all same color, t1 and t2 same stroke
        
        """

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # gets all default colors of plt
        linestyles = ['-', '--', '-.', ':']
        lines = [] # for the legend

        plt.figure(figsize=(10, 7))
        
        for i, el in enumerate(trainings):
            color = colors[i]
            if type(el) == cls:
                cls._plot_training(el, color=color)
            elif type(el) == list:
                for j, el_ in enumerate(el):
                    if type(el_) == cls:
                        cls._plot_training(el_, color=color)
                    elif type(el_) == list:
                        stroke = linestyles[j]
                        for el__ in el_:
                            if type(el__) == cls:
                                cls._plot_training(el__, color=color, stroke=stroke)
                            elif type(el__) == list: 
                                raise Exception("List in list in list is not allowed")
                            else:
                                raise Exception("Pass a Training instance")
                    else: 
                        raise Exception("You must either provide a Training instance or a list of them")
            else: 
                raise Exception("You must either provide a Training instance or a list of them")

        plt.xlim(0, max_epoch)
        plt.ylabel("Test Loss")
        plt.xlabel("Epoch")

        # remove duplicate legend names 
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="upper right")
        
        plt.show()
    

    def summary(self):
        """Shows a summary of the training session, including loss-plots"""

        print(f"Title: {self.title}")
        print(self.description, "\n")
        print(f"Done training after {round(self.time, 1)} seconds")
        if len(self.accuracies) > 0:
            print(f"A final accuracy of {round(100 * self.accuracies[-1], 1)}%\n\n")
        x0, x1 = range(len(self.train_losses)), range(len(self.test_losses))
        plt.plot(x0, self.train_losses, label="Training Loss")
        plt.plot(x1, self.test_losses, label="Validation Loss")
        plt.title(self.title)
        plt.legend()
        plt.show()

    def train(self, 
              model: nn.Module, 
              loaders, 
              optimizer: optim.Optimizer=None,
              criterion=nn.CrossEntropyLoss(), 
              epochs=5, 
              reload_=False):
        """Trains a given model based on the provided settings and saves
        statistics. Also provides nice metrics during training.

        Note: 
            'loader' must be a function returning two torchvision.DataLoader
            instances, with the first one being the trainloader, and the 
            second being the validation loader. If 'reload_' is set
            to True, the loaders are remade by calling the function. This
            can be usefull for redoing random transformation to stimulate
            generalization."""
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    
        if optimizer == None:
            optimizer = optim.Adam(params=model.parameters(), lr=0.001)
        trainloader, testloader = loaders()
    
        print("Training on: ", device)
    
    
        for epoch in range(epochs):

            start_time = time.time()

            print(f"New epoch: {epoch}")
    
            # validation
            testloss = []
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    out = model(images)
                    if self.autotrain:
                        loss = criterion(out, images)
                    else:
                        loss = criterion(out, labels)
                    testloss.append(loss.item())
            testloss = np.mean(testloss)
            print(f"Validation loss: {round(testloss, 3)}")
    
            # accuracy
            if not self.autotrain:
                accuracy = []
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    with torch.no_grad():
                        out = model(images)
                        topv, topi = torch.topk(out, 1, dim=1)
                        labels.resize_(*topi.shape)
                        eq = topi == labels
                        acc = torch.mean(eq.type(torch.FloatTensor))
                        accuracy.append(acc.item())
                accuracy = np.mean(accuracy)
                print(f"The accuracy is: {round(accuracy * 100, 1)}%")
    
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
            print(f"Training loss: {round(trainloss, 3)}")

            # store epoch data
            self.train_losses.append(trainloss)
            self.test_losses.append(testloss)
            if not self.autotrain:
                self.accuracies.append(accuracy)

            # save model if it's the best 
            if min(self.test_losses) > testloss:
                utils.save_pickle(model, "model")     

            end_time = time.time()
            self.time += end_time - start_time 

            # reload image loaders, so the transforms are done again 
            if reload_:
                trainloader, testloader = loaders()
    def denoise_train(self, 
              model, 
              loaders, 
              optimizer=None,
              criterion=nn.CrossEntropyLoss(), noise_type='gaussian',
              epochs=5, 
              reload_=False):
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    
        if optimizer == None:
            optimizer = optim.Adam(params=model.parameters(), lr=0.001)
        trainloader, testloader = loaders()
    
        print("Training on: ", device)
    
        start_time = time.time()
    
        for epoch in range(epochs):

            print(f"New epoch: {epoch}")
    
            # validation
            testloss = []
            for images, labels in testloader:
                noise_img = random_noise(images, mode=noise_type, mean=0, var=0.05, clip=True)
                noise_img = torch.Tensor(noise_img)
                images, labels, noise_img = images.to(device), labels.to(device), noise_img.to(device)
                with torch.no_grad():
                    out = model(noise_img)
                    if self.autotrain:
                        loss = criterion(out, images)
                    else:
                        loss = criterion(out, labels)
                    testloss.append(loss.item())
            testloss = np.mean(testloss)
            print(f"Validation loss: {round(testloss, 3)}")
    
            # final accuracy
            if not self.autotrain:
                accuracy = []
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    with torch.no_grad():
                        out = model(images)
                        topv, topi = torch.topk(out, 1, dim=1)
                        labels.resize_(*topi.shape)
                        eq = topi == labels
                        acc = torch.mean(eq.type(torch.FloatTensor))
                        accuracy.append(acc.item())
                accuracy = np.mean(accuracy)
                print(f"The accuracy is: {round(accuracy * 100, 1)}%")
    
            # training
            trainloss = []
            for images, labels in trainloader:
                noise_img = random_noise(images, mode=noise_type, mean=0, var=0.05, clip=True)
                noise_img = torch.Tensor(noise_img)
                images, labels, noise_img = images.to(device), labels.to(device), noise_img.to(device)
                optimizer.zero_grad()
                out = model(noise_img)
                if self.autotrain:
                    loss = criterion(out, images)
                else:
                    loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                trainloss.append(loss.item())
            trainloss = np.mean(trainloss)
            print(f"Training loss: {round(trainloss, 3)}")

            self.train_losses.append(trainloss)
            self.test_losses.append(testloss)
            if not self.autotrain:
                self.accuracies.append(accuracy)
    
            if reload_:
                trainloader, testloader = loaders()
    
        end_time = time.time()
        self.time += end_time - start_time 







