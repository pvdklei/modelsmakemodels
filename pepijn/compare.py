import train
import utils
import os
from pprint import pprint


def recur_dir(directory: str):
    data = []
    directory = directory.rstrip("/")
    for name in os.listdir(directory):
        if name.startswith("."):
            continue
        name = directory + "/" + name
        if os.path.isdir(name):
            substore = recur_dir(name)
            data.append(substore)
        elif os.path.isfile(name):
            training = train.Training.load(name)
            training.title = directory[len(homedirectory):]
            data.append(training)
    return data


homedirectory = "./comparisons/alexnet_cifar10_classification/"
data = recur_dir(homedirectory)
train.Training.compare(*data, max_epoch=17, metric="accuracy")
train.Training.compare(*data, max_epoch=17, metric="test_loss")
