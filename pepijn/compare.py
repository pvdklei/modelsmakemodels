import train
import utils
import os
from pprint import pprint


def recur_dir(directory: str):
    store = []
    directory = directory.rstrip("/")
    for name in os.listdir(directory):
        if name.startswith("."):
            continue
        name = directory + "/" + name
        if os.path.isdir(name):
            substore = recur_dir(name)
            store.append(substore)
        elif os.path.isfile(name):
            training = train.Training.load(name)
            store.append(training)
    return store


directory = "./comparisons/alexnet_cifar10_classification/"
store = recur_dir(directory)
for i in store:
    pprint(i)
    print("\n\n\n")
train.Training.compare(*store, max_epoch=20)
