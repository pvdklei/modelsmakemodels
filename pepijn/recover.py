import os 
import utils
import train

directory = "./comparisons/alexnet_cifar10_classification"
target_directory = "./comparisons/alexnet_cifar10_classification_recovered"

def cutoff(t):
    print("Training")
    for i in range(len(t.test_losses)-1, -1, -1):
        testloss = t.test_losses[i]
        if testloss > 2:
            break
    t.test_losses = t.test_losses[i:]
    t.accuracies = t.accuracies[i:]
    t.train_losses = t.train_losses[i:]
    return t

for f in os.listdir(directory):
    filename = directory + "/" + f.strip(".pkl")
    target_filename = target_directory + "/" + f.strip(".pkl")
    training = train.Training.load(filename)
    training.summary()
    training.time = -1
    training = cutoff(training)
    training.save(target_filename)

