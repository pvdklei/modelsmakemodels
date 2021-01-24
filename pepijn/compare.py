import train
import utils

base = "./comparisons/alexnet_cifar10_classification/"
filess = [["alexnet_transfer_body_frozen",
         "alexnet_transfer_body_frozen_1"],
         ["alexnet_transfer_final_conv_and_head_trained",
         "alexnet_transfer_final_conv_and_head_trained_1"],
         ["alexnet_transfer_fully_trained",
         "alexnet_transfer_fully_trained_1",
         "alexnet_transfer_fully_trained_2"]]
trainings = []
a = base + "alexnet_transfer_body_frozen_1"
training = train.Training.load(a)
training.title = "AlexNet transfer learning body frozen (CIFAR10)"
training.save(a)
for files in filess:
    sessions = []
    for f in files:
        path = base + f
        training = train.Training.load(path)
        sessions.append(training)
    trainings.append(sessions)

train.Training.compare(*trainings)
