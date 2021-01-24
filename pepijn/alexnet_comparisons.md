# AlexNet based CIFAR10 classifier: transfer, encode or neither?

In this experiment we trained CNN's to classify the CIFAR10 dataset. These neural nets all had the same architecture, namely the body (feature extractor) of the AlexNet architecture, and a simple head (classifier) consisting of two linear layers (2304 -> 500 -> 10) divided by a rectified linear unit. 

What differed mainly between these CNN's, was the initialization of the convolutional weigths. We specify three groups:

1. Weights were randomly initialized as a control group.
2. Weigths were transfered from a (AlexNet based) classifier that was trained on the ImageNet dataset.
3. Weigths were copied from a (AlexNet based) autoencoder that was trained on the CIFAR10 dataset.

They were also trained similarly with an Adam optimizer (lr=0.001) and using the cross-entropy loss function. Training differed slightly in that the image preprocessing was altered a bit, when a CNN had difficulty generalizing. Namely, in some occasions, a random horizontal flip, random vertical flip, and/or a random rotation of 10-20 degrees was applied to only the images in the trainingset, every epoch. 

Another way we altered the way we trained the nets was by freezing (parts) of the weigths. Again, we subdivide into three groups:

1. All weights were (re)trained. No weights were frozen.
2. Only the head (i.e classifier or linear units) was trained. The weights of the convolutional layers were frozen.
3. The head and the final convolutional layer was (re)trained. The weigths of the shallower convolutional layers were frozen.

To validate that is no outliers were found, experiments were repeated a couple of times for the groups and subgroups.
Below the performance of the CNN's are visualized, by providing showing the validation loss during every epoch of training. The time it took to train the CNN's is not shown, as data was discrupted, and only the losses and accuracies could still be saved. But from memory, it was remembered that (logically) the weight-initialization didn't alter the time-per-epoch. Just the amount of weights that were frozen during training had influence on the training time. Namely, group 1 (no weights frozen) took about twice the amount of time-per-epoch than groups 2 (convolutional weights frozen) and 3 (shallower convolutional weights frozen). 

### SHOW PLOT

What can be seen is that of the first three groups. Group 2 (transfer learning) seems to achieve lowest test-losses in less epochs, for all training types. This confirms that transfer learning can be a valuable choice, even if there is not lack of training data. The CNN's that copied the autoencoder's weights (group 3) did less good. Only when fully retraining the weights, it could achieve similar results to transfer learning. This could mean that the features extractors of autoencoders are of less value (in a classification setting) than those that were intended to do classifaction, even if the latter are trained on a different dataset. 


## Are autoencoder features actually worse than those of transfered classifiers?

Even though transfered feature extractors seem to outperform those of autoencoders, there are a few things that could have hindered the autoencoder from achieving his full potential. (1) First of all it took al long time to get an autoencoder working well, and very little optimizations could be done. (2) Also, for control purposes, the AlexNet architecture was used to train the encoder. This architecture was not originally intended to be used in autoencoders, and might therefore be a suboptimal solution. (3) To finalize, the pretrained AlexNet model has an accuracy of ???% and was trained for over ??? days on a dataset (ImageNet) that is a lot larger than CIFAR10. It can hardly be imaged that, if you could compare autoencoders and classifiers, a similar performance was achieved in this autoencoder.

Then, what else do we know?

On the internet, similar research can be found. Namely....

## Why partially training works best: feature introspection.





