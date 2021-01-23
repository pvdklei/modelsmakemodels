import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torchvision as tv
import torch 
from torch import nn, optim
import utils
import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import functional as FT


def show_feature_repr(model, image, size=500):
    """Laat zien hoe een losse foto aangepast word door de convolutie lagen,
    van het model. Ook hier weer, na een paar lagen worden het te veel fotos om
    te laten zien, dus pakt hij de eerste "depth" lagen. (als een laag 16 output
    channels heeft worden het dus ook 16 fotos). Hiermee kan je wel prima zien
    dat sommige convolutie filters als "edge detectors" ofzo werken. De foto
    moet geloof ik deze shape hebben: (3, width, height)"""
    
    image = image.unsqueeze(0) # add extra batch dimension
    conv_layers = utils.get_all_conv_layers(model)
    hooks = utils.hook_output_all_layers(conv_layers)

    model(image)
    
    show_image_channels(image[0])
    for layer in conv_layers: 
        output = layer._output_hook
        nrow = int(size / output.shape[2]) 
        show_image_channels(output[0], figsize=(20, 40), nrow=nrow)
    
    for hook in hooks: 
        hook.remove()


def show_feature_projection(model, images, labels):
    conv_layers = utils.get_all_conv_layers(model)
    lin_layers = utils.get_all_lin_layers(model)
    layers = conv_layers + lin_layers
    hooks = utils.hook_output_all_layers(layers)
    model(images)
    for layer in layers:
        print("Looking at the features of the output of layer: ", layer)
        output = layer._output_hook
        proj = utils.project2d(output)
        plot_labeled(proj, labels)
    for hook in hooks: 
        hook.remove()

def show_image_batch(images: torch.Tensor, figsize=(25, 20), nrow=5):
    """input shape: (b, c, w, h)"""
    images = images.detach().cpu()
    grid = tv.utils.make_grid(images, scale_each=True, normalize=True, nrow=nrow).permute(1, 2, 0)
    plt.figure(figsize=figsize)
    plt.imshow(grid)
    plt.show()

def show_image_channels(image: torch.Tensor, figsize=(25, 25), nrow=5):
    """input shape = (c, w, h)"""
    image = image.detach().cpu()
    image = image.unsqueeze(1)        
    grid = tv.utils.make_grid(image, scale_each=True, normalize=True, nrow=nrow).permute(1, 2, 0)
    plt.figure(figsize=figsize)
    plt.imshow(grid)
    plt.show()
    

def show_side_by_side_loss(original, reconstructed): 
    """Shows two images side by side, and shows the
    MSE loss above it. Usefull for autoencoder validation"""
    batchsize = original.shape[0]
    original = torch.clip(original, 0, 1).detach().cpu()
    reconstructed = torch.clip(reconstructed, 0, 1).detach().cpu()
    for i in range(batchsize):
        fig, axs = plt.subplots(1, 2, figsize=(10, 20))
        fig.tight_layout()
        mseloss = torch.nn.functional.mse_loss(original[i], reconstructed[i])
        print("The MSE loss is: ", mseloss.item())
        axs[0].imshow(original[i].permute(1, 2, 0))
        axs[1].imshow(reconstructed[i].permute(1, 2, 0))
        axs[0].tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)
        axs[1].tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)
        plt.show()
        print("\n\n\n")

def show_image(image, figsize=(12, 12)):
    """Shows an image with
    shape = (c, w, h)"""
    if type(image) == torch.Tensor:
        image = image.cpu().detach().numpy()
    image = image.transpose(1, 2, 0)
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.show()

def image_that_feature_responds_to_most(model: nn.Module, 
                                        layern: int, 
                                        channel: int, 
                                        size: tuple, 
                                        lr: float=0.1, 
                                        iters: int=20,
                                        upscaling_steps=5,
                                        upscaling_factor=1.2):
    """Algoritme dat uitzoekt op welke input image een bepaalde filter/feature
    het meest zou reageren. In een model dat gezichten herkent zou een filter 
    kunnen reageren op ogen, dus dan zou de perfecte input image dus een foto 
    met heel veel ogen zijn. Deze perfecte image wordt gevonden door met noise 
    te beginnen, en door middel van gradient descent deze noise aan te passen 
    zodat de filter/feature in de gegeven layer het meest is opgelicht.

    De noise image wordt ook een paar keer vergroot, dus je moet het 'dense' 
    deel van je model er even afhalen.

    Voorbeeld: 
        vgg = torchvision.models.vgg16(pretrained=True).features
        image_that_feature_responds_to_most(vgg, 30, 23, (128, 128)) 

    """

    assert len(size) == 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    model.to(device)
    
    noise = torch.rand(1,3, *size, requires_grad=True)
    
    feature = None
    def feature_hook(_module, _input, output):
        nonlocal feature
        feature = output
    modules = [module for module in model.children() if type(module) != nn.Sequential]
    layer = modules[layern]
    print(f"This will evaluate the features of channel {channel} after layer {layern}: {layer}")
    hook = layer.register_forward_hook(feature_hook)
    
    losses = []
    x = list(range(iters * upscaling_steps))

    for i in range(upscaling_steps):
        
        # image processing for better results
        size = tuple(int(x * upscaling_factor) for x in size)
        noise = FT.gaussian_blur(noise, 3)
        noise = FT.resize(noise, size)

        # resetting because noise was cloned a couple of times
        noise = torch.autograd.Variable(noise.to(device), requires_grad=True)
        optimizer = optim.Adam(params=[noise], lr=lr)

        for j in range(iters):
            
            out = model(noise)
            optimizer.zero_grad()
            
            # the more the feature is highlighed, the better the input image
            loss = -feature[0, channel].mean()
            
            losses.append(loss)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            noise = utils.normalize(noise)

        print("Done with upscaling step: ", i)   

    plt.plot(x, losses)
    plt.show()
    show_image(noise[0])

    hook.remove()

def response_of_features_to_image(image: torch.Tensor,
                                  model: nn.Module,
                                  layer_n: int):
    """Show the mean activations of every filter in "layer_n" 
    caused by the input image and prints out the top 5. 

    Image shape: (c, w, h)"""

    assert len(image.shape) == 3
    image = image.unsqueeze(0)

    modules = [module for module in model.children() if module is not nn.Sequential()]
    layer = modules[layer_n]

    print(f"Accessing layer {layer_n}: {layer}")

    features = None
    def layer_hook(_module, _input, output):
        nonlocal features
        features = output
    layer.register_forward_hook(layer_hook)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    model = model.to(device)
    image = image.to(device)

    model(image)

    activations = features.mean(dim=(0, 2, 3)).detach()

    topk, topi = activations.topk(3)
    print(f"Largest activations are {topk.tolist()}, at {topi.tolist()}")

    activations = activations.cpu().numpy()
    x = range(len(activations))
    plt.bar(x, activations)
    plt.show()

def read_image(path):
    image = cv2.imread(path)
    image = FT.to_tensor(image)
    return image

    
def plot_labeled(data, labels):
    """Plots 2d data with the color representing the label"""
    data = pd.DataFrame(data, columns=["x", "y"])
    data["label"] = labels
    grouped = data.groupby("label")
    ax = plt.axes()
    for label, group in grouped:
        ax.plot(group.x, group.y, "o", label=label)
    plt.legend()
    plt.show()
   
def plot_labeled_3d(data, labels):
    """Plots 3d data with the color representing the label"""
    data = pd.DataFrame(data, columns="x y z".split())
    data["label"] = labels
    grouped = data.groupby("label")
    ax = plt.axes(projection="3d")
    for label, group in grouped:
        ax.scatter(group.x, group.y, group.z, "o", label=label)
    plt.legend()
    plt.show()
