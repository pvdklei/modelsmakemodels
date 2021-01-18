import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torchvision as tv
import torch 
from torch import nn, optim
import utils
import cv2
from sklearn.manifold import TSNE
import sklearn.decomposition as decomp
import numpy as np
import pandas as pd
from torchvision.transforms import functional as FT

def show_filters(model, depth=2, tag="conv"):
    """Laat zien hoe alle filters in een convolutional neural network eruit zien.
    Je kan met "depth" aangeven tot hoeveel convolutielagen je wil kijken, aangezien
    er na de tweede laag al te veel filters zijn om te bekijken (hoeveel filters 
    er in zitten ligt aan het aantal "output_channels"). Hij doorzoekt de lagen
    op de "tag". Dus als je een lineaire laag "fc1" noemt skipt hij die."""
    count = 0
    for key, filters in model.state_dict().items():
        if tag not in key or "weight" not in key:
            continue
            
        if count >= depth:
            break
        count += 1
        
        fig, axs = plt.subplots(filters.shape[0], filters.shape[1], figsize=(15,15))
        fig.suptitle(key)

        f = normalize(filters)
        f = f.cpu().numpy()

        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                ax = axs[i,j]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(f[i,j,:,:])
        plt.plot()

def show_feature_repr(model, image, depth=2):
    """Laat zien hoe een losse foto aangepast word door de convolutie lagen,
    van het model. Ook hier weer, na een paar lagen worden het te veel fotos om
    te laten zien, dus pakt hij de eerste "depth" lagen. (als een laag 16 output
    channels heeft worden het dus ook 16 fotos). Hiermee kan je wel prima zien
    dat sommige convolutie filters als "edge detectors" ofzo werken. De foto
    moet geloof ik deze shape hebben: (3, width, height)"""
    image = image.view(1, *image.shape) # add extra batch dimension
    conv_layers = [child for i, child in enumerate(model.children()) if type(child) == nn.Conv2d and i <= depth]
    reprs = [image]
    for layer in conv_layers:
        reprs.append(layer(reprs[-1]))
    for i, repr in enumerate(reprs):
        repr = normalize(repr)
        fig, axs = plt.subplots(repr.shape[1])
        title = "Original" if i == 0 else f"After {conv_layers[i-1]}"
        fig.suptitle(title)
        for chan in range(repr.shape[1]):
            img = repr[0,chan,:,:].detach().numpy()
            ax = axs[chan]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(img)
        plt.show()

def show_image_channels(image: torch.Tensor, figsize=(25, 25)):
    """input shape = (c, w, h)"""
    image = image.detach().cpu()
    image = image.unsqueeze(1)        
    grid = tv.utils.make_grid(image, scale_each=True, normalize=True, nrow=5).permute(1, 2, 0)
    plt.figure(figsize=figsize)
    plt.imshow(grid)
    plt.show()
    
def tSNE(image, d=2):
    if type(image) == np.ndarray:
        image = torch.from_numpy(image)
    image = image.detach().cpu()
    projection = TSNE(n_components=d).fit_transform(image.view(image.shape[0], -1))  
    return projection

def PCA(image, d=50):
    if type(image) == np.ndarray:
        image = torch.from_numpy(image)
    image = image.detach().cpu()
    projection = decomp.PCA(n_components=d).fit_transform(image.view(image.shape[0], -1))
    return projection

def project2d(image, dpca=50):
    """PCA to 50 comps followed by t-SNE """
    image = PCA(image, d=dpca)
    return tSNE(image)

def plot_labeled(data, labels):
    data = pd.DataFrame(data, columns=["x", "y"])
    data["label"] = labels
    grouped = data.groupby("label")
    ax = plt.axes()
    for label, group in grouped:
        ax.plot(group.x, group.y, "o", label=label)
    plt.legend()
    plt.show()
   
def plot_labeled_3d(data, labels):
    data = pd.DataFrame(data, columns="x y z".split())
    data["label"] = labels
    grouped = data.groupby("label")
    ax = plt.axes(projection="3d")
    for label, group in grouped:
        ax.scatter(group.x, group.y, group.z, "o", label=label)
    plt.legend()
    plt.show()

def show_image(image, figsize=(12, 12)):
    """shape = (c, w, h)"""
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
                                        upscaling_factor=1.1):
    """Algoritme dat uitzoekt op welke input image een bepaalde filter/feature
    het meest zou reageren. In een model dat gezichten herkent zou een filter 
    kunnen reageren op ogen, dus dan zou de perfecte input image dus een foto 
    met heel veel ogen zijn. Deze perfecte image wordt gevonden door met noise 
    te beginnen, en door middel van gradient descent deze noise aan te passen 
    zodat de filter/feature in de gegeven layer het meest is opgelicht.

    De noise image wordt ook een paar keer vergroot, dus je moet het 'dense' 
    deel van je model er even afhalen."""

    assert len(size) == 2
    
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

    print(noise[0,0,0,0])

    for i in range(upscaling_steps):
        
        # resetting because noise was cloned a couple of times
        noise = torch.autograd.Variable(noise, requires_grad=True)
        optimizer = optim.Adam(params=[noise], lr=lr)

        for j  in range(iters):
            
            out = model(noise)
            optimizer.zero_grad()
            
            # the more the feature is highlighed, the better the input image
            loss = -feature[0, channel].mean()
            
            losses.append(loss)
            loss.backward()
            optimizer.step()

        size = tuple(int(x * upscaling_factor) for x in size)
        noise = FT.resize(noise, size)

        # nomalizing noise image
        #mean = noise.mean()
        #std = noise.std()
        #noise = (noise - mean) / std 
        #noise = noise / 2 + 0.5
        noise = utils.normalize(noise)

    mean = noise.mean()
    std = noise.std()
    noise = (noise - mean) / std 
    noise = noise / 2 + 0.5
    noise = utils.normalize(noise)

    print(noise[0,0,0,0])

    plt.plot(x, losses)
    plt.show()
    show_image(noise[0])

    hook.remove()
