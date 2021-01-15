import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torchvision as tv
import torch 
import cv2
from sklearn.manifold import TSNE
import numpy as np

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
    
def tSNE(image):
    tsne = TSNE().fit_transform(image.view(image.shape[0], -1))
    print(tsne)
    
    

