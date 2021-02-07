import pickle
import torch
from torch import nn
import numpy as np
from sklearn.manifold import TSNE
import sklearn.decomposition as decomp

def save_pickle(obj, filename):
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)

def _conv_out_shape(og_size, filter_size, stride, padding):
    return int((og_size - filter_size + 2 * padding) / stride) + 1

def conv_out_shape(og_size, filter_size, stride=1, padding=0):
    if type(og_size) == tuple:
        og_width = og_size[0]
        og_height = og_size[1]
        return _conv_out_shape(og_width, filter_size, stride, padding), _conv_out_shape(og_height, filter_size, stride, padding)
    else:
        out = _conv_out_shape(og_size, filter_size, stride, padding)
        return out, out

def _convt_out_shape(og_size, filter_size, stride, padding):
    return int((og_size - 1) * stride - 2 * padding + filter_size)

def convt_out_shape(og_size, filter_size, stride=1, padding=0):
    if type(og_size) == tuple:
        og_width = og_size[0]
        og_height = og_size[1]
        return _convt_out_shape(og_width, filter_size, stride, padding), _conv_out_shape(og_height, filter_size, stride, padding)
    else:
        out = _convt_out_shape(og_size, filter_size, stride, padding)
        return out, out

def flatten_size(x, chans):
    return chans * x[0] * x[1]

def intlerp(a, one, other):
    return int(one + a * (other - one))

def normalize(x):
    max, min = x.max(), x.min()
    return (x - min) / (max - min)

def hook_output(layer):
    def feature_hook(module, _input, output):
        setattr(module, "_output_hook", output)
    hook = layer.register_forward_hook(feature_hook)
    return hook

def hook_output_all_layers(layers):
    hooks = []
    for layer in layers:
        def feature_hook(module, _input, output):
            setattr(module, "_output_hook", output)
        hooks.append(layer.register_forward_hook(feature_hook))
    return hooks

def get_all_conv_layers(model):
    conv_layers = [child for child in model.modules() \
        if (type(child) == nn.Conv2d or type(child) == nn.ConvTranspose2d)]
    return conv_layers
  
def get_all_lin_layers(model):
    lin_layers = [child for child in model.modules() \
        if type(child) == nn.Linear]
    return lin_layers

def tSNE(images, d=2):
    if type(images) == np.ndarray:
        images = torch.from_numpy(images)
    images = images.detach().cpu()
    projection = TSNE(n_components=d).fit_transform(images.view(images.shape[0], -1))  
    return projection

def PCA(images, d=50):
    if type(images) == np.ndarray:
        images = torch.from_numpy(images)
    images = images.detach().cpu()
    projection = decomp.PCA(n_components=d).fit_transform(images.view(images.shape[0], -1))
    return projection

def project2d(images, dpca=50):
    """PCA to 50 comps followed by t-SNE """
    shape = images.shape
    lowest = shape[0] if shape[0] < shape[1] else shape[1]
    comps = dpca if dpca < lowest else lowest-1
    images = PCA(images, d=comps)
    return tSNE(images)

def tensor_size(tensor): 
    x = 1
    for i in tensor.shape:
        x *= i
    return x





