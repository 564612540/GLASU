import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import glob

class Model(nn.Module):
    def __init__(self, name):
        super(Model, self).__init__()
        self.name = name
    def save(self, path, epoch=0):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)
        torch.save(self.state_dict(), 
                os.path.join(complete_path, 
                    "model-{}.pth".format(str(epoch).zfill(5))))
    def save_results(self, path, data):
        raise NotImplementedError("Model subclass must implement this method.")
    def load(self, path, modelfile=None):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            raise IOError("{} directory does not exist in {}".format(self.name, path))
        if modelfile is None:
            model_files = glob.glob(complete_path+"/*")
            mf = max(model_files)
        else:
            mf = os.path.join(complete_path, modelfile)
        self.load_state_dict(torch.load(mf))

def hybrid_diff(model_1, global_model, matching_pattern, device):
    loss = 0
    state_dict=model_1.state_dict()
    state_dict_g=global_model.state_dict()
    for param_id, ((k_c,v_c),(k_g,v_g)) in enumerate(zip(state_dict.items(),state_dict_g.items())):
        layer_pattern_this = matching_pattern[param_id//2]
        layer_pattern_next = matching_pattern[param_id//2+1]
        if param_id%2 == 0:
            g_weight = v_g[layer_pattern_next,:]
            g_weight = g_weight[:,layer_pattern_this]
            loss += torch.norm(v_c-g_weight)
        else:
            g_bias = v_g[layer_pattern_next]
            loss += torch.norm(v_c - g_bias)
    return loss

def layer_matching_diff(model_1, model_2, matching_pattern, device):
    loss = 0
    model_1.to(device)
    model_2.to(device)
    if hasattr(model_1, 'encoder'):
        for layer_1, layer_2 in enumerate(matching_pattern['encoder']):
            loss += model_diff(model_1.encoder[layer_1], model_2.encoder[layer_2])
    if hasattr(model_1, 'decoder'):
        for layer_1, layer_2 in enumerate(matching_pattern['decoder']):
            loss += model_diff(model_1.decoder[layer_1], model_2.decoder[layer_2])
    return loss

def model_diff(model_1, model_2, device = None):
    loss = 0
    if device is not None:
        model_1.to(device)
        model_2.to(device)
    state_dict_1=model_1.state_dict()
    state_dict_2=model_2.state_dict()
    for param_id, ((k_c,v_c),(k_g,v_g)) in enumerate(zip(state_dict_1.items(),state_dict_2.items())):
        loss += torch.norm(v_c-v_g)
    return loss