import math
import torch
import torch.optim as optim
import torch.nn as nn
import copy
from fedframe.aggregator.model_averaging import FedAvg

class GFL_server_serial(object):
    def __init__(self, device, average = True):
        self.device = device
        self.average = average

    def update(self, client_model_list, global_iter):
        if self.average:
            with torch.no_grad():
                layer_blocks = []
                for client_model in client_model_list:
                    layer_blocks.append(client_model.getFFN())
                averaged_FFN = FedAvg(layer_blocks)
                for client_model in client_model_list:
                    client_model.load_state_dict(averaged_FFN, strict = False)
        return client_model_list
                