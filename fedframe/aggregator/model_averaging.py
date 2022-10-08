import copy
import torch
from torch import nn
import numpy as np

def FedAvg(w, device = None):
    '''
    Input w: a list of parameter dicts need to be averaged
    Output w_avg: an averaged parameter dict
    '''
    # print(device)
    if not isinstance(w[0], dict):
        w = [wi.state_dict() for wi in w]
    w_avg = copy.deepcopy(w[0])
    if device is not None and next(iter(w[0].items()))[1].device != device:
        # print("changing device")
        for k in w_avg.keys():
            w_avg[k] = w_avg[k].to(device)
    else:
        device = next(iter(w_avg.items()))[1].device
    # print(next(iter(w_avg.items()))[1].device)
    for k in w_avg.keys():
        for i in range(1, len(w)):
            wt = w[i][k].to(device)
            w_avg[k] += wt
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg