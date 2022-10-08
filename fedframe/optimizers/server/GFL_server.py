import math
import torch
import torch.optim as optim
import torch.nn as nn
import copy
from fedframe.aggregator.model_averaging import FedAvg
from fedframe.aggregator.gnn_cummunicator import ServerCommunicatorTorch

class GFLServerD(object):
    def __init__(self, dataset_train, dataset_test, client_num, batchsize ,layer_num, device):
        self.dataset_train = copy.deepcopy(dataset_train)
        self.dataset_test = copy.deepcopy(dataset_test)
        self.client_num = client_num
        self.batchsize = batchsize
        self.device = device
        self.layer_num = layer_num

    def train(self):
        pass

    def test(self):
        pass

    def sync_model(self):
        pass

    # def wrapper(self):
    #     pass

class GFLServerAvg(GFLServerD):
    def __init__(self, dataset_train, dataset_test, client_num, batchsize ,layer_num, feature_size, device):
        super().__init__(dataset_train, dataset_test, client_num, batchsize ,layer_num, device)
        self.feature_size = feature_size

    def train(self, client_iter, global_iter, v1: bool, communicator: ServerCommunicatorTorch):
        for it in range(client_iter):
            # collect batch index
            index = self.dataset_train.sample_batch(self.batchsize)
            communicator.send_idx(index)
            batch_size = [None for _ in range(self.layer_num)]
            for layer in range(self.layer_num-1, -1, -1):
                batch_size[layer] = len(index)
                index_list = communicator.recv_idx()
                index = list(set.union(*[set(idx) for idx in index_list]))
                communicator.send_idx(index)
            # forward propagation
            for layer in range(self.layer_num):
                feature_list = communicator.recv(torch.Size([batch_size[layer], self.feature_size]))
                feature = torch.mean(torch.stack(feature_list), dim = 0)
                communicator.send(feature)
            # backward propagation
            if v1:
                for layer in range(self.layer_num-1,-1,-1):
                    grad_list = communicator.recv(torch.Size([batch_size[layer], self.feature_size]))
                    grad = torch.mean(torch.stack(grad_list), dim = 0)
                    communicator.send(grad)
            # print("finish communication: %d: %d"%(global_iter, it))

    @torch.no_grad()
    def test(self, global_iter, communicator: ServerCommunicatorTorch):
        while True:
            # Prepare data batch node indices
            index = self.dataset_test.sample_batch(self.batchsize)
            if index == -1:
                index = [-1]
            communicator.send_idx(index)
            if index == [-1]:
                break
            # print("index:%d"%len(index))
            batch_size = [None for _ in range(self.layer_num)]
            for layer in range(self.layer_num-1, -1, -1):
                batch_size[layer] = len(index)
                index_list = communicator.recv_idx()
                index = list(set.union(*[set(idx) for idx in index_list]))
                communicator.send_idx(index)
            # forward propagation
            for layer in range(self.layer_num):
                feature_list = communicator.recv(torch.Size([batch_size[layer], self.feature_size]))
                feature = torch.mean(torch.stack(feature_list), dim = 0)
                communicator.send(feature)

    @torch.no_grad()
    def recv_result(self, communicator):
        result_list = communicator.recv_idx()
        # print("server finished sending")
        result = torch.mean(torch.stack(result_list), dim = 0)
        acc = result[0].item()
        loss = result[1].item()
        comp_time = result[2].item()
        comm_time = result[3].item()
        return acc, loss, comp_time, comm_time

    @torch.no_grad()
    def sync_model(self, communicator: ServerCommunicatorTorch):
        layer_list = communicator.recv_idx()
        avg_layer = FedAvg(layer_list, self.device)
        communicator.send_idx(avg_layer)

def run(comm_iter: int, local_iter: int, print_freq: int, agent: GFLServerD, comm: ServerCommunicatorTorch, log_file, v1: bool):
    for global_iter in range(comm_iter):
        agent.train(local_iter, global_iter, v1, comm)
        # agent.sync_model(comm)
        if global_iter % print_freq == 0:
            agent.test(global_iter, comm)
            # print("Server is here")
            server_acc, server_loss, comp_time, comm_time = agent.recv_result(comm)
            print("Comm time: %0.4f, comp time: %0.4f"%(comm_time, comp_time))
            log_file.update([global_iter, comp_time+comm_time, comp_time, comm_time], [server_acc,server_loss])
    agent.test(global_iter, comm)
    server_acc, server_loss, comp_time, comm_time= agent.recv_result(comm)
    log_file.update([global_iter+1, comp_time+comm_time, comp_time, comm_time], [server_acc,server_loss])