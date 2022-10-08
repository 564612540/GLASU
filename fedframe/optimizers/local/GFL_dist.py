# from inspect import Parameter
# from curses import init_pair
import math
# from selectors import EpollSelector
import torch
import torch.optim as optim
import torch.nn as nn
# from torch.utils.data import DataLoader
import copy
import time
from fedframe.aggregator.gnn_cummunicator import ClienCommunicatorTorch
from fedframe.model.gnn import BaseClientLinear

class GFLClientD(object):
    def __init__(self, dataset, client_num, client_id, batchsize, device, layer_list):
        self.dataset = copy.deepcopy(dataset)
        self.client_num = client_num
        self.client_id = client_id
        self.batchsize = batchsize
        self.device = device
        self.layer_list = layer_list
        self.layer_num = len(layer_list)
        self.lr = None
        self.momentum = None
        self.optimizer = None
        self.comp_time = 0
        self.comm_time = 0

    def _update_lr(self, lr, momentum):
        if lr is not None:
            self.lr = lr
        elif self.lr is None:
            self.lr = 0.001
        if momentum is not None:
            self.momentum = momentum
        elif self.momentum is None:
            self.momentum = 0.9

    def _collect_params(self, client_model):
        parameter_set = set()
        client_model.to(self.device)
        client_model.train()
        parameter_set |= set(client_model.getParameters())
        parameter_list = []
        for layer in range(self.layer_num+1):
            parameter_list.append(set(client_model.getParameters(layer)))
        return parameter_set, parameter_list
    
    def _set_optimizer(self, parameter_set, global_iter, update):
        if update:
            if self.optimizer is None:
        # if global_iter<2 and self.optimizer is None:
                self.optimizer = optim.SGD(parameter_set, lr = self.lr, momentum = self.momentum, weight_decay=5e-5/self.client_num)
        # elif global_iter == 2:
        #     self.optimizer = optim.SGD(parameter_set, lr = self.lr/(1*math.sqrt(global_iter//4)+1), momentum=self.momentum)
            else:
                for g in self.optimizer.param_groups:
                    g['lr'] /= 4
    
    def train(self, client_model, client_iter, global_iter, lr = None, momentum = None, update = False):
        pass

    @torch.no_grad()
    def test(self, client_model, global_iter, communicator):
        client_model.to(self.device)
        client_model.eval()
        total = 0
        correct = 0
        losses = 0
        it = 0
        criterion = nn.CrossEntropyLoss()
        
        while True:
            # Prepare data batch node indices
            data_batch = [[] for _ in range(self.layer_num)]
            output= communicator.recv_idx()
            if output == [-1]:
                break
            # print("idx:%d"%len(output))
            for layer in range(self.layer_num-1,-1,-1):
                sampled_data = self.dataset.sample_id(output, self.batchsize, self.layer_list[layer])
                communicator.send_idx(sampled_data[0], self.client_id)
                data_batch[layer] = sampled_data
                output = communicator.recv_idx()
                data_batch[layer][0] = output
            # Forward propagation
            input_feature = self.dataset.sample_feature(data_batch[0][0]).to(self.device)
            for layer in range(self.layer_num):
                adj, out_index = self.dataset.sample_graph(index_list = data_batch[layer], device = self.device)
                # adj.to(self.device)
                # out_index.to(self.device)
                output = client_model(input_feature, layer, adj, out_index)
                communicator.send(output, self.client_id)
                input_feature = communicator.recv(-1)
                input_feature = input_feature.to(self.device)
            predict = client_model(input_feature, self.layer_num)
            label = self.dataset.sample_label(data_batch[-1][-1]).to(self.device)
            loss = criterion(predict, label)
            correct += predict.argmax(dim=-1).eq(label).sum().item()
            losses += loss.detach().to('cpu').numpy()
            total += len(label)
            # print("Testing, it:%d"%it)
            it += 1
        return correct/total, losses/it

    # @classmethod
    # def wrapper():
    @torch.no_grad()
    def send_result(self, acc, loss, communicator):
        # print("clients are here")
        data = torch.tensor([acc, loss, self.comp_time, self.comm_time])
        # print(data, data.size())
        communicator.send_idx(data)

    @torch.no_grad()
    def sync_model(self, model, communicator: ClienCommunicatorTorch):
        t_0 = time.process_time()
        comm_time = 0
        layer_dict = model.getFFN()
        ts = time.process_time()
        communicator.send_idx(layer_dict)
        layer_dict = communicator.recv_idx()
        comm_time += time.process_time() - ts
        for key, layer in layer_dict.items():
            layer = layer.to(self.device)
        # model.load_state_dict(layer_dict, strict = False)
        t_end = time.process_time()
        self.comm_time += comm_time
        self.comp_time += t_end-t_0 - comm_time
        return model

class GFLClientSv1(GFLClientD):
    def __init__(self, dataset, client_num, client_id, batchsize, device, layer_list):
        super(GFLClientSv1, self).__init__(dataset, client_num, client_id, batchsize, device, layer_list)

    def train(self, client_model, client_iter, global_iter, communicator, lr = None, momentum = None, update = False):
        # update learning rate and momentum
        t_0 = time.process_time()
        comm_time = 0
        self._update_lr(lr, momentum)
        parameter_set, parameter_list = self._collect_params(client_model)
        self._set_optimizer(parameter_set, global_iter, update)
        criterion = nn.CrossEntropyLoss()
        loss_sum = 0
        for it in range(client_iter):
            self.optimizer.zero_grad()
            # Prepare data batch node indices
            data_batch = [[] for _ in range(self.layer_num)]
            ts = time.process_time()
            output= communicator.recv_idx()
            comm_time += time.process_time() - ts
            for layer in range(self.layer_num-1,-1,-1):
                sampled_data = self.dataset.sample_id(output, self.batchsize, self.layer_list[layer])
                ts = time.process_time()
                communicator.send_idx(sampled_data[0], self.client_id)
                comm_time += time.process_time() - ts
                data_batch[layer] = sampled_data
                ts = time.process_time()
                output = communicator.recv_idx()
                comm_time += time.process_time() - ts
                data_batch[layer][0] = output
            # Start computing loss
            output_list = []
            input_list = []
            input_feature = self.dataset.sample_feature(data_batch[0][0]).to(self.device)
            for layer in range(self.layer_num):
                adj, out_index = self.dataset.sample_graph(index_list = data_batch[layer], device = self.device)
                # adj.to(self.device)
                # out_index.to(self.device)
                output = client_model(input_feature, layer, adj, out_index)
                ts = time.process_time()
                communicator.send(output, self.client_id)
                comm_time += time.process_time() - ts
                output_list.append(output)
                ts = time.process_time()
                input_feature = communicator.recv(-1).to(self.device)
                comm_time += time.process_time() - ts
                input_feature.requires_grad = True
                input_feature.retain_grad()
                input_list.append(input_feature)
            predict = client_model(input_feature, self.layer_num)
            label = self.dataset.sample_label(data_batch[-1][-1]).to(self.device)
            loss = criterion(predict, label)
            loss_sum += loss.cpu().item()
            # print("Iter:", it,"  Loss:", loss_sum/(it+1))
            input_feature.requires_grad = True
            input_feature.retain_grad()
            # start backward propagation
            inputs = parameter_list[-1]
            inputs |= set((input_list[-1],))
            loss.backward(inputs = inputs, retain_graph = True) # get gradient of classifier
            for layer in range(self.layer_num-1,-1,-1):
                ts = time.process_time()
                communicator.send(input_list[layer].grad, -1)
                comm_time += time.process_time() - ts
                inputs = parameter_list[layer]
                if layer > 0:
                    inputs |= set((input_list[layer-1],))
                ts = time.process_time()
                partial_grad = communicator.recv(self.client_id)
                comm_time += time.process_time() - ts
                output_list[layer].backward(gradient = partial_grad, inputs = inputs, retain_graph = True)
            self.optimizer.step()
        t_end = time.process_time()
        self.comm_time += comm_time
        self.comp_time += t_end-t_0 - comm_time
        return client_model

class GFLClientSv2(GFLClientD):
    def __init__(self, dataset, client_num, client_id, batchsize, device, layer_list, q):
        super(GFLClientSv2, self).__init__(dataset, client_num, client_id, batchsize, device, layer_list)
        self.q = q

    def _prepare_batch(self, communicator):
        comm_time = 0
        adj = [[] for _ in range(self.layer_num)] 
        out_index= [[] for _ in range(self.layer_num)]
        data_batch = [[] for _ in range(self.layer_num)]
        ts = time.process_time()
        output = communicator.recv_idx()
        comm_time += time.process_time()-ts
        for layer in range(self.layer_num-1,-1,-1):
            sampled_data = self.dataset.sample_id(output, self.batchsize, self.layer_list[layer])
            ts = time.process_time()
            communicator.send_idx(sampled_data[0], self.client_id)
            comm_time += time.process_time()-ts
            data_batch[layer] = sampled_data
            ts = time.process_time()
            output = communicator.recv_idx()
            comm_time += time.process_time()-ts
            data_batch[layer][0] = output
            adj[layer], out_index[layer] = self.dataset.sample_graph(index_list = data_batch[layer], device = self.device)
        return adj, out_index, data_batch, comm_time

    def train(self, client_model, client_iter, global_iter, communicator, lr = None, momentum = None, update = False):
        # update learning rate and momentum
        t_0 = time.process_time()
        comm_time = 0
        self._update_lr(lr, momentum)
        parameter_set, parameter_list = self._collect_params(client_model)
        self._set_optimizer(parameter_set, global_iter, update)
        criterion = nn.CrossEntropyLoss()
        # Prepare data batch node indices
        loss_sum = 0
        if global_iter == 0:
            self._set_optimizer(parameter_set, global_iter, update)
        for it in range(client_iter):
            adj_list, out_index_list, data_batch, sub_comm_time = self._prepare_batch(communicator)
            # print("finished batching, it: %d"%it)
            comm_time += sub_comm_time
            input_feature_0 = self.dataset.sample_feature(data_batch[0][0]).to(self.device)
            staled_input_list = [None for _ in range(self.layer_num)]
            for itr in range(self.q):
                self.optimizer.zero_grad()
                # Start computing loss
                output_list = []
                # input_list = []
                for layer in range(self.layer_num):
                    if layer == 0:
                        # print(input_feature_0.device, out_index_list[layer].device, adj_list[layer][0].device)
                        output = client_model(input_feature_0, layer, adj_list[layer], out_index_list[layer])
                    else:
                        input_feature = torch.add(staled_input_list[layer-1], output_list[layer-1], alpha= 1/self.client_num)
                        output = client_model(input_feature, layer, adj_list[layer], out_index_list[layer])
                    if staled_input_list[layer] is None:
                        ts = time.process_time()
                        communicator.send(output, self.client_id)
                        comm_time += time.process_time() - ts
                    output_list.append(output)
                    if staled_input_list[layer] is None:
                        ts = time.process_time()
                        staled_input = communicator.recv(-1).to(self.device)
                        comm_time += time.process_time() - ts
                        with torch.no_grad():
                            staled_input_list[layer] = torch.add(staled_input.detach(), output_list[layer], alpha= -1/self.client_num)
                input_feature = torch.add(staled_input_list[-1], output_list[-1], alpha= 1/self.client_num)
                predict = client_model(input_feature, self.layer_num)
                label = self.dataset.sample_label(data_batch[-1][-1]).to(self.device)
                loss = criterion(predict, label)
                loss.backward()
                self.optimizer.step()
                if itr == 0:
                    loss_sum += loss.cpu().item()
                    # print("inner iter:")
            # print("Iter:", it,"  Loss:", loss_sum/(it+1))
        t_end = time.process_time()
        self.comm_time += comm_time
        self.comp_time += t_end-t_0 - comm_time
        return client_model

def run(comm_iter: int, local_iter: int, print_freq: int, agent: GFLClientD, comm: ClienCommunicatorTorch, client_model: BaseClientLinear, args):
    for global_iter in range(comm_iter):
        if global_iter % 8 == 0:
            update = True
        else:
            update = False
        client_model = agent.train(client_model, local_iter, global_iter, comm, args.lr, args.momentum, update)
        # client_model = agent.sync_model(client_model, comm)
        if global_iter % print_freq == 0:
            # print("finished avg, start comp acc")
            local_acc, local_loss = agent.test(client_model, global_iter, comm)
            agent.send_result(local_acc, local_loss, comm)
    local_acc, local_loss = agent.test(client_model, global_iter, comm)
    agent.send_result(local_acc, local_loss, comm)