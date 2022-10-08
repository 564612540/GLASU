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

class GFLClientS(object):
    def __init__(self, dataset_list, client_num, batchsize, device, layer_list):
        assert (len(dataset_list) == client_num)
        self.dataset_list = copy.deepcopy(dataset_list)
        self.client_num = client_num
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

    def _collect_params(self, client_model_list):
        parameter_set = set()
        for client_model in client_model_list:
            client_model.to(self.device)
            client_model.train()
            parameter_set |= set(client_model.getParameters())
        parameter_list = []
        for layer in range(self.layer_num+1):
            layer_parameter_set = set()
            for client_model in client_model_list:
                layer_parameter_set |= set(client_model.getParameters(layer))
            parameter_list.append(layer_parameter_set)
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
    
    def train(self, client_model_list, client_iter, global_iter, lr = None, momentum = None, update = False):
        pass

    def test(client_model_list, global_iter):
        pass

class GFLClientSv1(GFLClientS):
    def __init__(self, dataset_list, client_num, batchsize, device, layer_list):
        super(GFLClientSv1, self).__init__(dataset_list, client_num, batchsize, device, layer_list)

    def train(self, client_model_list, client_iter, global_iter, communicator, lr = None, momentum = None, update = False):
        assert(len(client_model_list) == self.client_num)
        # update learning rate and momentum
        self._update_lr(lr, momentum)
        parameter_set, parameter_list = self._collect_params(client_model_list)
        self._set_optimizer(parameter_set, global_iter, update)
        criterion = nn.CrossEntropyLoss()
        
        loss_sum = 0
        for it in range(client_iter):
            self.optimizer.zero_grad()
            # Prepare data batch node indices
            data_batch = [[[] for _ in range(self.layer_num)] for _ in range(self.client_num)]
            output= communicator.recv_idx()
            # print(output)
            for layer in range(self.layer_num-1,-1,-1):
                for client_id in range(self.client_num):
                    sampled_data = self.dataset_list[client_id].sample_id(output, self.batchsize, self.layer_list[layer])
                    communicator.send_idx(sampled_data[0], client_id)
                    data_batch[client_id][layer] = sampled_data
                output = communicator.recv_idx()
                for client_batch in data_batch:
                    client_batch[layer][0] = output
            # Start computing loss
            output_list = []
            input_list = []
            input_feature = [self.dataset_list[client_id].sample_feature(data_batch[client_id][0][0]).to(self.device) for client_id in range(self.client_num)]
            for layer in range(self.layer_num):
                output_list_temp = []
                for client_id in range(self.client_num):
                    adj, out_index = self.dataset_list[client_id].sample_graph(index_list = data_batch[client_id][layer], device = self.device)
                    if layer == 0:
                        output = client_model_list[client_id](input_feature[client_id], layer, adj, out_index)
                    else:
                        output = client_model_list[client_id](input_feature, layer, adj, out_index)
                    output_list_temp.append(output)
                    communicator.send(output, client_id)
                output_list.append(output_list_temp)
                input_feature = communicator.recv(-1)
                input_feature.retain_grad()
                input_list.append(input_feature)
            loss = 0
            for client_id in range(self.client_num):
                predict = client_model_list[client_id](input_feature, self.layer_num)
                # print(predict.sum())
                label = self.dataset_list[client_id].sample_label(data_batch[client_id][-1][-1]).to(self.device)
                loss += criterion(predict, label)
                # print(predict, label)
            loss_sum += loss.cpu().item()/(self.client_num)
            print("Iter:", it,"  Loss:", loss_sum/(it+1))
            input_feature.requires_grad_()
            # start backward propagation
            inputs = parameter_list[-1]
            inputs |= set((input_list[-1],))
            loss.backward(inputs = inputs, retain_graph = True) # get gradient of classifier
            for layer in range(self.layer_num-1,-1,-1):
                communicator.send(input_list[layer].grad, -1)
                inputs = parameter_list[layer]
                if layer > 0:
                    inputs |= set((input_list[layer-1],))
                for client_id in range(self.client_num):
                    partial_grad = communicator.recv(client_id)
                    output_list[layer][client_id].backward(gradient = partial_grad, inputs = inputs, retain_graph = True)
            self.optimizer.step()
        return client_model_list

    def test(self, client_model_list, global_iter, communicator):
        assert(len(client_model_list) == self.client_num)
        for client_model in client_model_list:
            client_model.to(self.device)
            client_model.eval()
        total = 0
        correct = 0
        losses = 0
        criterion = nn.CrossEntropyLoss()
        
        while True:
            # Prepare data batch node indices
            data_batch = [[[] for _ in range(self.layer_num)] for _ in range(self.client_num)]
            output= communicator.recv_idx()
            if output == -1:
                break
            for layer in range(self.layer_num-1,-1,-1):
                for client_id in range(self.client_num):
                    sampled_data = self.dataset_list[client_id].sample_id(output, self.batchsize, self.layer_list[layer])
                    communicator.send_idx(sampled_data[0], client_id)
                    data_batch[client_id][layer] = sampled_data
                output = communicator.recv_idx()
                for client_batch in data_batch:
                    client_batch[layer][0] = output
            # Forward propagation
            input_feature = [self.dataset_list[client_id].sample_feature(data_batch[client_id][0][0]).to(self.device) for client_id in range(self.client_num)]
            for layer in range(self.layer_num):
                for client_id in range(self.client_num):
                    adj, out_index = self.dataset_list[client_id].sample_graph(index_list = data_batch[client_id][layer], device = self.device)
                    if layer == 0:
                        output = client_model_list[client_id](input_feature[client_id], layer, adj, out_index)
                    else:
                        output = client_model_list[client_id](input_feature, layer, adj, out_index)
                    communicator.send(output, client_id)
                input_feature = communicator.recv(-1)
            predict = client_model_list[0](input_feature, self.layer_num)
            label = self.dataset_list[0].sample_label(data_batch[0][-1][-1]).to(self.device)
            loss = criterion(predict, label)
            correct += predict.argmax(dim=-1).eq(label).sum().item()
            losses += loss.detach().to('cpu').numpy()
            total += len(label)
        return correct/total, losses/total

class GFLClientSv2(GFLClientS):
    def __init__(self, dataset_list, client_num, batchsize, device, layer_list, q):
        super(GFLClientSv2, self).__init__(dataset_list, client_num, batchsize, device, layer_list)
        self.q = q

    def _prepare_batch(self, communicator):
        adj = [[[] for _ in range(self.client_num)] for _ in range(self.layer_num)]
        out_index= [[[] for _ in range(self.client_num)] for _ in range(self.layer_num)]
        data_batch = [[[] for _ in range(self.client_num)] for _ in range(self.layer_num)]
        output = communicator.recv_idx()
        # print(output, communicator.dataset.dataset.data.train_mask[output])
        for layer in range(self.layer_num-1,-1,-1):
            for client_id in range(self.client_num):
                sampled_data = self.dataset_list[client_id].sample_id(output, self.batchsize, self.layer_list[layer])
                communicator.send_idx(sampled_data[0], client_id)
                data_batch[layer][client_id] = sampled_data
            output = communicator.recv_idx()
            for client_batch in data_batch[layer]:
                client_batch[0] = output
            for client_id in range(self.client_num):
                adj[layer][client_id], out_index[layer][client_id] = self.dataset_list[client_id].sample_graph(index_list = data_batch[layer][client_id], device = self.device)
        return adj, out_index, data_batch

    def train(self, client_model_list, client_iter, global_iter, communicator, lr = None, momentum = None, update = False):
        assert(len(client_model_list) == self.client_num)
        # update learning rate and momentum
        self._update_lr(lr, momentum)
        parameter_set, parameter_list = self._collect_params(client_model_list)
        self._set_optimizer(parameter_set, global_iter, update)
        criterion = nn.CrossEntropyLoss()
        # Prepare data batch node indices
        loss_sum = 0
        # if global_iter == 0:
        #     self._set_optimizer(parameter_set, global_iter, update)
        for it in range(client_iter):
            adj_list, out_index_list, data_batch = self._prepare_batch(communicator)
            # print(data_batch[-1][0][-1])
            input_feature_0 = [self.dataset_list[client_id].sample_feature(data_batch[0][client_id][0]).to(self.device) for client_id in range(self.client_num)]
            staled_input_list = [[None for _ in range(self.client_num)] for _ in range(self.layer_num)]
            for itr in range(self.q):
                self.optimizer.zero_grad()
                # Start computing loss
                output_list = []
                # input_list = []
                for layer in range(self.layer_num):
                    output_list_temp = []
                    for client_id in range(self.client_num):
                        if layer == 0:
                            output = client_model_list[client_id](input_feature_0[client_id], layer, adj_list[layer][client_id], out_index_list[layer][client_id])
                        else:
                            input_feature = torch.add(staled_input_list[layer-1][client_id], output_list[layer-1][client_id], alpha= 1/self.client_num)
                            output = client_model_list[client_id](input_feature, layer, adj_list[layer][client_id], out_index_list[layer][client_id])
                        output_list_temp.append(output)
                        if staled_input_list[layer][client_id] is None:
                            communicator.send(output, client_id)
                    output_list.append(output_list_temp)
                    if staled_input_list[layer][0] is None:
                        staled_input = communicator.recv(-1)
                        for client_id in range(self.client_num):
                            with torch.no_grad():
                                staled_input_list[layer][client_id] = torch.add(staled_input.detach(), output_list[layer][client_id], alpha= -1/self.client_num)
                loss = 0
                for client_id in range(self.client_num):
                    input_feature = torch.add(staled_input_list[-1][client_id], output_list[-1][client_id], alpha= 1/self.client_num)
                    predict = client_model_list[client_id](input_feature, self.layer_num)
                    label = self.dataset_list[client_id].sample_label(data_batch[-1][client_id][-1]).to(self.device)
                    try:
                        loss += criterion(predict, label)
                    except:
                        print(label, data_batch[-1][client_id][-1])
                if itr == 0:
                    loss_sum += loss.cpu().item()/self.client_num
                # print(loss.item())
                loss.backward()
                self.optimizer.step()
            print("Iter:", it,"  Loss:", loss_sum/(it+1))
            # input_feature.requires_grad_()
            # start backward propagation
            # inputs = parameter_list[-1]
            # inputs |= set((input_list[-1],))
            # loss.backward(inputs = inputs, retain_graph = True) # get gradient of classifier
            # for layer in range(self.layer_num-1,-1,-1):
            #     communicator.send(input_list[layer+1].grad, -1)
            #     inputs = parameter_list[layer]
            #     if layer > 0:
            #         inputs |= set((input_list[layer],))
            #     for client_id in range(self.client_num):
            #         partial_grad = communicator.recv(client_id)
            #         output_list[layer][client_id].backward(gradient = partial_grad, inputs = inputs, retain_graph = True)
        return client_model_list

    def test(self, client_model_list, global_iter, communicator):
        assert(len(client_model_list) == self.client_num)
        for client_model in client_model_list:
            client_model.to(self.device)
            client_model.eval()
        total = 0
        correct = 0
        losses = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            while True:
                # Prepare data batch node indices
                data_batch = [[[] for _ in range(self.layer_num)] for _ in range(self.client_num)]
                output= communicator.recv_idx()
                if output == -1:
                    break
                for layer in range(self.layer_num-1,-1,-1):
                    for client_id in range(self.client_num):
                        sampled_data = self.dataset_list[client_id].sample_id(output, self.batchsize, self.layer_list[layer])
                        communicator.send_idx(sampled_data[0], client_id)
                        data_batch[client_id][layer] = sampled_data
                    output = communicator.recv_idx()
                    for client_batch in data_batch:
                        client_batch[layer][0] = output
                # Forward propagation
                input_feature = [self.dataset_list[client_id].sample_feature(data_batch[client_id][0][0]).to(self.device) for client_id in range(self.client_num)]
                output_list = [None for _ in range(self.client_num)]
                for layer in range(self.layer_num):
                    for client_id in range(self.client_num):
                        adj, out_index = self.dataset_list[client_id].sample_graph(index_list = data_batch[client_id][layer], device = self.device)
                        if layer == 0:
                            output = client_model_list[client_id](input_feature[client_id], layer, adj, out_index)
                        else:
                            input = input_feature
                            output = client_model_list[client_id](input, layer, adj, out_index)
                        output_list[client_id] = output
                        communicator.send(output, client_id)
                    input_feature = communicator.recv(-1)
                for client_id in range(self.client_num):
                    input = input_feature
                    predict = client_model_list[client_id](input, self.layer_num)
                    label = self.dataset_list[0].sample_label(data_batch[0][-1][-1]).to(self.device)
                    loss = criterion(predict, label)
                    correct += predict.argmax(dim=-1).eq(label).sum().item()
                    losses += loss.detach().to('cpu').numpy()
                    total += len(label)
        return correct/total, losses/total

"""
class GFLClientSCat(GFLClientS):
    def __init__(self, dataset_list, client_num, batchsize, device, layer_list, communicator):
        super(GFLClientSCat, self).__init__(dataset_list, client_num, batchsize, device, layer_list, communicator)

    def train(self, client_model_list, client_iter, global_iter, lr = None, momentum = None):
        assert(len(client_model_list) == self.client_num)
        # update learning rate and momentum
        self._update_lr(lr, momentum)
        parameter_set, parameter_list = self._collect_params(client_model_list)
        self._set_optimizer(parameter_set, global_iter)
        criterion = nn.CrossEntropyLoss()
        
        loss_sum = 0
        for it in range(client_iter):
            self.optimizer.zero_grad()
            # Prepare data batch node indices
            data_batch = [[[] for _ in range(self.layer_num)] for _ in range(self.client_num)]
            output= communicator.recv_idx()
            for layer in range(self.layer_num-1,-1,-1):
                for client_id in range(self.client_num):
                    sampled_data = self.dataset_list[client_id].sample_id(output, self.batchsize, self.layer_list[layer])
                    communicator.send_idx(sampled_data[0], client_id)
                    data_batch[client_id][layer] = sampled_data
                output = communicator.recv_idx()
                for client_batch in data_batch:
                    client_batch[layer][0] = output
            # Start computing loss
            output_list = []
            input_list = []
            input_feature = [self.dataset_list[client_id].sample_feature(data_batch[client_id][0][0]) for client_id in range(self.client_num)]
            for layer in range(self.layer_num):
                input_list.append(input_feature)
                output_list_temp = []
                for client_id in range(self.client_num):
                    adj, out_index = self.dataset_list[client_id].sample_graph(index_list = data_batch[client_id][layer])
                    if layer == 0:
                        output = client_model_list[client_id](input_feature[client_id], layer, adj, out_index)
                    else:
                        output = client_model_list[client_id](input_feature, layer, adj, out_index)
                    output_list_temp.append(output)
                    communicator.send(output, client_id)
                output_list.append(output_list_temp)
                input_feature = communicator.recv(-1)
                input_feature.retain_grad()
            input_list.append(input_feature)
            loss = 0
            for client_id in range(self.client_num):
                predict = client_model_list[client_id](input_feature, self.layer_num)
                label = self.dataset_list[client_id].sample_label(data_batch[client_id][-1][-1])
                loss += criterion(predict, label)
            loss_sum += loss.cpu().item()/(self.client_num*len(label))
            print("Iter:", it,"  Loss:", loss_sum/(it+1))
            input_feature.requires_grad_()
            # start backward propagation
            inputs = parameter_list[-1]
            inputs |= set((input_list[self.layer_num],))
            loss.backward(inputs = inputs, retain_graph = True) # get gradient of classifier
            for layer in range(self.layer_num,0,-1):
                communicator.send(input_list[layer].grad, -1)
                inputs = parameter_list[layer-1]
                if layer >1:
                    inputs |= set((input_list[layer-1],))
                for client_id in range(self.client_num):
                    partial_grad = communicator.recv(client_id)
                    output_list[layer-1][client_id].backward(gradient = partial_grad, inputs = inputs, retain_graph = True)
            self.optimizer.step()
        return client_model_list

    def test(self, client_model_list, global_iter):
        assert(len(client_model_list) == self.client_num)
        for client_model in client_model_list:
            client_model.to(self.device)
            client_model.eval()
        total = 0
        correct = 0
        losses = 0
        criterion = nn.CrossEntropyLoss()
        
        while True:
            # Prepare data batch node indices
            data_batch = [[[] for _ in range(self.layer_num)] for _ in range(self.client_num)]
            output= communicator.recv_idx()
            if output == -1:
                break
            for layer in range(self.layer_num-1,-1,-1):
                for client_id in range(self.client_num):
                    sampled_data = self.dataset_list[client_id].sample_id(output, self.batchsize, self.layer_list[layer])
                    communicator.send_idx(sampled_data[0], client_id)
                    data_batch[client_id][layer] = sampled_data
                output = communicator.recv_idx()
                for client_batch in data_batch:
                    client_batch[layer][0] = output
            # Forward propagation
            input_feature = [self.dataset_list[client_id].sample_feature(data_batch[client_id][0][0]) for client_id in range(self.client_num)]
            for layer in range(self.layer_num):
                for client_id in range(self.client_num):
                    adj, out_index = self.dataset_list[client_id].sample_graph(index_list = data_batch[client_id][layer])
                    if layer == 0:
                        output = client_model_list[client_id](input_feature[client_id], layer, adj, out_index)
                    else:
                        output = client_model_list[client_id](input_feature, layer, adj, out_index)
                    communicator.send(output, client_id)
                input_feature = communicator.recv(-1)
            predict = client_model_list[0](input_feature, self.layer_num)
            label = self.dataset_list[0].sample_label(data_batch[0][-1][-1])
            loss = criterion(predict, label)
            correct += predict.argmax(dim=-1).eq(label).sum().item()
            losses += loss.detach().to('cpu').numpy()
            total += len(label)
        return correct/total, losses/total

class GFL_client(object):
    def __init__(self, client_id, dataset, batchsize, device, layer_num):
        self.dataset = copy.deepcopy(dataset)
        self.client_id = client_id
        self.batchsize = batchsize
        self.device = device
        self.layer_num = layer_num
        self.lr = None
        self.momentum = None

    def train(self, client_model, client_iter, global_iter, communicator, lr = None, momentum = None):
        if lr is not None:
            self.lr = lr
        elif self.lr is None:
            self.lr = 0.001
        if momentum is not None:
            self.momentum = momentum
        elif self.momentum is None:
            self.momentum = 0.9
        client_model.to(self.device)
        client_model.train()
        optimizer = optim.SGD(client_model.parameters(), lr = self.lr/(0*math.sqrt(global_iter//4)+1), momentum=self.momentum)
        total = 0
        losses = []
        criterion = nn.CrossEntropyLoss()
        correct = 0

        for it in range(client_iter):
            optimizer.zero_grad()
            # Prepare data batch node indices
            data_batch = [[] for _ in range(self.layer_num+1)]
            data_batch[self.layer_num] = communicator.recv_idx()
            for layer in range(self.layer_num-1,-1,-1):
                sampled_data = self.dataset.sample_id(data_batch[layer+1], self.batchsize)
                communicator.send_idx(sampled_data, self.client_id)
                data_batch[layer] = communicator.recv_idx()
            # Start computing loss
            output_list = []
            input_list = []
            input_feature = self.dataset.sample_feature(data_batch[0])
            for layer in range(self.layer_num):
                input_list.append(input_feature)
                adj = self.dataset.sample_graph(input = data_batch[layer], output = data_batch[layer+1])
                output = client_model(input_feature, layer, adj)
                output_list.append(output)
                communicator.send(output, self.client_id)
                input_feature = communicator.recv(self.client_id)
            input_list.append(input_feature)
            predict = client_model(input_feature, None, self.layer_num)
            label = self.dataset.sample_label(data_batch[-1])
            loss = criterion(predict, label)
            # start backward propagation
            loss.backward() # get gradient of classifier
            for layer in range(self.layer_num,0,-1):
                communicator.send(input_feature[layer].grad, self.client_id)
                partial_grad = communicator.recv(self.client_id)
                output_list[layer-1].backward(partial_grad)
            optimizer.step()
            # correct += predict.argmax(dim=-1).eq(label).sum().item()
            # losses.append(loss.detach().to('cpu').numpy())
        return client_model

    def test(self, client_model, global_iter, communicator):
        client_model.to(self.device)
        client_model.eval()
        total = 0
        losses = []
        criterion = nn.CrossEntropyLoss()
        correct = 0
        while True:
            data_batch = [[] for _ in range(self.layer_num+1)]
            data_batch[self.layer_num] = communicator.recv_idx()
            if data_batch[self.layer_num] == -1:
                break
            for layer in range(self.layer_num-1,-1,-1):
                sampled_data = self.dataset.sample_id(data_batch[layer+1], self.batchsize)
                communicator.send_idx(sampled_data, self.client_id)
                data_batch[layer] = communicator.recv_idx()
            # Start computing loss
            input_feature = self.dataset.sample_feature(data_batch[0])
            for layer in range(self.layer_num):
                adj = self.dataset.sample_graph(input = data_batch[layer], output = data_batch[layer+1])
                output = client_model(input_feature, layer, adj)
                communicator.send(output, self.client_id)
                input_feature = communicator.recv(self.client_id)
            predict = client_model(input_feature, None, self.layer_num)
            label = self.dataset.sample_label(data_batch[-1])
            loss = criterion(predict, label)
            correct += predict.argmax(dim=-1).eq(label).sum().item()
            losses+=loss.detach().to('cpu').numpy()
            total += len(label)
        return correct/total, losses/total
"""