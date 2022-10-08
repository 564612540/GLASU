from dis import dis
from os import lseek
import numpy
import torch
import copy
import torch.distributed as dist
from datetime import timedelta

class Communicator(object):
    def __init__(self, num_client):
        self.num_client = num_client
    
    def send(self,data, id):
        pass

    def recv(self,id):
        pass

    def send_idx(self,index, id):
        pass

    def recv_idx(self):
        pass

class FakeCommunicatorCat(Communicator):
    def __init__(self, num_client, dataset, batch_size):
        super(FakeCommunicatorCat,self).__init__(num_client)
        self.dataset = copy.deepcopy(dataset)
        self.batch_size = batch_size
        self.state = 0
        self.count = 0
        self.buffer = None
        self.block_size = None
        '''
        state: 
            0 initial
            1 recieve index
            2 send index
            3 recieve feature
            4 send feature
            5 recieve gradient
            6 send gradient
        '''

    def send(self, data, id):
        assert(id<self.num_client)
        if self.state == 0:
            if id<0:#send gradient
                self.buffer = data
                self.state = 6
            else: # send feature
                self.buffer = [None for _ in range(self.num_client)]
                self.count = self.num_client -1
                self.buffer[id] = data
                self.state = 4
        elif self.state == 4 and self.count>0:
            self.buffer[id] = data
            self.count -= 1
        return
    
    def recv(self, id):
        assert(id<self.num_client)
        if self.state == 6: #recieving gradient
            total_length = self.buffer.size(1)
            assert (total_length & self.num_client == 0)
            self.block_size = total_length//self.num_client
            self.buffer = self.buffer.split(self.block_size, dim = 1)
            self.state = 5
            self.count = self.num_client
        if self.state == 5:
            data = self.buffer[id]
        if self.state == 4 and self.count == 0: # revieving feature
            self.buffer = torch.cat(self.buffer, dim = 1)
            self.state = 0
            self.count = 1
            data = self.buffer
        self.count -= 1
        if (self.count == 0):
            self.state = 0
        return data

    def send_idx(self, index, id):
        # print("send",self.state, self.count)
        if self.state == 0:
            self.buffer = [None for _ in range(self.num_client)]
            self.count = self.num_client -1
            self.buffer[id] = index
            self.state = 2
        elif self.state == 2 and self.count>0:
            self.buffer[id] = index
            self.count -= 1
        return

    def recv_idx(self):
        # print("recv", self.state, self.count)
        if self.state == 0:
            index = self.dataset.sample_batch(self.batch_size)
        if self.state == 2 and self.count == 0:
            index_set = set()
            for index in self.buffer:
                index_set |= set(index)
            self.buffer = list(index_set)
            self.buffer.sort()
            self.count = 0
            self.state = 0
            index = self.buffer
        return index

class FakeCommunicatorAvg(Communicator):
    def __init__(self, num_client, dataset, batch_size):
        super(FakeCommunicatorAvg,self).__init__(num_client)
        self.dataset = copy.deepcopy(dataset)
        self.batch_size = batch_size
        self.state = 0
        self.count = 0
        self.buffer = None
        self.block_size = None
        '''
        state: 
            0 initial
            1 recieve index
            2 send index
            3 recieve feature/gradient
            4 send feature/gradient
        '''

    def send(self, data, id):
        assert(id<self.num_client)
        if self.state == 0: # start sending
            self.count = self.num_client -1
            if id > -1:
                self.buffer = [None for _ in range(self.num_client)]
                self.buffer[id] = data
            else:
                self.buffer = [data]
            self.state = 4
        elif self.state == 4 and self.count>0:
            self.buffer[id] = data
            self.count -= 1
        return
    
    def recv(self, id):
        assert(id<self.num_client)
        if self.state == 4: # start receiving
            if id<0: # send once 
                data = torch.mean(torch.stack(self.buffer), dim = 0)
                self.state = 0
                self.count = 0
            else: # send multiple times
                self.buffer = torch.mean(torch.stack(self.buffer), dim = 0)
                # data = self.buffer
                self.count = self.num_client
                self.state = 3
        if self.state == 3:
            data = self.buffer
            self.count -= 1
            if self.count == 0:
                self.state = 0
        return data

    def send_idx(self, index, id):
        # print("send",self.state, self.count)
        if self.state == 0:
            self.buffer = [None for _ in range(self.num_client)]
            self.count = self.num_client -1
            self.buffer[id] = index
            self.state = 2
        elif self.state == 2 and self.count>0:
            self.buffer[id] = index
            self.count -= 1
        return

    def recv_idx(self, id = None):
        # print("recv", self.state, self.count)
        if self.state == 0:
            index = self.dataset.sample_batch(self.batch_size)
        if self.state == 2 and self.count == 0:
            index_set = set()
            for index in self.buffer:
                index_set |= set(index)
            self.buffer = list(index_set)
            self.buffer.sort()
            self.count = 0
            self.state = 0
            index = self.buffer
        return index

class ClienCommunicatorTorch(Communicator):
    def __init__(self, num_client, client_id, backend, init_method, device):
        super().__init__(num_client)
        self.client_id = client_id
        dist.init_process_group(backend, init_method, world_size=num_client+1, rank= client_id, timeout=timedelta(minutes = 30))
        self.comm_group = dist.new_group(ranks = list(range(num_client+1)))
        self.buffer = None
        self.device = device
    
    @torch.no_grad()
    def send(self, data, id = None):
        self.buffer = torch.empty_like(data).to(self.device)
        data = data.to(self.device)
        empty_list = [torch.empty_like(data).to(self.device) for _ in range(self.num_client+1)]
        dist.all_gather(empty_list, data, group= self.comm_group)

    @torch.no_grad()
    def recv(self, id = None):
        dist.broadcast(self.buffer, src = self.num_client, group= self.comm_group)
        data = self.buffer
        self.buffer = None
        return data
    
    @torch.no_grad()
    def send_idx(self, index, id = None):
        empty_list = [None for _ in range(self.num_client+1)]
        dist.all_gather_object(empty_list, index, group= self.comm_group)

    @torch.no_grad()
    def recv_idx(self, id = None):
        self.buffer = [None]
        dist.broadcast_object_list(self.buffer, src=self.num_client, group= self.comm_group)
        data = self.buffer[0]
        return data

class ServerCommunicatorTorch(Communicator):
    def __init__(self, num_client, backend, init_method, device):
        super().__init__(num_client)
        self.id = 0
        dist.init_process_group(backend, init_method, world_size=num_client+1, rank= self.num_client, timeout=timedelta(minutes = 30))
        self.comm_group = dist.new_group(ranks = list(range(num_client+1)))
        self.buffer = None
        self.device = device
    
    @torch.no_grad()
    def send(self, data):
        data = data.to(self.device)
        dist.broadcast(data, src = self.num_client, group= self.comm_group)

    @torch.no_grad()
    def recv(self, size):
        self.buffer = [torch.empty(size).to(self.device) for _ in range(self.num_client+1)]
        empty_data = torch.empty(size).to(self.device)
        dist.all_gather(self.buffer,empty_data, group= self.comm_group)
        data = self.buffer[:-1]
        self.buffer = None
        return data
    
    @torch.no_grad()
    def send_idx(self, index):
        # data = torch.tensor(index).to(self.device)
        # length = torch.tensor(len(index)).to(self.device)
        # dist.broadcast(length, src = self.num_client, group= self.comm_group)
        dist.broadcast_object_list([index], src=self.num_client, group = self.comm_group)

    @torch.no_grad()
    def recv_idx(self):
        self.buffer = [None for _ in range(self.num_client+1)]
        empty_data = set()
        dist.all_gather_object(self.buffer, empty_data, group= self.comm_group)
        data = self.buffer[:-1]
        self.buffer = None
        return data