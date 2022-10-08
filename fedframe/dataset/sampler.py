from dataclasses import replace
from torch_geometric import utils
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_dense_adj
from fedframe.dataset.dataset import HeriGraph
import torch_geometric as pyg
import numpy as np
import torch.utils.data
import torch
import copy
from torch_sparse.tensor import SparseTensor

class GraphDataLoader():
    def __init__(self, dataset, train = True, GAT = False):
        self.train = train
        self.GAT = GAT
        self.dataset = copy.deepcopy(dataset)
    
    def sample_batch(self, batch_size):
        pass

    def sample_id(self, output, batch_size, prop = 1):
        pass

    def sample_feature(self, index):
        pass

    def sample_label(self, index):
        pass

    def sample_graph(self, index_list):
        pass

class PlanetoidSplit(GraphDataLoader):
    def __init__(self, dataset, client_id, feature_id, edge_dropout, train =True, GAT = False):
        super(PlanetoidSplit, self).__init__(dataset, train, GAT)
        self.client_id = client_id
        self.undirected = utils.is_undirected(dataset.data.edge_index)
        if client_id >-1:
            self.feature_id = torch.tensor(feature_id)
            self.dropout = edge_dropout
            self.dataset.data.x = self.dataset.data.x.index_select(dim = 1, index = self.feature_id)
            self.dataset.data.edge_index = utils.dropout_adj(self.dataset.data.edge_index, p = self.dropout, force_undirected= self.undirected, training= self.train)[0]
        if not self.train:
            self.test_set = (self.dataset.data.test_mask).nonzero(as_tuple=False).view(-1).tolist()
        else:
            self.node_idx = (self.dataset.data.train_mask).nonzero(as_tuple=False).view(-1).tolist()
        self.node_num = self.dataset.data.x.size(0)
        self.adj = SparseTensor(row = dataset.data.edge_index[0], col=dataset.data.edge_index[1], value = None, sparse_sizes = (dataset.data.num_nodes, dataset.data.num_nodes)).t()
    
    def sample_batch(self, batch_size:int):
        if self.train:
            batch_idx = list(np.random.permutation(self.node_idx)[:batch_size])
            batch_idx.sort()
        else:
            if len(self.test_set) == 0:
                batch_idx = -1
                self.test_set = (self.dataset.data.test_mask).nonzero(as_tuple=False).view(-1).tolist()#[i for (i,v) in enumerate(self.dataset.data.test_mask) if v]
            else:
                batch_size = min(batch_size, len(self.test_set))
                batch_idx = list(np.random.permutation(self.test_set)[:batch_size])
                self.test_set = list(set(self.test_set) - set(batch_idx))
                self.test_set.sort()
                batch_idx.sort()
        return batch_idx

    def sample_id(self, output:list, batch_size:int, prop = 1):
        index_list = [output]
        for _ in range(prop):
            input = self._sample_id(output, batch_size)
            index_list.insert(0, input)
            output = input
        return index_list
    
    def _sample_id(self, output: list, batch_size:int):
        # edge_list = [self.dataset.data.edge_index[0,i].item() for i in range(self.dataset.data.edge_index.size(1)) if self.dataset.data.edge_index[1,i].item() in output]
        # edge_list_1 = list(set(edge_list))
        # true_batch_size = min(len(output)*batch_size, len(edge_list_1))
        # new_node = list(set(np.random.permutation(edge_list)[:true_batch_size])|set(output))
        adj_t, new_node = self.adj.sample_adj(torch.tensor(output), batch_size, replace = False)
        new_node = new_node.tolist()
        new_node.sort()
        return new_node

    def sample_feature(self, index:list):
        if not isinstance(index, torch.Tensor):
            index = torch.tensor(index)
        return self.dataset.data.x.index_select(dim = 0, index = index)

    def sample_label(self, index:list):
        if not isinstance(index, torch.Tensor):
            index = torch.tensor(index)
        return self.dataset.data.y.index_select(dim = 0, index = index)

    def sample_graph(self, index_list, device = None):
        adj_list = []
        if self.GAT:
            output_idx = []
            for input, output in zip(index_list[:-1], index_list[1:]):
                adj, idx = self._sample_graph_gat(input, output)
                if device is not None:
                    adj = adj.to(device)
                    idx = idx.to(device)
                adj_list.append(adj)
                output_idx.append(idx)
        else:
            for input, output in zip(index_list[:-1], index_list[1:]):
                adj_list.append(self._sample_graph(input, output))
                if device is not None:
                    adj_list[-1] = adj_list[-1].to(device)
            output_idx = torch.tensor([index_list[0].index(idx) for idx in index_list[-1]])
            if device is not None:
                output_idx = output_idx.to(device)
        return adj_list, output_idx

    def _sample_graph(self, input:list, output:list):
        subgraph = self.dataset.data.subgraph(torch.tensor(input))
        edge_index = subgraph.edge_index
        output_idx = [input.index(idx) for idx in output]
        # mask = [i in output_idx for i in edge_index[1,:]]
        edge_index, weight = gcn_norm(edge_index, num_nodes= len(input), improved= True)
        adj = to_dense_adj(edge_index, edge_attr = weight).squeeze()
        adj = torch.index_select(adj, dim = 0, index= torch.tensor(output_idx))
        return adj
    
    def _sample_graph_gat(self, input:list, output:list):
        subgraph = self.dataset.data.subgraph(torch.tensor(input))
        edge_index = subgraph.edge_index
        output_idx = torch.tensor([input.index(idx) for idx in output])
        return edge_index, output_idx

def RandomSplit(dataset, portion = 0.1):
    data = dataset.data
    node_num = data.x.size(0)
    num_train_per_class = int(node_num*portion/dataset.num_classes)
    num_val = 0
    num_test = int(min(max(1200,dataset.num_classes*150), node_num*portion))
    setattr(data, 'train_mask', torch.zeros(node_num, dtype=torch.bool))
    setattr(data, 'test_mask', torch.zeros(node_num, dtype=torch.bool))
    setattr(data, 'val_mask', torch.zeros(node_num, dtype=torch.bool))
    data.train_mask.fill_(False)
    for c in range(dataset.num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
        data.train_mask[idx] = True

    remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
    remaining = remaining[torch.randperm(remaining.size(0))]

    data.val_mask.fill_(False)
    data.val_mask[remaining[:num_val]] = True

    data.test_mask.fill_(False)
    data.test_mask[remaining[num_val:num_val + num_test]] = True
    return dataset

def HeriSplit(dataset:HeriGraph, split: int, train: float, test: float):
    assert(train>0 and test >0 and (train+test)<=1),"incorrect train, test portion"
    Block_1 = list(range(0,512))
    Block_2 = list(range(515,880))
    Block_3 = list(range(880,982))
    SPLIT_IDX = [Block_1, Block_2, Block_3, Block_1+Block_2, Block_1+Block_3, Block_2+Block_3, Block_1+Block_2+Block_3]
    data_1 = dataset.data
    # print(data_1.x[:,SPLIT_IDX[split]].size(), len(SPLIT_IDX[split]))
    data = pyg.data.Data(x = data_1.x[:, SPLIT_IDX[split]] if data_1.x is not None else None, y = data_1.y2.type(torch.LongTensor), edge_index = data_1.edge_index)
    setattr(data, 'num_nodes', len(data_1.y2))
    node_num = data.num_nodes
    setattr(data, 'train_mask', torch.zeros(node_num, dtype=torch.bool))
    setattr(data, 'test_mask', torch.zeros(node_num, dtype=torch.bool))
    data.train_mask.fill_(False)
    data.test_mask.fill_(False)
    for c in range(9):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        num_class_node = idx.size(0)
        num_train = int(num_class_node *train)
        num_test = int(num_class_node* test)
        mask = torch.randperm(num_class_node)
        idx_train = idx[mask[:num_train]]
        idx_test = idx[mask[num_train:num_train+num_test]]
        data.train_mask[idx_train] = True
        data.test_mask[idx_test] = True
    dataset.data = data
    # print(dataset.data)
    return dataset

class HeriLoader(GraphDataLoader):
    def __init__(self, dataset, train =True, GAT = False):
        super(HeriLoader, self).__init__(dataset, train, GAT)
        self.undirected = utils.is_undirected(dataset.data.edge_index)
        if not self.train:
            self.test_set = (self.dataset.data.test_mask).nonzero(as_tuple=False).view(-1).tolist()
        else:
            self.node_idx = (self.dataset.data.train_mask).nonzero(as_tuple=False).view(-1).tolist()
        self.node_num = self.dataset.data.num_nodes
        self.adj = SparseTensor(row = dataset.data.edge_index[0], col=dataset.data.edge_index[1], value = None, sparse_sizes = (dataset.data.num_nodes, dataset.data.num_nodes)).t()
    
    def sample_batch(self, batch_size:int):
        if self.train:
            batch_idx = list(np.random.permutation(self.node_idx)[:batch_size])
            batch_idx.sort()
        else:
            if len(self.test_set) == 0:
                batch_idx = -1
                self.test_set = (self.dataset.data.test_mask).nonzero(as_tuple=False).view(-1).tolist()
            else:
                batch_size = min(batch_size, len(self.test_set))
                batch_idx = list(np.random.permutation(self.test_set)[:batch_size])
                self.test_set = list(set(self.test_set) - set(batch_idx))
                self.test_set.sort()
                batch_idx.sort()
        return batch_idx

    def sample_id(self, output:list, batch_size:int, prop = 1):
        index_list = [output]
        for _ in range(prop):
            input = self._sample_id(output, batch_size)
            index_list.insert(0, input)
            output = input
        return index_list
    
    def _sample_id(self, output: list, batch_size:int):
        adj_t, new_node = self.adj.sample_adj(torch.tensor(output), batch_size, replace = False)
        new_node = new_node.tolist()
        new_node.sort()
        return new_node

    def sample_feature(self, index:list):
        if not isinstance(index, torch.Tensor):
            index = torch.tensor(index)
        if self.dataset.data.x is None:
            raise RuntimeError("data feature is none!")
        return self.dataset.data.x.index_select(dim = 0, index = index)

    def sample_label(self, index:list):
        if not isinstance(index, torch.Tensor):
            index = torch.tensor(index)
        return self.dataset.data.y.index_select(dim = 0, index = index)

    def sample_graph(self, index_list, device = None):
        adj_list = []
        if self.GAT:
            output_idx = []
            for input, output in zip(index_list[:-1], index_list[1:]):
                adj, idx = self._sample_graph_gat(input, output)
                if device is not None:
                    adj = adj.to(device)
                    idx = idx.to(device)
                adj_list.append(adj)
                output_idx.append(idx)
        else:
            for input, output in zip(index_list[:-1], index_list[1:]):
                adj_list.append(self._sample_graph(input, output))
                if device is not None:
                    adj_list[-1] = adj_list[-1].to(device)
            output_idx = torch.tensor([index_list[0].index(idx) for idx in index_list[-1]])
            if device is not None:
                output_idx = output_idx.to(device)
        return adj_list, output_idx

    def _sample_graph(self, input:list, output:list):
        subgraph = self.dataset.data.subgraph(torch.tensor(input))
        edge_index = subgraph.edge_index
        output_idx = [input.index(idx) for idx in output]
        # mask = [i in output_idx for i in edge_index[1,:]]
        edge_index, weight = gcn_norm(edge_index, num_nodes= len(input), improved= True)
        adj = to_dense_adj(edge_index, edge_attr = weight).squeeze()
        adj = torch.index_select(adj, dim = 0, index= torch.tensor(output_idx))
        return adj
    
    def _sample_graph_gat(self, input:list, output:list):
        subgraph = self.dataset.data.subgraph(torch.tensor(input))
        edge_index = subgraph.edge_index
        output_idx = torch.tensor([input.index(idx) for idx in output])
        return edge_index, output_idx