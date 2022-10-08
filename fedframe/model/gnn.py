import torch
import numpy as np
import copy
import os
import torch.nn as nn
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from fedframe.model.simple import Model
import math
from torch import Tensor
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse

class BaseClientLinear(Model):
    def __init__(self, name:str, model_size:str, input_size:int, hidden_size:int, output_size:int, mix_first = True, FFN = True):
        super().__init__(name)
        layer_list = self._get_layer_list(model_size)
        self.layer_list = layer_list
        self.block_num = len(layer_list)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.FFN = FFN
        if not self.FFN:
            fn = nn.Sequential(nn.Linear(input_size, hidden_size, bias = True), nn.ReLU(inplace=True))
            self.preprocess = FFN_Block(fn)
        self.block_list = None
        self.classifier = None

    def getFFN(self):
        state_dict = self.state_dict()
        state_dict_FFN = dict()
        count = 0
        for k,v in state_dict.items():
            if "FFN" in k or "classifier" in k:
                if count<2:
                    count += 1
                else:
                    state_dict_FFN[k] = v
        return state_dict_FFN
    
    def getParameters(self, layer_id = None):
        if layer_id is None:
            return self.parameters()
        else:
            assert(layer_id<= self.block_num)
            if layer_id == self.block_num or layer_id == -1:
                return self.classifier.parameters()
            elif layer_id == 0 and not self.FFN:
                return set(self.block_list[layer_id].parameters())|set(self.preprocess.parameters())
            else:
                return self.block_list[layer_id].parameters()

    def _get_layer_list(self, model_size:str):
        return self.get_layer_list(model_size)

    @staticmethod
    def get_layer_num(model_size:str):
        if model_size.startswith('2'):
            layer_num = 2
        elif model_size.startswith('4'):
            layer_num = 4
        elif model_size.startswith('1'):
            layer_num = 1
        elif model_size.startswith('3'):
            layer_num = 3
        else:
            raise NotImplementedError("model size incorrect")
        return layer_num

    @staticmethod
    def get_layer_list(model_size:str):
        if model_size == "2_1":
            layer_list = [1,1]
        elif model_size == "2_2":
            layer_list = [2,2]
        elif model_size == "2_3":
            layer_list = [3,3]
        elif model_size == "4_1":
            layer_list = [1,1,1,1]
        elif model_size == "4_2":
            layer_list = [2,2,2,2]
        elif model_size == "1_2":
            layer_list = [2]
        elif model_size == "1_3":
            layer_list = [3]
        elif model_size == "1_4":
            layer_list = [4]
        elif model_size == "3_1":
            layer_list = [1,1,1]
        elif model_size == "3_2":
            layer_list = [2,2,2]
        else:
            raise NotImplementedError("model size incorrect")
        return layer_list

class ClientGNNLinearCat(BaseClientLinear):
    def __init__(self, name:str, model_size:str, input_size:int, hidden_size:int, output_size:int, head_num:int, mix_first = True, FFN = True):
        super(ClientGNNLinearCat, self).__init__(name, model_size, input_size, hidden_size, output_size, mix_first, FFN)
        self.head_num = head_num
        assert (hidden_size % head_num == 0)
        block_list = []
        if not self.FFN:
            block_list.append(BasicLinearBlock(hidden_size, hidden_size, hidden_size//head_num, self.layer_list[0], mix_first, FFN))
        else:
            block_list.append(BasicLinearBlock(input_size, hidden_size, hidden_size//head_num, self.layer_list[0], mix_first, FFN))
        if self.block_num>1:
            for layer_id in range(1,self.block_num,1):
                block_list.append(BasicLinearBlock(hidden_size, hidden_size, hidden_size//head_num, self.layer_list[layer_id], mix_first, FFN))
        self.block_list = nn.Sequential(*block_list)
        self.classifier = nn.Linear(hidden_size, output_size, bias= True)

    def forward(self, input, block_id, adj = None, out_index = None):
        if block_id == self.block_num or block_id == -1 :
            output = self.classifier(input)
        elif block_id == 0 and not self.FFN:
            output = self.preprocess(input)
            output = self.block_list[block_id](output, adj)
        else:
            output = self.block_list[block_id](input, adj)
        return output

class ClientGNNLinearAvg(BaseClientLinear):
    def __init__(self, name:str, model_size:str, input_size:int, hidden_size:int, output_size:int, mix_first = True, residual = False, mu = None, FFN = True, GAT = False):
        super(ClientGNNLinearAvg, self).__init__(name,model_size, input_size, hidden_size, output_size, mix_first, FFN)
        self.residual = residual
        self.GAT = GAT
        self.mu = mu
        block_list = []
        if residual:
            if not self.FFN:
                block_list.append(BasicResidualBlock(hidden_size, hidden_size, hidden_size, self.layer_list[0], mix_first, mu, FFN))
            else:
                block_list.append(BasicResidualBlock(input_size, hidden_size, hidden_size, self.layer_list[0], mix_first, mu, FFN))
            if self.block_num>1:
                for layer_id in range(1,self.block_num,1):
                    if mu is None:
                        mu_1 = mu
                    else:
                        mu_1 = mu/(layer_id+1)
                    block_list.append(BasicResidualBlock(hidden_size, hidden_size, hidden_size, self.layer_list[layer_id], mix_first, mu_1, FFN))
        else:
            if GAT:
                block = BasicAttentionBlock
            else:
                block = BasicLinearBlock
            if not self.FFN:
                block_list.append(block(hidden_size, hidden_size, hidden_size, self.layer_list[0], mix_first, FFN))
            else:
                block_list.append(block(input_size, hidden_size, hidden_size, self.layer_list[0], mix_first, FFN))
            if self.block_num>1:
                for layer_id in range(1,self.block_num,1):
                    block_list.append(block(hidden_size, hidden_size, hidden_size, self.layer_list[layer_id], mix_first, FFN))
        self.block_list = nn.Sequential(*block_list)
        self.classifier = nn.Linear(hidden_size, output_size, bias= True)

    def forward(self, input, block_id, adj = None, out_index = None):
        if block_id == self.block_num or block_id == -1:
            output = self.classifier(input)
        elif block_id == 0 and not self.FFN:
            input = self.preprocess(input)
            if self.residual or self.GAT:
                output = self.block_list[block_id](input, adj, out_index)
            else:
                output = self.block_list[block_id](input, adj)
        else:
            if self.residual or self.GAT:
                output = self.block_list[block_id](input, adj, out_index)
            else:
                output = self.block_list[block_id](input, adj)
        return output

class BasicLinearBlock(Module):
    def __init__(self, input_size, hidden_size, output_size, layer_num, mix_first = True, FFN = True):
        super(BasicLinearBlock, self).__init__()
        self.layer_num = layer_num
        self.FFN = FFN
        layer_list = []
        if layer_num == 1 and not FFN:
            fn_io = nn.Sequential(nn.Linear(input_size, output_size, bias = True), nn.ReLU(inplace=True))
            layer_list.append(GCN_Block(fn_io, output_size, mix_first))
        else:
            fn_in = nn.Sequential(nn.Linear(input_size, hidden_size, bias = True), nn.ReLU(inplace=True))
            fn_out = nn.Sequential(nn.Linear(hidden_size, output_size, bias = True), nn.ReLU(inplace=True))
            fn = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias = True), nn.ReLU(inplace=True))
            for layer in range(layer_num):
                if FFN:
                    if layer == 0:
                        layer_list.append(FFN_Block(fn_in))
                    else:
                        layer_list.append(FFN_Block(fn))
                if layer == 0 and not FFN:
                    layer_list.append(GCN_Block(fn_in, hidden_size, mix_first))
                elif layer == layer_num-1:
                    layer_list.append(GCN_Block(fn_out, output_size, mix_first))
                else:
                    layer_list.append(GCN_Block(fn, hidden_size, mix_first))
        self.block = nn.Sequential(*layer_list)
    
    def forward(self, input, adj):
        output = input
        device = input.device
        if self.FFN:
            for layer_id, layer in enumerate(self.block):
                if layer_id % 2 == 0:
                    output = layer(output)
                else:
                    output = layer(output, adj[layer_id//2].to(device))
        else:
            for layer_id, layer in enumerate(self.block):
                output = layer(output, adj[layer_id].to(device))
        return output

class BasicResidualBlock(Module):
    def __init__(self, input_size, hidden_size, output_size, layer_num, mix_first = True, mu = None, FFN = True):
        super(BasicResidualBlock, self).__init__()
        self.layer_num = layer_num
        self.mu = mu
        self.FFN = FFN
        self.start_second = (input_size != output_size)
        layer_list = []
        if layer_num == 1 and not FFN:
            fn_io = nn.Sequential(nn.Linear(input_size, output_size, bias = True), nn.ReLU(inplace=True))
            layer_list.append(GCN_Block(fn_io, output_size, mix_first))
        else:
            fn_in = nn.Sequential(nn.Linear(input_size, hidden_size, bias = True), nn.ReLU(inplace=True))
            fn_out = nn.Sequential(nn.Linear(hidden_size, output_size, bias = True), nn.ReLU(inplace=True))
            fn = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias = True), nn.ReLU(inplace=True))
            for layer in range(layer_num):
                if FFN:
                    if layer == 0:
                        layer_list.append(FFN_Block(fn_in))
                    else:
                        layer_list.append(FFN_Block(fn))
                if layer == 0 and not FFN:
                    layer_list.append(GCN_Block(fn_in, hidden_size, mix_first))
                elif layer == layer_num-1:
                    layer_list.append(GCN_Block(fn_out, output_size, mix_first))
                else:
                    layer_list.append(GCN_Block(fn, hidden_size, mix_first))
        self.block = nn.Sequential(*layer_list)
    
    def forward(self, input, adj, out_index):
        output = input
        device = input.device
        if self.start_second:
            identity = None
        else: 
            identity = output.index_select(dim = 0, index = out_index.to(device))
        if self.FFN:
            for layer_id, layer in enumerate(self.block):
                if layer_id % 2 == 0:
                    output = layer(output)
                    if identity is None:
                        identity = output.index_select(dim = 0, index = out_index.to(device))
                else:
                    output = layer(output, adj[layer_id//2].to(device))
        else:
            for layer_id, layer in enumerate(self.block):
                output = layer(output, adj[layer_id].to(device))
        if self.mu is not None:
            return output* self.mu + (1-self.mu)*identity
        else:
            return output + identity

class BasicAttentionBlock(Module):
    def __init__(self, input_size, hidden_size, output_size, layer_num, mix_first = True, FFN = True):
        super().__init__()
        self.layer_num = layer_num
        self.FFN = FFN
        layer_list = []
        if layer_num == 1 and not FFN:
            fn_io = input_size #nn.Sequential(nn.Linear(input_size, output_size, bias = True), nn.ReLU(inplace=True))
            layer_list.append(GATBlock(fn_io, output_size//8, 8, dropout = 0))
        else:
            fn_in = nn.Sequential(nn.Linear(input_size, hidden_size, bias = True), nn.ReLU(inplace=True))
            # fn_out = nn.Sequential(nn.Linear(hidden_size, output_size, bias = True), nn.ReLU(inplace=True))
            fn = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias = True), nn.ReLU(inplace=True))
            for layer in range(layer_num):
                if FFN:
                    if layer == 0:
                        layer_list.append(FFN_Block(fn_in))
                    else:
                        layer_list.append(FFN_Block(fn))
                if layer == 0 and not FFN:
                    layer_list.append(GATBlock(input_size, hidden_size//8, 8, dropout=0))
                elif layer == layer_num-1:
                    layer_list.append(GATBlock(hidden_size, output_size//8, 8, dropout=0))
                else:
                    layer_list.append(GATBlock(hidden_size, hidden_size//8, 8, dropout=0))
        self.block = nn.Sequential(*layer_list)
    
    def forward(self, input, adj, index):
        output = input
        device = input.device
        if self.FFN:
            for layer_id, layer in enumerate(self.block):
                if layer_id % 2 == 0:
                    output = layer(output)
                else:
                    output = layer(output, adj[layer_id//2].to(device))
                    output = torch.index_select(output, dim = 0, index = index[layer_id//2])
        else:
            for layer_id, layer in enumerate(self.block):
                output = layer(output, adj[layer_id].to(device))
                output = torch.index_select(output, dim = 0, index = index[layer_id])
        return output

class GCN_Block(Module):
    def __init__(self, fn, output_size = None, mix_first = True):
        super(GCN_Block, self).__init__()
        self.GCN_fn = copy.deepcopy(fn)
        self.mix_first = mix_first
        self.output_size = output_size
        # self.output_size = None
        self.dropout = torch.nn.Dropout(p = 0.4)
        if output_size is not None:
            self.bn = nn.BatchNorm1d(output_size)
    
    def forward(self, input, adj):
        if self.mix_first:
            mixed = torch.mm(adj, input)
            output = self.GCN_fn(mixed)
        else:
            output = self.GCN_fn(input)
            output = torch.mm(adj, output)
        if self.output_size is not None:
            output = self.bn(output)
        return self.dropout(output)
    
class FFN_Block(Module):
    def __init__(self, fn):
        super(FFN_Block, self).__init__()
        self.FFN_fn = copy.deepcopy(fn)

    def forward(self, input):
        return self.FFN_fn(input)


class GATLayer(torch.nn.Module):
    head_dim = 1
    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)
        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()
        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class GATBlock(GATLayer):
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index
    nodes_dim = 0      # node dimension/axis
    head_dim = 1       # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout=0.6, add_skip_connection=True, bias=False, log_attention_weights=False):

        # Delegate initialization to the base class
        super().__init__(num_in_features, num_out_features, num_of_heads, concat, activation, dropout,
                      add_skip_connection, bias, log_attention_weights)

    def forward(self, in_nodes_features, edge_index):
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        attentions_per_edge = self.dropout(attentions_per_edge)
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)
        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return out_nodes_features

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)