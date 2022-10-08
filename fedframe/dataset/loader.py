import numpy as np
from fedframe.dataset.sampler import PlanetoidSplit, RandomSplit, HeriSplit, HeriLoader #, ClientSplit
from torch_geometric.datasets import Planetoid, Reddit, Reddit2, Flickr, Coauthor
from fedframe.dataset.dataset import HeriGraph

def load_graph_dataset(dataset:str, num_clients: int, dropout: float, GAT: bool = False):
    if dataset == "Cora" or dataset == "CiteSeer" or dataset == "PubMed":
        DS = load_Planetoid(dataset, num_clients, dropout)
    elif dataset == "Reddit":
        DS = load_Reddit(dataset, num_clients, dropout)
    elif dataset == "Reddit2":
        DS = load_Reddit2(dataset, num_clients, dropout)
    elif dataset == "Flickr":
        DS = load_Flickr(dataset, num_clients, dropout)
    elif dataset == "CS" or dataset == "Physics":
        DS = load_Coauthor(dataset, num_clients, dropout)
    elif dataset == "Suzhou" or dataset == "Amsterdam" or dataset == "Venice":
        client_datasets = []
        client_input_size = []
        for client_id in range(num_clients):
            DS = load_HeriGraph(dataset, num_clients, client_id)
            data_train = HeriLoader(DS, train = True, GAT = GAT)
            client_input_size.append(DS.data.x.size(1))
            client_datasets.append(data_train)
        DS = load_HeriGraph(dataset, num_clients, num_clients)
        server_dataset_train = HeriLoader(DS, train = True, GAT = GAT)
        server_dataset_test = HeriLoader(DS, train= False, GAT = GAT)
        return client_datasets, client_input_size, server_dataset_train, server_dataset_test
    feature_length = DS.data.x.size(1)
    feature_per_client = int((feature_length-1) // num_clients +1)
    feature_set = set(range(feature_length))
    client_datasets = []
    client_input_size = []
    for client_id in range(num_clients):
        feature_id = list(np.random.choice(list(feature_set), min(feature_per_client,len(feature_set)), replace= False))
        feature_id.sort()
        feature_set -= set(feature_id)
        client_input_size.append(len(feature_id))
        client_datasets.append(PlanetoidSplit(DS, client_id, feature_id, dropout, GAT = GAT))
    server_dataset_train = PlanetoidSplit(DS, -1, None, None, GAT = GAT)
    server_dataset_test = PlanetoidSplit(DS, -1, None, None, train= False, GAT=GAT)
    return client_datasets, client_input_size, server_dataset_train, server_dataset_test

def load_dist_graph_dataset(dataset:str, num_clients: int, client_id: int, dropout: float, GAT: bool = False):
    if dataset == "Cora" or dataset == "CiteSeer" or dataset == "PubMed":
        DS = load_Planetoid(dataset, num_clients, dropout)
    elif dataset == "Reddit":
        DS = load_Reddit(dataset, num_clients, dropout)
    elif dataset == "Reddit2":
        DS = load_Reddit2(dataset, num_clients, dropout)
    elif dataset == "Flickr":
        DS = load_Flickr(dataset, num_clients, dropout)
    elif dataset == "CS" or dataset == "Physics":
        DS = load_Coauthor(dataset, num_clients, dropout)
    elif dataset == "Suzhou" or dataset == "Amsterdam" or dataset == "Venice":
        DS = load_HeriGraph(dataset, num_clients, client_id)
        data_train = HeriLoader(DS, train = True, GAT = GAT)
        if client_id == num_clients:
            extra = HeriLoader(DS, train = False, GAT = GAT)
        else:
            extra = DS.data.x.size(1)
        return data_train, extra
    if client_id == num_clients:
        data_train = PlanetoidSplit(DS, -1, None, None, GAT = GAT)
        extra = PlanetoidSplit(DS, -1, None, None, train= False, GAT = GAT)
    else:
        feature_length = DS.data.x.size(1)
        feature_per_client = int((feature_length-1) // num_clients +1)
        feature_list = list(range(feature_length))
        feature_id = feature_list[client_id*feature_per_client: min((client_id+1)*feature_per_client, feature_length)]
        client_input_size = len(feature_id)
        data_train = PlanetoidSplit(DS, client_id, feature_id, dropout, GAT = GAT)
        extra = client_input_size
    return data_train, extra


def load_Planetoid(dataset:str, num_clients: int, dropout: float):
    if dataset == "PubMed":
        DS = Planetoid("./data/Planetoid", dataset, split="random", num_train_per_class= 320)
    elif dataset == "CiteSeer":
        DS = Planetoid("./data/Planetoid", dataset, split="random", num_train_per_class= 84)
    else:
        DS = Planetoid("./data/Planetoid", dataset, split="random", num_train_per_class= 60)
    return DS

def load_Reddit(dataset:str, num_clients: int, dropout: float):
    DS = Reddit("./data/Reddit")
    DS = RandomSplit(DS, portion = 0.66)
    return DS

def load_Reddit2(dataset:str, num_clients: int, dropout: float):
    DS = Reddit2("./data/Reddit2")
    DS = RandomSplit(DS, portion = 0.66)
    return DS

def load_Flickr(dataset:str, num_clients: int, dropout: float):
    DS = Flickr("./data/Flickr")
    DS = RandomSplit(DS, portion = 0.5)
    return DS

def load_Coauthor(dataset:str, num_clients: int, dropout: float):
    DS = Coauthor("./data/Coauthor", name = dataset)
    DS = RandomSplit(DS, portion = 0.5)
    return DS

def load_HeriGraph(dataset: str, num_clients: int, client_id: int):
    if client_id == num_clients:
        DS = HeriGraph("./data/HeriGraph", dataset, 3)
    elif num_clients == 1 and client_id == 0:
        DS = HeriGraph("./data/HeriGraph", dataset, 3)
    else:
        DS = HeriGraph("./data/HeriGraph", dataset, client_id%3)
    if num_clients == 1 and client_id == 0:
        DS = HeriSplit(DS, 6, train = 0.2, test = 0.5)
    else:
        DS = HeriSplit(DS, client_id%6, train = 0.2, test = 0.5)
    return DS