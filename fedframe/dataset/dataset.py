import torch
from torch_geometric.data import InMemoryDataset
import os.path as osp
from zipfile import ZipFile
import gdown
import torch_geometric as pyg
import pandas as pd
import networkx as nx
import pickle
import numpy as np

class HeriGraph(InMemoryDataset):
    def __init__(self, root, name, id, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.id = id
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[id])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw/dataset', self.name+'/')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed', self.name+'/')
    
    @property
    def raw_file_names(self):
        return ['Attribute_Labels.csv', 'Edge_List.csv', 'Textual_Features.csv', 'Value_Labels.csv', 'Visual_Features.csv']

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt', 'data_3.pt', 'data_4.pt']

    def download(self):
        # Download to `self.raw_dir`.
        file_id = 'https://drive.google.com/uc?id=1F-kNhIWyUboOBdVVeygYc5qPHeSbR1ou'
        destination =self.root+'/raw/dataset.zip'
        gdown.download(file_id, destination)
        with ZipFile(destination, 'r') as zipObj:
            zipObj.extractall(path = self.root+"/raw")
        

    def process(self):
        # Read data into huge `Data` list.
        data_list = self.read_heri()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        for id, data in enumerate(data_list):
            data, slices = self.collate([data])
            torch.save((data, slices), self.processed_paths[id])

    def read_heri(self):
        All_Edges_Graph = pd.read_csv(self.raw_dir+'Edge_List.csv', sep='\t', index_col='Unnamed: 0')
        v_feature = pd.read_csv(self.raw_dir+'Visual_Features.csv', sep='\t', index_col='Unnamed: 0')
        value_labels = pd.read_csv(self.raw_dir+'Value_Labels.csv', sep='\t', index_col='Unnamed: 0')
        attr_lables = pd.read_csv(self.raw_dir+ 'Attribute_Labels.csv', sep='\t', index_col='Unnamed: 0')
        # Texrt_Feature = pd.read_csv('data/HeriGraph/Venice/Textual_Features.csv', sep='\t', index_col='Unnamed: 0')
        img = v_feature.to_numpy()
        value_labels = value_labels.to_numpy()
        attr_labels = attr_lables.to_numpy()
        node = img[:,0]
        value_label = np.argmax(value_labels[:, 2:13], axis = 1)
        prob_value_label = np.max(value_labels[:, 2:13], axis = 1)
        value_label[value_labels[:, -1] == False] = -1
        prob_value_label[value_labels[:, -1] == False] = -1
        value_label = value_label.astype(np.float32)
        prob_value_label = prob_value_label.astype(np.float32)
        attr_label = np.argmax(attr_labels[:, 1:10], axis = 1)
        prob_attr_label = np.max(attr_labels[:, 1:10], axis = 1)
        attr_label[attr_labels[:, -1] == False] = -1
        prob_attr_label[attr_labels[:, -1] == False] = -1
        attr_label = attr_label.astype(np.float32)
        prob_attr_label = prob_attr_label.astype(np.float32)
        feature = img[:, 2:].astype(np.float32)

        G = nx.from_pandas_edgelist(All_Edges_Graph[All_Edges_Graph.Temporal_Similarity>0].loc[:,['0','1','Temporal_Similarity']],source='0', target='1',edge_attr=['Temporal_Similarity'])
        # feature_1 = img[:, 2:514].astype(np.float32)
        im1 = [(node,  {"x": feature[idx,:], "y1": value_label[idx], "py1": prob_value_label[idx], "y2": attr_label[idx], "py2": prob_attr_label[idx]}) for idx, node in enumerate(node)]
        G.add_nodes_from(im1)
        G1 = nx.Graph()
        G1.add_nodes_from(sorted(G.nodes(data=True)))
        G1.add_edges_from(G.edges(data=True))

        G = nx.from_pandas_edgelist(All_Edges_Graph[All_Edges_Graph.Social_Similarity>0].loc[:,['0','1','Social_Similarity','relationship']], source='0', target='1',edge_attr=['Social_Similarity','relationship'])
        # feature_2 = img[:, 517:882].astype(np.float32)
        im2 = [(node,  {"x": feature[idx,:], "y1": value_label[idx], "py1": prob_value_label[idx], "y2": attr_label[idx], "py2": prob_attr_label[idx]}) for idx, node in enumerate(node)]
        G.add_nodes_from(im2)
        G2 = nx.Graph()
        G2.add_nodes_from(sorted(G.nodes(data=True)))
        G2.add_edges_from(G.edges(data=True))

        G = nx.from_pandas_edgelist(All_Edges_Graph[All_Edges_Graph.Spatial_Similarity>0].loc[:,['0','1','Spatial_Similarity','geo_distance']], source='0', target='1',edge_attr=['Spatial_Similarity','geo_distance'])
        # feature_3 = img[:, 882:].astype(np.float32)
        im3 = [(node,  {"x": feature[idx,:], "y1": value_label[idx], "py1": prob_value_label[idx], "y2": attr_label[idx], "py2": prob_attr_label[idx]}) for idx, node in enumerate(node)]
        G.add_nodes_from(im3)
        G3 = nx.Graph()
        G3.add_nodes_from(sorted(G.nodes(data=True)))
        G3.add_edges_from(G.edges(data=True))

        G = nx.from_pandas_edgelist(All_Edges_Graph.loc[:,['0','1']], source='0', target='1')
        im4 = [(node,  {"x": feature[idx,:], "y1": value_label[idx], "py1": prob_value_label[idx], "y2": attr_label[idx], "py2": prob_attr_label[idx]}) for idx, node in enumerate(node)]
        G.add_nodes_from(im4)
        G4 = nx.Graph()
        G4.add_nodes_from(sorted(G.nodes(data=True)))
        G4.add_edges_from(G.edges(data=True))


        G1 = pyg.utils.from_networkx(G1)
        G2 = pyg.utils.from_networkx(G2)
        G3 = pyg.utils.from_networkx(G3)
        G4 = pyg.utils.from_networkx(G4)

        return [G1, G2, G3, G4]
