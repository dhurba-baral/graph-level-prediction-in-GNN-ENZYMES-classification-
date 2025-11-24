import pandas as pd
import os
import torch
from torch_geometric.data import Data, Dataset
import numpy as np

# Custom dataset class for ENZYMES data
class ENZYMESDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.root = root
        super(ENZYMESDataset, self).__init__(root, transform, pre_transform)
        self.data_list = self.process_data()
    
    #process the raw data files
    def process_data(self):
        # Read all files
        adjacency = pd.read_csv(os.path.join(self.root, 'ENZYMES_A.txt'), 
                                header=None, names=['source', 'target'])
        graph_indicator = pd.read_csv(os.path.join(self.root, 'ENZYMES_graph_indicator.txt'), 
                                       header=None, names=['graph_id'])
        graph_labels = pd.read_csv(os.path.join(self.root, 'ENZYMES_graph_labels.txt'), 
                                    header=None, names=['label'])
        node_attributes = pd.read_csv(os.path.join(self.root, 'ENZYMES_node_attributes.txt'), 
                                      header=None)
        node_features = node_attributes.values

        # Compute mean and std for each feature dimension
        mean = np.mean(node_features, axis=0)
        std = np.std(node_features, axis=0)
        
        # Avoid division by zero (if a feature has no variance)
        std[std == 0] = 1
        
        # Standardize: (x - mean) / std
        # This ensures each feature has mean=0 and std=1
        node_features = (node_features - mean) / std
        
        # Adjust indices to start from 0
        adjacency['source'] = adjacency['source'] - 1
        adjacency['target'] = adjacency['target'] - 1
        graph_indicator['graph_id'] = graph_indicator['graph_id'] - 1
        graph_labels['label'] = graph_labels['label'] - 1
        
        # Create data list for each graph
        data_list = []
        num_graphs = graph_labels.shape[0]
        
        for graph_id in range(num_graphs):
            # Get nodes for this graph
            node_mask = graph_indicator['graph_id'] == graph_id
            node_indices = np.where(node_mask)[0]
            
            # Map global node indices to local indices
            global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(node_indices)}
            
            # Get edges for this graph
            edge_mask = adjacency['source'].isin(node_indices) & adjacency['target'].isin(node_indices)
            edges = adjacency[edge_mask].copy()
            
            # Convert to local indices
            edges['source'] = edges['source'].map(global_to_local)
            edges['target'] = edges['target'].map(global_to_local)
            
            # Create edge index
            edge_index = torch.tensor([edges['source'].values, edges['target'].values], dtype=torch.long)
            
            # Get node features
            x = torch.tensor(node_features[node_indices], dtype=torch.float)
            
            # Get label
            y = torch.tensor([graph_labels.iloc[graph_id]['label']], dtype=torch.long)
            
            # Create data object
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        
        return data_list
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]