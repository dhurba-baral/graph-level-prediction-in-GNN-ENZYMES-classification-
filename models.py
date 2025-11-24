import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

# graph convolutional network model with 1 layer
class GCN_1Layer(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.5):
        super(GCN_1Layer, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.dropout = dropout
        self.lin = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, batch):
        # GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification layer
        x = self.lin(x)
        return x

# 2-layer graph convolutional network model
class GCN_2Layer(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.5):
        super(GCN_2Layer, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = dropout
        self.lin = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, batch):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification layer
        x = self.lin(x)
        return x

# Graph Attention Network model
class GAT_Model(nn.Module):
    """Graph Attention Network model"""
    
    def __init__(self, num_features, hidden_channels, num_classes, heads=4, dropout=0.5):
        super(GAT_Model, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        self.dropout = dropout
        self.lin = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, batch):
        # First GAT layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification layer
        x = self.lin(x)
        return x