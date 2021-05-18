import os.path as osp
import torch
from torch.nn import Linear, LSTM
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from torch_geometric.datasets import JODIEDataset
from torch_geometric.nn.models.tgn import (LastNeighborLoader, IdentityMessage, LastAggregator)
from torch_geometric.data import InMemoryDataset, TemporalData, download_url
import pandas
from torch_geometric.nn import TGNMemory, TransformerConv
from utils import genericDataset, GraphAttentionEmbedding, LinkPredictor_khop
from model import local_neighbourhood_cluster

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join('..', 'data', 'JODIE')
dataset = JODIEDataset(path, name='wikipedia')
data = dataset[0].to(device)

train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)

model = local_neighbourhood_cluster(data, device)

optimizer = torch.optim.Adam(
    set(model.memory.parameters()) | set(model.gnn.parameters())
    | set(model.link_pred.parameters()), lr=0.0001)

model.create_adj_list()
model.set_optimizer(optimizer)

def main():    
    val_ap_list = []
    test_ap_list = []
    
    for epoch in range(1, 20):
        loss = model.train(train_data)
        print(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')
        val_ap, val_auc = model.test(val_data)
        test_ap, test_auc = model.test(test_data)
        val_ap_list.append(val_ap)
        test_ap_list.append(test_ap)
        print(f' Val AP: {val_ap:.4f},  Val AUC: {val_auc:.4f}')
        print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')
        
if __name__ == '__main__':
    main()