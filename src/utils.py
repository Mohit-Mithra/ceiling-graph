import os.path as osp
import torch
from torch.nn import Linear, LSTM
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from torch_geometric.datasets import JODIEDataset
from torch_geometric.data import InMemoryDataset, TemporalData, download_url
from torch_geometric.nn import TGNMemory, TransformerConv
import pandas

class genericDataset(InMemoryDataset):

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name == 'retweet'

        super(genericDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.name}.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        df = pandas.read_csv(self.raw_paths[0], skiprows=1, header=None)

        src = torch.from_numpy(df.iloc[:, 0].values).to(torch.long)
        dst = torch.from_numpy(df.iloc[:, 1].values).to(torch.long)
        dst += int(src.max()) + 1
        t = torch.from_numpy(df.iloc[:, 2].values).to(torch.long)
        y = torch.from_numpy(df.iloc[:, 3].values).to(torch.long)
        msg = torch.from_numpy(df.iloc[:, 4:].values).to(torch.float)

        data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'{self.name.capitalize()}()'
        

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor_global(torch.nn.Module):
    def __init__(self, in_channels, global_in_channel):
        super(LinkPredictor_global, self).__init__()
        self.lin_src = Linear(in_channels+global_in_channel, in_channels+global_in_channel)
        self.lin_dst = Linear(in_channels+global_in_channel, in_channels+global_in_channel)
        self.lin_final = Linear(in_channels+global_in_channel,1)

    def forward(self, z_src, z_dst, z_global):
        z_global_repeat = z_global.view(1,-1).repeat(z_src.size()[0], 1)
        h = self.lin_src(torch.cat((z_src, z_global_repeat), dim=1)) + self.lin_dst(torch.cat((z_dst, z_global_repeat), dim=1))
        # print(h.size())
        h = h.relu()
        return self.lin_final(h)

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)