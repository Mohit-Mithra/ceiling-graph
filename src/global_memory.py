from torch_geometric.nn.inits import zeros
import os.path as osp
import torch
from torch.nn import Linear, LSTM

class global_memory_linear(torch.nn.Module):
    def __init__(self, in_channel, global_mem_dimension):
        super(global_memory_linear, self).__init__() 
        self.lin_global1 = Linear(in_channel*2, in_channel*2)
        self.lin_global2 = Linear(in_channel*2, global_mem_dimension)

    def forward(self, src_embedding, dst_embedding):
        inp = torch.cat((src_embedding, dst_embedding), dim=1)
        global_embedding = self.lin_global1(inp)
        global_embedding = self.lin_global2(global_embedding)
        # print('Global embedding shape')
        # print(global_embedding.size())
        return global_embedding


class global_memory_lstm(torch.nn.Module):
    def __init__(self, in_channel, hidden_dim, embedding_size):
        super(global_memory_lstm, self).__init__() 
        self.hidden_dim = hidden_dim
        self.h = torch.zeros(1, 1, hidden_dim).cuda()
        self.c = torch.zeros(1, 1, hidden_dim).cuda()
        self.lstm_out = torch.zeros(200, 1, hidden_dim).cuda()
        self.lstm = LSTM(in_channel*2, hidden_dim)
        # self.linear = Linear(hidden_dim, embedding_size)

    def forward(self, src_embedding, dst_embedding):
        inp = torch.cat((src_embedding, dst_embedding), dim=1)
        lstm_out, (self.h, self.c) = self.lstm(inp.view(len(inp), 1, -1), (self.h, self.c))
        # print(lstm_out.size())
        # embedding = self.linear(lstm_out.view(len(inp), -1))
        return self.h

    def reset_state(self):
        zeros(self.h)
        zeros(self.c)
        zeros(self.lstm_out)

    def detach(self):
        self.h.detach_()
        self.c.detach_()
        self.lstm_out.detach_()

class global_lstmcell(torch.nn.Module):
    def __init__(self, in_channel, hidden_dim, embedding_size):
        super(global_lstmcell, self).__init__() 
        self.hidden_dim = hidden_dim
        self.h = torch.zeros(1, hidden_dim).cuda()
        self.c = torch.zeros(1, hidden_dim).cuda()
        self.lstm = torch.nn.LSTMCell(in_channel*2, embedding_size)
        # self.linear = Linear(hidden_dim, embedding_size)

    def forward(self, src_embedding, dst_embedding):
        inp = torch.cat((src_embedding, dst_embedding), dim=1)
        for i in range(inp.size()[0]):
            self.h, self.c = self.lstm(inp[i].view(1, -1), (self.h, self.c))
        return self.h

    def reset_state(self):
        self.lstm.reset_parameters()
        zeros(self.h)
        zeros(self.c)    

    def detach(self):
        self.h.detach_()
        self.c.detach_()