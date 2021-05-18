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

class local_neighbourhood_cluster(torch.nn.Module):

	def __init__(self, data, device):
		super(local_neighbourhood_cluster, self).__init__()

		memory_dim = time_dim = self.embedding_dim = 100
		self.data = data
		self.device = device 

		self.memory =TGNMemory(
						self.data.num_nodes,
						self.data.msg.size(-1),
						memory_dim,
						time_dim,
						message_module = IdentityMessage(self.data.msg.size(-1), memory_dim, time_dim),
						aggregator_module = LastAggregator(),
					).to(self.device)

		self.gnn = GraphAttentionEmbedding(
						in_channels = memory_dim,
						out_channels = self.embedding_dim,
						msg_dim = self.data.msg.size(-1),
						time_enc = self.memory.time_enc,
					).to(self.device)

		self.link_pred = LinkPredictor_khop(in_channels=self.embedding_dim).to(self.device)

		self.neighbor_loader = LastNeighborLoader(self.data.num_nodes, size=10, device=self.device)

		self.criterion = torch.nn.BCEWithLogitsLoss()

		self.assoc = torch.empty(self.data.num_nodes, dtype=torch.long, device=self.device)
		
		self.min_dst_idx, self.max_dst_idx = int(self.data.dst.min()), int(self.data.dst.max())
		self.adj_list = {}
		
	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

	def create_adj_list(self):

		for i in self.data.src.cpu().numpy():
			self.adj_list[i] = set()

		for i in self.data.dst.cpu().numpy():
			self.adj_list[i] = set()

		for i in range(self.min_dst_idx, self.max_dst_idx+1):
			self.adj_list[i] = set()


	def train(self, train_data):

		self.memory.train()
		self.gnn.train()
		self.link_pred.train()

		self.memory.reset_state()  # Start with a fresh memory.
		self.neighbor_loader.reset_state()  # Start with an empty graph.

		total_loss = 0
		i = 0
		for batch in train_data.seq_batches(batch_size=200):
			
			self.optimizer.zero_grad()
			
			src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
			
			neg_dst = torch.randint(self.min_dst_idx, self.max_dst_idx + 1, (src.size(0), ),dtype=torch.long, device=self.device)
			
			all_nodes = torch.cat([src, pos_dst, neg_dst])

			for s,d,n in zip(src, pos_dst, neg_dst):
				
				self.adj_list[int(s.cpu())].add(int(d.cpu()))
				self.adj_list[int(d.cpu())].add(int(s.cpu()))

				adj_nodes_s = torch.tensor(list(self.adj_list[int(s.cpu())]), dtype=torch.long, device=self.device)
				adj_nodes_d = torch.tensor(list(self.adj_list[int(d.cpu())]), dtype=torch.long, device=self.device)
				adj_nodes_n = torch.tensor(list(self.adj_list[int(n.cpu())]), dtype=torch.long, device=self.device)
				
				all_nodes = torch.cat([all_nodes, adj_nodes_s, adj_nodes_d, adj_nodes_n])
			
			n_id = all_nodes.unique()
			n_id, edge_index, e_id = self.neighbor_loader(n_id)

			self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

			# print(src)
			# print(assoc[src])
			
			# Get updated memory of all nodes involved in the computation.
			z, last_update = self.memory(n_id)
			z = self.gnn(z, last_update, edge_index, self.data.t[e_id], self.data.msg[e_id])

			z_adj_src = torch.empty(0, self.embedding_dim, dtype=torch.long, device = self.device)
			z_adj_dst = torch.empty(0, self.embedding_dim, dtype=torch.long, device = self.device)
			z_adj_neg = torch.empty(0, self.embedding_dim, dtype=torch.long, device = self.device)

			for s,d,n in zip(src, pos_dst, neg_dst):

				adj_nodes_s = torch.tensor(list(self.adj_list[int(s.cpu())])[-3:], dtype=torch.long, device=self.device)
				adj_nodes_d = torch.tensor(list(self.adj_list[int(d.cpu())])[-3:], dtype=torch.long, device=self.device)
				adj_nodes_n = torch.tensor(list(self.adj_list[int(n.cpu())])[-3:], dtype=torch.long, device=self.device)

				# print(adj_nodes_s)
				# print(adj_nodes_n)
				
				z_adjs = z[self.assoc[adj_nodes_s]]
				z_adjd = z[self.assoc[adj_nodes_d]]
				z_adjn = z[self.assoc[adj_nodes_n]]
				
				if self.assoc[adj_nodes_n].tolist() == []:
					z_adjn = torch.zeros(1, self.embedding_dim, dtype=torch.float32, device=self.device)

				z_adj_src = torch.cat((z_adj_src, z_adjs.mean(0, True)), 0)
				z_adj_dst = torch.cat((z_adj_dst, z_adjd.mean(0, True)), 0)
				z_adj_neg = torch.cat((z_adj_neg, z_adjn.mean(0, True)), 0)

			# print(z_adj_src.size(), z_adj_dst.size(), z_adj_neg.size())

			pos_out = self.link_pred(z[self.assoc[src]], z[self.assoc[pos_dst]], z_adj_src, z_adj_dst)
			neg_out = self.link_pred(z[self.assoc[src]], z[self.assoc[neg_dst]], z_adj_src, z_adj_neg)

			# print(pos_out[0])
			# print(neg_out[0])
			# print()

			self.loss = self.criterion(pos_out, torch.ones_like(pos_out))
			self.loss += self.criterion(neg_out, torch.zeros_like(neg_out))

			# Update memory and neighbor loader with ground-truth state.
			self.memory.update_state(src, pos_dst, t, msg)
			self.neighbor_loader.insert(src, pos_dst)

			self.loss.backward()
			self.optimizer.step()
			self.memory.detach()

			total_loss += float(self.loss) * batch.num_events
		
		return total_loss / train_data.num_events


	@torch.no_grad()
	def test(self, inference_data):
	   
		self.memory.eval()
		self.gnn.eval()
		self.link_pred.eval()

		torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

		aps, aucs = [], []

		i = 0
		for batch in inference_data.seq_batches(batch_size=200):

			src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
			neg_dst = torch.randint(self.min_dst_idx, self.max_dst_idx + 1, (src.size(0), ),dtype=torch.long, device=self.device)
			all_nodes = torch.cat([src, pos_dst, neg_dst])

			for s,d,n in zip(src, pos_dst, neg_dst):
				
				self.adj_list[int(s.cpu())].add(int(d.cpu()))
				self.adj_list[int(d.cpu())].add(int(s.cpu()))

				adj_nodes_s = torch.tensor(list(self.adj_list[int(s.cpu())]), dtype=torch.long, device=self.device)
				adj_nodes_d = torch.tensor(list(self.adj_list[int(d.cpu())]), dtype=torch.long, device=self.device)
				adj_nodes_n = torch.tensor(list(self.adj_list[int(n.cpu())]), dtype=torch.long, device=self.device)
				
				all_nodes = torch.cat([all_nodes, adj_nodes_s, adj_nodes_d, adj_nodes_n])

			n_id = all_nodes.unique()
			n_id, edge_index, e_id = self.neighbor_loader(n_id)
			
			self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

			z, last_update = self.memory(n_id)
			z = self.gnn(z, last_update, edge_index, self.data.t[e_id], self.data.msg[e_id])
			
			z_adj_src = torch.empty(0, self.embedding_dim, dtype=torch.long, device=self.device)
			z_adj_dst = torch.empty(0, self.embedding_dim, dtype=torch.long, device=self.device)
			z_adj_neg = torch.empty(0, self.embedding_dim, dtype=torch.long, device=self.device)

			for s,d, n in zip(src, pos_dst, neg_dst):

				adj_nodes_s = torch.tensor(list(self.adj_list[int(s.cpu())])[-3:], dtype=torch.long, device=self.device)
				adj_nodes_d = torch.tensor(list(self.adj_list[int(d.cpu())])[-3:], dtype=torch.long, device=self.device)
				adj_nodes_n = torch.tensor(list(self.adj_list[int(n.cpu())])[-3:], dtype=torch.long, device=self.device)

				z_adjs = z[self.assoc[adj_nodes_s]]
				z_adjd = z[self.assoc[adj_nodes_d]]
				z_adjn = z[self.assoc[adj_nodes_n]]

				if self.assoc[adj_nodes_n].tolist() == []:
					z_adjn = torch.zeros(1, self.embedding_dim, dtype=torch.float32, device=self.device)

				z_adj_src = torch.cat((z_adj_src, z_adjs.mean(0, True)), 0)
				z_adj_dst = torch.cat((z_adj_dst, z_adjd.mean(0, True)), 0)
				z_adj_neg = torch.cat((z_adj_neg, z_adjn.mean(0, True)), 0)

			# print(z_adj_src.size(), z_adj_dst.size())

			pos_out = self.link_pred(z[self.assoc[src]], z[self.assoc[pos_dst]], z_adj_src, z_adj_dst)
			neg_out = self.link_pred(z[self.assoc[src]], z[self.assoc[neg_dst]], z_adj_src, z_adj_neg)

			y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
			y_true = torch.cat(
				[torch.ones(pos_out.size(0)),
				 torch.zeros(neg_out.size(0))], dim=0)

			aps.append(average_precision_score(y_true, y_pred))
			aucs.append(roc_auc_score(y_true, y_pred))

			self.memory.update_state(src, pos_dst, t, msg)
			self.neighbor_loader.insert(src, pos_dst)

		return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())