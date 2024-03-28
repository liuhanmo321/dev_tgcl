import torch
from torch import nn
import numpy as np
from models.Backbone import TemporalGNNClassifier

# The following code is to initialize the class for finetune, which is a vanilla baseline in continual learning.
# Please generate a template code for me.

class Joint(nn.Module):
    def __init__(self, args, neighbor_finder, node_features, edge_features, src_label, dst_label):
        super(Joint, self).__init__()
        self.args = args

        if args.supervision == 'supervised':
            self.model = TemporalGNNClassifier(args, neighbor_finder, node_features, edge_features, src_label, dst_label)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)

    def forward(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx=None, src_avail_mask=None, dst_avail_mask=None):
        self.model.detach_memory()
        if self.args.task == 'nodecls':
            return self.forward_nodecls(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx)
        elif self.args.task == 'linkpred':
            return self.forward_linkpred(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx)

    def forward_linkpred(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx):
        return
    
    def forward_nodecls(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx=None, src_avail_mask=None, dst_avail_mask=None):

        data_dict = {}

        if self.args.supervision == 'supervised':
            loss = self.model(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx)

            data_dict['loss'] = loss.item()

            if self.args.l2_norm:
                l2_reg = torch.tensor(0.).to(self.args.device)
                for param in self.model.parameters():
                    if param.requires_grad:
                        l2_reg += torch.norm(param)
                loss += self.args.l2_weight * l2_reg

                data_dict['l2_reg'] = l2_reg.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        elif self.args.supervision == 'semi-supervised':
            return NotImplementedError
        
        return data_dict

    def set_neighbor_finder(self, neighbor_finder):
        self.model.set_neighbor_finder(neighbor_finder)

    def detach_memory(self):
        self.model.detach_memory()

    def end_epoch(self):
        self.scheduler.step()

    def get_acc(self, x, y):
        output = self.model(x)
        _, pred = torch.max(output, 1)
        correct = (pred == y).sum().item()
        return correct

    def get_logits(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx):
        src_logits = self.model(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx, return_logits=True)
        return src_logits
    
    def end_dataset(self, train_data, args):
        return
    
    def begin_task(self, args, data, task):
        mask = np.ones_like(data.src).astype(bool)
        return mask, mask, mask

    def set_features(self, node_features, edge_features):
        if node_features is not None:
            self.model.base_model.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(self.args.device)
        else:
            self.model.base_model.node_raw_features = None
        
        if edge_features is not None:
            self.model.base_model.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(self.args.device)
        else:
            self.model.base_model.edge_raw_features = None

    def reset_graph(self):
        return

    def back_up_memory(self):
        return

    def restore_memory(self, back_up):
        return

    def get_parameters(self):
        return self.model.parameters()

    def get_optimizer(self):
        return self.optimizer

    def get_scheduler(self):
        return self.scheduler

    def get_model(self):
        return self.model


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
        self.memory = []
        self.total_memory = None
        # self.memory_size = memory_size

    
    def update_memory(self, new_memory):
        if len(self.memory) == 0:
            self.memory.append(new_memory)
            self.total_memory = new_memory
        else:
            self.memory.append(new_memory)
            self.total_memory.add_data(new_memory)
        

    def get_data(self, size, mode='random'):
        if mode == 'random':
            idx = np.random.choice(len(self.total_memory.src), size, replace=False)
            return self.total_memory.src[idx], self.total_memory.dst[idx], self.total_memory.edge_idxs[idx], self.total_memory.timestamps[idx]
    

class Data:
    def __init__(
        self, src, dst, timestamps, edge_idxs, labels_src, labels_dst, induct_nodes=None
    ):
        self.src = src
        self.dst = dst

        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels_src = labels_src
        self.labels_dst = labels_dst

        self.n_interactions = len(src)
        self.unique_nodes = set(src) | set(dst)
        self.n_unique_nodes = len(self.unique_nodes)
        self.induct_nodes = induct_nodes

    def add_data(self, x):
        self.src = np.concatenate((self.src, x.src))
        self.dst = np.concatenate((self.dst, x.dst))

        self.timestamps = np.concatenate((self.timestamps, x.timestamps))
        self.edge_idxs = np.concatenate((self.edge_idxs, x.edge_idxs))
        self.labels_src = np.concatenate((self.labels_src, x.labels_src))
        self.labels_dst = np.concatenate((self.labels_dst, x.labels_dst))

        self.n_interactions = len(self.src)
        self.unique_nodes = set(self.src) | set(self.dst)
        self.n_unique_nodes = len(self.unique_nodes)