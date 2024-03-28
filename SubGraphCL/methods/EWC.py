import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.Backbone import TemporalGNNClassifier

from copy import deepcopy
import math

# The following code is to initialize the class for finetune, which is a vanilla baseline in continual learning.
# Please generate a template code for me.

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class EWC_loss(nn.Module):
    def __init__(self, model: nn.Module, dataset: list, args, dataset_idx):
        super(EWC_loss, self).__init__()
        self.model = model
        self.device = args.device
        self.dataset = dataset
        self.args = args
        self.dataset_idx = dataset_idx

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data).to(self.device)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data).to(self.device)

        self.model.eval()
        num_batch = math.ceil(len(self.dataset.src) / self.args.batch_size)
        for i in range(num_batch):
            st_idx = i * self.args.batch_size
            ed_idx = min((i + 1) * self.args.batch_size, len(self.dataset.src))

            src_batch = self.dataset.src[st_idx:ed_idx]
            dst_batch = self.dataset.dst[st_idx:ed_idx]
            edge_batch = self.dataset.edge_idxs[st_idx:ed_idx]
            timestamp_batch = self.dataset.timestamps[st_idx:ed_idx]

            loss = self.model(src_batch, dst_batch, edge_batch, timestamp_batch, self.args.num_neighbors, self.dataset_idx)

            loss.backward()

            for n, p in self.model.named_parameters():
                # print(n, p.requires_grad)
                if p.grad is not None:
                    precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset.src)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            if p.grad is not None:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss


class EWC(nn.Module):
    def __init__(self, args, neighbor_finder, node_features, edge_features, src_label, dst_label):
        super(EWC, self).__init__()
        self.args = args

        if args.supervision == 'supervised':
            self.model = TemporalGNNClassifier(args, neighbor_finder, node_features, edge_features, src_label, dst_label)
        
        self.ewc_loss = None

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)

        self.dataset_idx = None

    def forward(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx=None):
        self.model.detach_memory()
        if self.args.task == 'nodecls':
            return self.forward_nodecls(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx)
        elif self.args.task == 'linkpred':
            return self.forward_linkpred(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx)

    def forward_linkpred(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx):
        return
    
    def forward_nodecls(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx=None):
        
        self.dataset_idx = dataset_idx

        data_dict = {}

        if self.args.supervision == 'supervised':
            loss = self.model(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx)

            if dataset_idx > 0:
                ewc_loss = self.ewc_loss.penalty(self.model) * self.args.ewc_weight
                loss += ewc_loss
                data_dict['ewc_loss'] = ewc_loss.item()

            data_dict['loss'] = loss.item()

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
        src_logits, dst_logits = self.model(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx, return_logits=True)
        return src_logits, dst_logits
    
    def end_dataset(self, train_data, args):
        self.ewc_loss = EWC_loss(self.model, train_data, args, self.dataset_idx)
        return

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