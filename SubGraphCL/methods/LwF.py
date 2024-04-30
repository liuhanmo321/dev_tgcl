import torch
from torch import nn

import numpy as np
from models.Backbone import TemporalGNNClassifier

# The following code is to initialize the class for finetune, which is a vanilla baseline in continual learning.
# Please generate a template code for me.

# the function for knowledge distillation based on learning without forgetting
def distillation_loss(old_logits, new_logits, T=2):
    old_soft_logits = torch.pow(old_logits, 1/T)
    old_soft_logits = old_soft_logits / torch.sum(old_soft_logits, dim=1, keepdim=True)

    new_soft_logits = torch.pow(new_logits, 1/T)
    new_soft_logits = new_soft_logits / torch.sum(new_soft_logits, dim=1, keepdim=True)

    distill_loss = - torch.sum(old_soft_logits * torch.log(new_soft_logits), dim=1).mean()
    return distill_loss


class LwF(nn.Module):
    def __init__(self, args, neighbor_finder, node_features, edge_features, src_label, dst_label):
        super(LwF, self).__init__()
        self.args = args

        if args.supervision == 'supervised':
            self.model = TemporalGNNClassifier(args, neighbor_finder, node_features, edge_features, src_label, dst_label)
        
        self.old_model = None

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
            # loss = self.model(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx)

            loss_src, loss_dst, new_src_logits, new_dst_logits = self.model(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx, return_logits_loss=True)

            loss = loss_src.mean() + loss_dst.mean()

            if dataset_idx > 0:
                self.old_model.eval()
                with torch.no_grad():
                    old_src_logits, old_dst_logits = self.old_model(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx, return_logits=True)
                
                # new_src_logits, new_dst_logits = self.model(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx, return_logits=True)

                loss += distillation_loss(old_src_logits, new_src_logits, T=self.args.temperature) + distillation_loss(old_dst_logits, new_dst_logits, T=self.args.temperature)

            data_dict['loss'] = loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        elif self.args.supervision == 'semi-supervised':
            return NotImplementedError
        
        return data_dict

    def set_neighbor_finder(self, neighbor_finder):
        self.model.set_neighbor_finder(neighbor_finder)
        if self.old_model is not None:
            self.old_model.set_neighbor_finder(neighbor_finder)

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
        return

    def begin_task(self, args, data, task):
        visible_class = [task * args.num_class_per_dataset + i for i in range(args.num_class_per_dataset)]
        src_mask = np.isin(data.labels_src, visible_class)
        dst_mask = np.isin(data.labels_dst, visible_class)
        mask = src_mask | dst_mask
        return mask, src_mask, dst_mask

    def set_features(self, node_features, edge_features):
        if node_features is not None:
            self.model.base_model.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(self.args.device)
        else:
            self.model.base_model.node_raw_features = None
        
        if edge_features is not None:
            self.model.base_model.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(self.args.device)
        else:
            self.model.base_model.edge_raw_features = None

    def set_class_weight(self, class_weight):
        self.model.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight).float().to(self.args.device), reduction='none')

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