# This is a class that takes the models of temporal gnns of this folder and add a classification head on top of it.
# give me a template code for this class.

import torch
import torch.nn as nn
from typing import List

from copy import deepcopy

from .TGAT import TGAN
from .TGN import TGN
from .CIGNN import CIGNN

def get_base_model(args, neighbor_finder, node_features, edge_features):

    if args.model == 'TGAT':
        return TGAN(neighbor_finder, node_features, edge_features, device=args.device,
                attn_mode='prod', use_time='time', agg_method='attn', node_dim=None, time_dim=None,
                num_layers=args.num_layer, n_head=args.num_attn_heads, null_idx=0, num_heads=1, drop_out=args.dropout, seq_len=None)
    elif args.model == 'TGN':
        return TGN(neighbor_finder=neighbor_finder, node_features=node_features, edge_features=edge_features, device=args.device, n_layers=args.num_layer,
                            n_heads=args.num_attn_heads, dropout=args.dropout, use_memory=True, forbidden_memory_update=False,
                            memory_update_at_start=True, 
                            message_dimension=128, memory_dimension=128, embedding_module_type="graph_attention",
                            message_function="identity",
                            mean_time_shift_src=args.time_shift['mean_time_shift_src'], std_time_shift_src=args.time_shift['std_time_shift_src'],
                            mean_time_shift_dst=args.time_shift['mean_time_shift_dst'], std_time_shift_dst=args.time_shift['std_time_shift_dst'], 
                            n_neighbors=args.num_neighbors, aggregator_type="last", memory_updater_type="gru",
                            use_destination_embedding_in_message=True,
                            use_source_embedding_in_message=True,
                            dyrep=False)
    elif args.model == 'CIGNN':
        return CIGNN(args)
    


class TemporalGNNClassifier(nn.Module):
    def __init__(self, args, neighbor_finder, node_features, edge_features, src_label, dst_label):
        super(TemporalGNNClassifier, self).__init__()
        self.args = args

        self.src_label = torch.tensor(src_label).to(args.device)
        self.dst_label = torch.tensor(dst_label).to(args.device)
        # self.src_label = src_label
        # self.dst_label = dst_label

        self.base_model = get_base_model(args, neighbor_finder, node_features, edge_features)
        self.multihead = args.multihead

        if self.multihead:
            self.num_heads = args.num_datasets

        self.num_class_per_dataset = args.num_class_per_dataset

        if args.feature_type == 'embedded':
            input_dim = args.node_embedding_dim
        
        elif args.feature_type == 'raw':
            input_dim = args.node_init_dim

        elif args.feature_type == 'both':
            input_dim = args.node_embedding_dim + args.node_init_dim


        self.head_layer1 = nn.Linear(input_dim, args.head_hidden_dim)

        if self.multihead:
            # Create multiple MLP heads for each subset of classes
            self.multihead_layer2 = nn.ModuleList([nn.Linear(args.head_hidden_dim, args.num_class_per_dataset) for i in range(args.num_datasets)])
        else:
            # Create a single MLP head for all classifications
            self.head_layer2 = nn.Linear(args.head_hidden_dim, args.num_class)

        self.dropout = nn.Dropout(args.dropout)

        self.criterion = nn.CrossEntropyLoss()

    def get_embeddings(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors):
        return self.base_model.get_embeddings(src_nodes, dst_nodes, edges, edge_times, n_neighbors)
    

    def forward(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx=None, return_logits=False, candidate_weights_dict=None):

        # Get the embeddings
        if self.multihead:
            cur_label_src = deepcopy(self.src_label[edges])
            cur_label_dst = deepcopy(self.dst_label[edges])

            src_embeddings, dst_embeddings = self.base_model.get_embeddings(src_nodes, dst_nodes, edges, edge_times, n_neighbors, candidate_weights_dict=candidate_weights_dict)
            
            # Pass the embeddings through the MLP heads
            if self.args.feature_type == 'embedded':
                src_input_features = src_embeddings
                dst_input_features = dst_embeddings

            elif self.args.feature_type == 'raw':
                src_input_features = self.base_model.node_raw_features[src_nodes]
                dst_input_features = self.base_model.node_raw_features[dst_nodes]

            elif self.args.feature_type == 'both':
                src_input_features = torch.cat((src_embeddings, self.base_model.node_raw_features[src_nodes]), dim=1)
                dst_input_features = torch.cat((dst_embeddings, self.base_model.node_raw_features[dst_nodes]), dim=1)

            src_outputs = self.dropout(torch.relu(self.head_layer1(src_input_features)))
            dst_outputs = self.dropout(torch.relu(self.head_layer1(dst_input_features)))

            src_preds = torch.zeros((len(src_outputs), self.num_class_per_dataset)).to(self.args.device)
            dst_preds = torch.zeros((len(dst_outputs), self.num_class_per_dataset)).to(self.args.device)

            for ds_id in range(self.num_heads):
                src_ds_mask = (cur_label_src >= ds_id * self.num_class_per_dataset) & (cur_label_src < (ds_id + 1) * self.num_class_per_dataset)
                dst_ds_mask = (cur_label_dst >= ds_id * self.num_class_per_dataset) & (cur_label_dst < (ds_id + 1) * self.num_class_per_dataset)

                cur_label_src[src_ds_mask] = cur_label_src[src_ds_mask] - ds_id * self.args.num_class_per_dataset
                cur_label_dst[dst_ds_mask] = cur_label_dst[dst_ds_mask] - ds_id * self.args.num_class_per_dataset

                src_preds[src_ds_mask] += self.dropout(torch.relu(self.multihead_layer2[ds_id](src_outputs[src_ds_mask])))
                dst_preds[dst_ds_mask] += self.dropout(torch.relu(self.multihead_layer2[ds_id](dst_outputs[dst_ds_mask])))

                # if return_logits:
                #     src_preds[src_ds_mask] = src_outputs[src_ds_mask].detached()
                #     dst_preds[dst_ds_mask] = dst_outputs[dst_ds_mask].detached()
            if return_logits:
                return src_preds, dst_preds

            loss_src = self.criterion(src_outputs, cur_label_src)
            loss_dst = self.criterion(dst_outputs, cur_label_dst)

            loss = loss_src + loss_dst
            return loss
        else:
            embeddings = self.base_model.get_embeddings(src_nodes, dst_nodes, edges, edge_times, n_neighbors)
            # Pass the embeddings through the MLP head
            outputs = self.head_layer2(torch.relu(self.head_layer1(embeddings)))
            return outputs


    def set_neighbor_finder(self, neighbor_finder):
        self.base_model.set_neighbor_finder(neighbor_finder)

    def reset_graph(self):
        self.base_model.reset_graph()

    def detach_memory(self):
        if self.args.method == 'OTGNet':
            self.base_model.detach_memory()
        else:
            return