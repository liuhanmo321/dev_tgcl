# This is a class that takes the models of temporal gnns of this folder and add a classification head on top of it.
# give me a template code for this class.

import torch
import torch.nn as nn
from typing import List

import numpy as np

from copy import deepcopy

from .TGAT import TGAN
from .TGN import TGN
from .CIGNN import CIGNN

from .DyGLib.models.TGAT import TGAT
from .DyGLib.models.DyGFormer import DyGFormer
from .DyGLib.models.GraphMixer import GraphMixer

def get_base_model(args, neighbor_finder, node_features, edge_features):

    if args.model == 'TGAT':
        # time_feat_dim = node_features.shape[1]
        return TGAT(node_raw_features=node_features, edge_raw_features=edge_features, neighbor_sampler=neighbor_finder,
                        time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_attn_heads, dropout=args.dropout, device=args.device)
        
        # return TGAN(neighbor_finder, node_features, edge_features, device=args.device,
        #         attn_mode='prod', use_time='time', agg_method='attn', node_dim=None, time_dim=None,
        #         num_layers=args.num_layer, n_head=args.num_attn_heads, null_idx=0, num_heads=1, drop_out=args.dropout, seq_len=None)
    # elif args.model == 'TGN':
    #     return TGN(neighbor_finder=neighbor_finder, node_features=node_features, edge_features=edge_features, device=args.device, n_layers=args.num_layer,
    #                         n_heads=args.num_attn_heads, dropout=args.dropout, use_memory=True, forbidden_memory_update=False,
    #                         memory_update_at_start=True, 
    #                         message_dimension=128, memory_dimension=128, embedding_module_type="graph_attention",
    #                         message_function="identity",
    #                         mean_time_shift_src=args.time_shift['mean_time_shift_src'], std_time_shift_src=args.time_shift['std_time_shift_src'],
    #                         mean_time_shift_dst=args.time_shift['mean_time_shift_dst'], std_time_shift_dst=args.time_shift['std_time_shift_dst'], 
    #                         n_neighbors=args.num_neighbors, aggregator_type="last", memory_updater_type="gru",
    #                         use_destination_embedding_in_message=True,
    #                         use_source_embedding_in_message=True,
    #                         dyrep=False)
    # elif args.model == 'CIGNN':
    #     return CIGNN(args)
    elif args.model == 'GraphMixer':
        return GraphMixer(node_raw_features=node_features, edge_raw_features=edge_features, neighbor_sampler=neighbor_finder,
                                        time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
    elif args.model == 'DyGFormer':
        return DyGFormer(node_raw_features=node_features, edge_raw_features=edge_features, neighbor_sampler=neighbor_finder,
                                        time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                        num_layers=args.num_layers, num_heads=args.num_attn_heads, dropout=args.dropout,
                                        max_input_sequence_length=args.max_input_sequence_length, device=args.device)
    
    elif args.model == 'WordEmb':
        return WordEmb(node_raw_features=node_features, edge_raw_features=edge_features, neighbor_sampler=neighbor_finder,
                        time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_attn_heads, dropout=args.dropout, device=args.device)

    

# create a 2 layer MLP template for me.
class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(Predictor, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.layer1(x)))
        x = self.layer2(x)
        return x


class TimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output


class WordEmb(nn.Module):
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler,
                 time_feat_dim: int, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1, device: str = 'cpu'):
        super(WordEmb, self).__init__()
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.device = device
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.layers = Predictor(self.node_feat_dim + self.time_feat_dim, 6 * self.node_feat_dim, self.node_feat_dim, self.dropout)
        
    def compute_src_dst_node_temporal_embeddings(self, src_nodes, dst_nodes, edge_times):
        time_emb = self.time_encoder(timestamps=torch.zeros(edge_times.shape).unsqueeze(dim=1).to(self.device))
        time_emb = time_emb.squeeze(dim=1)
        # print(torch.cat((self.node_raw_features[src_nodes], time_emb), dim=1).shape)
        src_node_embeddings = self.layers(torch.cat((self.node_raw_features[src_nodes], time_emb), dim=1))
        dst_node_embeddings = self.layers(torch.cat((self.node_raw_features[dst_nodes], time_emb), dim=1))
        # print(src_node_embeddings.shape, dst_node_embeddings.shape)
        return src_node_embeddings, dst_node_embeddings
    
    def set_neighbor_sampler(self, neighbor_sampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

class TemporalGNNClassifier(nn.Module):
    def __init__(self, args, neighbor_finder, node_features, edge_features, src_label, dst_label):
        super(TemporalGNNClassifier, self).__init__()
        self.args = args

        self.src_label = torch.tensor(src_label).to(args.device)
        self.dst_label = torch.tensor(dst_label).to(args.device)

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

        # self.head_layer1 = nn.Linear(input_dim, args.head_hidden_dim)

        if self.multihead:
            # Create multiple MLP heads for each subset of classes
            self.head_layer = nn.ModuleList([Predictor(input_dim, args.head_hidden_dim, args.num_class_per_dataset, args.dropout) for i in range(args.num_datasets)])
            for head in self.head_layer:
                head.requires_grad_(False)
            # self.multihead_layer2 = nn.ModuleList([nn.Linear(args.head_hidden_dim, args.num_class_per_dataset) for i in range(args.num_datasets)])
        else:
            # Create a single MLP head for all classifications
            self.head_layer = Predictor(input_dim, args.head_hidden_dim, args.num_class, args.dropout)

        self.dropout = nn.Dropout(args.dropout)

        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.neighbor_finder = neighbor_finder

        self.eye = torch.tril(torch.ones((self.args.num_class, self.args.num_class))).bool().to(args.device)

    def get_embeddings(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors):
        if self.args.model == 'TGAT':
            src_embeddings = self.base_model.compute_node_temporal_embeddings(src_nodes, edge_times, self.args.num_layers, n_neighbors)
            if dst_nodes is not None:
                dst_embeddings = self.base_model.compute_node_temporal_embeddings(dst_nodes, edge_times, self.args.num_layers, n_neighbors)
            else:
                dst_embeddings = None
        if self.args.model == 'DyGFormer':
            src_embeddings, dst_embeddings = self.base_model.compute_src_dst_node_temporal_embeddings(src_nodes, dst_nodes, edge_times)
        if self.args.model == 'GraphMixer':
            src_embeddings, dst_embeddings = self.base_model.compute_src_dst_node_temporal_embeddings(src_nodes, dst_nodes,
                                                 edge_times, n_neighbors, self.args.time_gap)
        if self.args.model == 'WordEmb':
            src_embeddings, dst_embeddings = self.base_model.compute_src_dst_node_temporal_embeddings(src_nodes, dst_nodes, edge_times)
        
        return src_embeddings, dst_embeddings

    def forward(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx=None, src_avail_mask=None, dst_avail_mask=None, return_logits=False, return_emb_loss=False, return_logits_loss=False):

        # Get the embeddings
        if self.multihead:
            for ds_id in range(dataset_idx + 1):
                self.head_layer[ds_id].requires_grad_(True)

            cur_label_src = deepcopy(self.src_label[edges])
            cur_label_dst = deepcopy(self.dst_label[edges])

            # src_embeddings, dst_embeddings = self.get_embeddings(src_nodes, dst_nodes, edges, edge_times, n_neighbors, candidate_weights_dict=candidate_weights_dict)
            src_embeddings, dst_embeddings = self.get_embeddings(src_nodes, dst_nodes, edges, edge_times, n_neighbors)
            
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

            # src_outputs = self.dropout(torch.relu(self.head_layer1(src_input_features)))
            # dst_outputs = self.dropout(torch.relu(self.head_layer1(dst_input_features)))
            
            init_order = torch.arange(0, len(src_input_features)).to(self.args.device)
            src_order, dst_order = [], []
            src_preds, dst_preds = [], []

            for ds_id in range(self.num_heads):
                src_ds_mask = (cur_label_src >= ds_id * self.num_class_per_dataset) & (cur_label_src < (ds_id + 1) * self.num_class_per_dataset)
                dst_ds_mask = (cur_label_dst >= ds_id * self.num_class_per_dataset) & (cur_label_dst < (ds_id + 1) * self.num_class_per_dataset)

                src_order.append(init_order[src_ds_mask])
                dst_order.append(init_order[dst_ds_mask])

                cur_label_src[src_ds_mask] = cur_label_src[src_ds_mask] - ds_id * self.num_class_per_dataset
                cur_label_dst[dst_ds_mask] = cur_label_dst[dst_ds_mask] - ds_id * self.num_class_per_dataset

                # src_preds.append(self.dropout(torch.relu(self.head_layer[ds_id](src_input_features[src_ds_mask]))))
                # dst_preds.append(self.dropout(torch.relu(self.head_layer[ds_id](dst_input_features[dst_ds_mask]))))
                src_preds.append(self.head_layer[ds_id](src_input_features[src_ds_mask]))
                dst_preds.append(self.head_layer[ds_id](dst_input_features[dst_ds_mask]))

            src_preds = torch.cat(src_preds, dim=0)
            dst_preds = torch.cat(dst_preds, dim=0)

            # change back the order
            src_order = torch.cat(src_order, dim=0)
            dst_order = torch.cat(dst_order, dim=0)
            src_order = (len(src_input_features) - src_order - 1)[torch.arange(len(src_input_features)-1, -1, -1)]
            dst_order = (len(dst_input_features) - dst_order - 1)[torch.arange(len(dst_input_features)-1, -1, -1)]

            src_preds = src_preds[src_order]
            dst_preds = dst_preds[dst_order]

            if src_avail_mask is not None and dst_avail_mask is not None:
                src_preds = src_preds[src_avail_mask]
                dst_preds = dst_preds[dst_avail_mask]
                cur_label_src = cur_label_src[src_avail_mask]
                cur_label_dst = cur_label_dst[dst_avail_mask]
                src_embeddings = src_embeddings[src_avail_mask]
                dst_embeddings = dst_embeddings[dst_avail_mask]

            if return_logits:
                return src_preds, dst_preds

            loss_src = self.criterion(src_preds, cur_label_src)
            loss_dst = self.criterion(dst_preds, cur_label_dst)

            if return_emb_loss:
                return loss_src, loss_dst, src_embeddings, dst_embeddings

            # if src_avail_mask is not None and dst_avail_mask is not None:
            #     loss_src = loss_src[src_avail_mask]
            #     loss_dst = loss_dst[dst_avail_mask]

            loss = loss_src.mean() + loss_dst.mean()

            return loss
        else:
            cur_label_src = deepcopy(self.src_label[edges])
            cur_label_dst = deepcopy(self.dst_label[edges])

            src_embeddings, dst_embeddings = self.get_embeddings(src_nodes, dst_nodes, edges, edge_times, n_neighbors)
            
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
                
            src_preds = self.head_layer(src_input_features)
            dst_preds = self.head_layer(dst_input_features)

            mask = self.eye[(dataset_idx + 1) * self.num_class_per_dataset - 1]
            cur_dataset_len = (dataset_idx + 1) * self.num_class_per_dataset

            # src_preds = src_preds[:, mask]
            # dst_preds = dst_preds[:, mask]

            src_preds = src_preds[:, :cur_dataset_len]
            dst_preds = dst_preds[:, :cur_dataset_len]

            if src_avail_mask is not None and dst_avail_mask is not None:
                src_preds = src_preds[src_avail_mask]
                dst_preds = dst_preds[dst_avail_mask]
                cur_label_src = cur_label_src[src_avail_mask]
                cur_label_dst = cur_label_dst[dst_avail_mask]
                src_embeddings = src_embeddings[src_avail_mask]
                dst_embeddings = dst_embeddings[dst_avail_mask]

            if return_logits:
                return src_preds, dst_preds
            
            loss_src = self.criterion(src_preds, cur_label_src)
            loss_dst = self.criterion(dst_preds, cur_label_dst)

            if return_emb_loss:
                return loss_src, loss_dst, src_embeddings, dst_embeddings

            if return_logits_loss:
                return loss_src, loss_dst, src_preds, dst_preds
            
            if len(loss_src) == 0:
                return torch.tensor(0.0).to(self.args.device)
            else:
                loss = loss_src.mean() + loss_dst.mean()
                return loss



    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.base_model.set_neighbor_sampler(neighbor_finder)

    def reset_graph(self):
        self.base_model.reset_graph()

    def detach_memory(self):
        if self.args.method == 'OTGNet':
            self.base_model.detach_memory()
        else:
            return