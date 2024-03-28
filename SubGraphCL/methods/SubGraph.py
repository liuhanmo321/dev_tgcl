import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

import math
from models.Backbone import TemporalGNNClassifier
from copy import deepcopy
# from methods.SubGraph_utils import local_rbf_kernel, select_prototypes
# The following code is to initialize the class for finetune, which is a vanilla baseline in continual learning.
# Please generate a template code for me.


class ResidualDistiller(nn.Module):
    def __init__(self, args):
        super(ResidualDistiller, self).__init__()
        self.args = args

        self.residual_predictor = nn.Sequential(
            nn.Linear(args.node_embedding_dim, args.num_neighbors),
            nn.Tanh()
        )

        self.dist_predictor = nn.Sequential(
            # nn.BatchNorm1d(args.num_neighbors * 2),
            # nn.ReLU(),
            nn.Linear(args.num_neighbors * 2, args.num_neighbors),
            nn.Tanh()
            # nn.BatchNorm1d(args.num_neighbors),
            # nn.Tanh(),
            # nn.Linear(args.num_neighbors, args.num_neighbors)
        )

        # self.residual_projector_optimizer = torch.optim.Adam(self.residual_projector.parameters(), lr=args.lr)

    def forward(self, new_embedding, new_similarity):
        new_embedding = new_embedding.view(new_embedding.shape[0], -1)
        residual = self.residual_predictor(new_embedding)
        
        cat_new_similarity = torch.cat((new_similarity, residual), dim=-1)

        act_new_residual = self.dist_predictor(cat_new_similarity)

        return act_new_residual

class SubGraph(nn.Module):
    def __init__(self, args, neighbor_finder, node_features, edge_features, src_label, dst_label):
        super(SubGraph, self).__init__()
        self.args = args

        if args.supervision == 'supervised':
            self.model = TemporalGNNClassifier(args, neighbor_finder, node_features, edge_features, src_label, dst_label)
        elif self.args.supervision == 'semi-supervised':
            return NotImplementedError
        
        self.memory = Memory()
        self.new_memory = Memory()
        self.loss_bank = []
        self.old_model = None
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)

        if args.rand_neighbor:
            self.rand_sampler = None

        if args.emb_proj:
            self.emb_projector = nn.Sequential(
                nn.Linear(args.node_embedding_dim, args.node_embedding_dim),
                nn.ReLU(),
                nn.Linear(args.node_embedding_dim, args.node_embedding_dim)
            )
            # self.emb_projector = nn.Linear(args.node_embedding_dim, args.node_embedding_dim)
            self.emb_projector_optimizer = torch.optim.Adam(self.emb_projector.parameters(), lr=args.lr)
        
        if args.emb_residual:
            self.emb_residual_projector = nn.Sequential(
                nn.BatchNorm1d(args.node_embedding_dim),
                nn.ReLU(),
                nn.Linear(args.node_embedding_dim, args.node_embedding_dim),
                nn.BatchNorm1d(args.node_embedding_dim),
                nn.ReLU(),
                nn.Linear(args.node_embedding_dim, args.node_embedding_dim)
            )
            # self.emb_projector = nn.Linear(args.node_embedding_dim, args.node_embedding_dim)
            self.emb_residual_projector_optimizer = torch.optim.Adam(self.emb_residual_projector.parameters(), lr=args.lr)
        
        if args.residual_distill:
            self.residual_projector = ResidualDistiller(args)
            # self.residual_projector = nn.Sequential(
            #     nn.BatchNorm1d(args.num_neighbors),
            #     # nn.ReLU(),
            #     nn.Linear(args.num_neighbors, args.num_neighbors),
            #     nn.BatchNorm1d(args.num_neighbors),
            #     nn.ReLU(),
            #     nn.Linear(args.num_neighbors, args.num_neighbors)
            # )

            self.residual_projector_optimizer = torch.optim.Adam(self.residual_projector.parameters(), lr=args.lr)

    def forward(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, is_old_data=False, dataset_idx=None, src_avail_mask=None, dst_avail_mask=None):
        self.model.detach_memory()

        if self.args.task == 'nodecls':
            return self.forward_nodecls(src_nodes, dst_nodes, edges, edge_times, n_neighbors, is_old_data, dataset_idx, src_avail_mask, dst_avail_mask)
        elif self.args.task == 'linkpred':
            return self.forward_linkpred(src_nodes, dst_nodes, edges, edge_times, n_neighbors, is_old_data, dataset_idx)

    def forward_linkpred(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx):
        return
    
    def forward_nodecls(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, is_old_data, dataset_idx=None, src_avail_mask=None, dst_avail_mask=None):
        
        # if self.args.memory_replay and dataset_idx > 0:
        #     memory_src_nodes, memory_dst_nodes, memory_edge_idxs, memory_timestamps = self.memory.get_data(self.args.batch_size, mode=self.args.replay_select_mode)
            
        data_dict = {}

        if self.args.supervision == 'supervised':
            if self.old_model is not None:
                self.old_model.eval()

            if is_old_data:
                # print("old data replayed")
                loss = self.model(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx)
                
                if self.args.distill:
                    with torch.no_grad():
                        old_src_emb, old_dst_emb = self.old_model.get_embeddings(src_nodes, dst_nodes, edges, edge_times, n_neighbors)
                    
                    new_src_emb, new_dst_emb = self.model.get_embeddings(src_nodes, dst_nodes, edges, edge_times, n_neighbors)

                    if self.args.old_emb_distribution_distill:
                        # print("distill old")
                        total_new_emb, total_old_emb = torch.cat([new_src_emb, new_dst_emb], dim=0), torch.cat([old_src_emb, old_dst_emb], dim=0)
                        old_distribution_kernel = rbf_kernel(torch.cat([total_new_emb, total_old_emb], dim=0), gamma=self.args.reg_gamma)
                        n, m = len(total_new_emb), len(total_old_emb)
                        XX = old_distribution_kernel[:n, :n]
                        YY = old_distribution_kernel[n:, n:]
                        XY = old_distribution_kernel[:n, n:]
                        YX = old_distribution_kernel[n:, :n]

                        XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  # Source<->Source
                        XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) # Source<->Target

                        YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) #Target<->Source
                        YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  # Target<->Target
                            
                        old_distribution_loss = (XX + XY).sum() + (YX + YY).sum()
                        # old_distribution_loss = torch.mean(XX + YY - XY -YX)
                        loss += old_distribution_loss * self.args.emb_distribution_distill_weight
                        data_dict['old_distribution_loss'] = old_distribution_loss.item()
                    
                    if self.args.new_emb_distribution_distill:
                        print("distill new")
                        temp_src_nodes, temp_dst_nodes, temp_edges, temp_edge_times = self.new_memory.get_data(self.args.batch_size)
                        temp_new_src_emb, temp_new_dst_emb = self.model.get_embeddings(temp_src_nodes, temp_dst_nodes, temp_edges, temp_edge_times, n_neighbors)
                        
                        total_new_emb, total_temp_new_emb = torch.cat([new_src_emb, new_dst_emb], dim=0), torch.cat([temp_new_src_emb, temp_new_dst_emb], dim=0)

                        new_distribution_kernel = rbf_kernel(torch.cat([total_new_emb, total_temp_new_emb], dim=0), gamma=self.args.reg_gamma)
                        n, m = len(total_new_emb), len(total_temp_new_emb)
                        XX = new_distribution_kernel[:n, :n]
                        YY = new_distribution_kernel[n:, n:]
                        XY = new_distribution_kernel[:n, n:]
                        YX = new_distribution_kernel[n:, :n]

                        XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  # Source<->Source
                        XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) # Source<->Target

                        YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) #Target<->Source
                        YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  # Target<->Target
                            
                        new_distribution_loss = (XX + XY).sum() + (YX + YY).sum()
                        loss += new_distribution_loss * self.args.emb_distribution_distill_weight
                        data_dict['new_distribution_loss'] = new_distribution_loss.item()

                    if self.args.emb_proj:
                        new_src_emb = self.emb_projector(new_src_emb)
                        new_dst_emb = self.emb_projector(new_dst_emb)
                    
                    if self.args.emb_distill:
                        emb_distill_loss = - self.similarity(new_src_emb, old_emb=old_src_emb, weight=None) - self.similarity(new_dst_emb, old_emb=old_dst_emb, weight=None)
                        emb_distill_loss = emb_distill_loss / 2

                        loss += emb_distill_loss * self.args.emb_distill_weight
                        data_dict['emb_distill_loss'] = emb_distill_loss.item()

                    if self.args.struct_distill:
                        if self.args.rand_neighbor:
                            src_rand_neighbors, dst_rand_neighbors = [], []
                            for _ in range(n_neighbors):
                                sample_results = self.rand_sampler.sample(len(src_nodes))
                                src_rand_neighbors.append(sample_results[0]), dst_rand_neighbors.append(sample_results[1])
                            src_ngh_node_batch, dst_ngh_node_batch = np.array(src_rand_neighbors).T, np.array(dst_rand_neighbors).T

                            print(src_ngh_node_batch.shape)
                            tmp_edge_times = np.repeat(edge_times.reshape(-1, 1), n_neighbors, axis=-1)

                            src_ngh_mask = src_ngh_node_batch == 0
                            dst_ngh_mask = dst_ngh_node_batch == 0

                            src_ngh_node_batch_flat = src_ngh_node_batch.flatten().astype('int64')  #reshape(batch_size, -1)
                            src_ngh_t_batch_flat = tmp_edge_times.flatten().astype('int64')  #reshape(batch_size, -1)
                            dst_ngh_node_batch_flat = dst_ngh_node_batch.flatten().astype('int64')  #reshape(batch_size, -1)
                            dst_ngh_t_batch_flat = tmp_edge_times.flatten().astype('int64')  #reshape(batch_size, -1)
                            
                            # print("source neightbor node batch flat shape:", src_ngh_node_batch_flat.shape)
                            with torch.no_grad():
                                old_src_ngh_embs, _ = self.old_model.get_embeddings(src_ngh_node_batch_flat, None, None, src_ngh_t_batch_flat, n_neighbors)
                                old_dst_ngh_embs, _ = self.old_model.get_embeddings(dst_ngh_node_batch_flat, None, None, dst_ngh_t_batch_flat, n_neighbors)

                            new_src_ngh_embs, _ = self.model.get_embeddings(src_ngh_node_batch_flat, None, None, src_ngh_t_batch_flat, n_neighbors)
                            new_dst_ngh_embs, _ = self.model.get_embeddings(dst_ngh_node_batch_flat, None, None, dst_ngh_t_batch_flat, n_neighbors)

                        else:
                            src_ngh_node_batch, src_ngh_edge_batch, src_ngh_t_batch = self.model.neighbor_finder.get_temporal_neighbor(src_nodes, edge_times, n_neighbors, find_future=False)
                            dst_ngh_node_batch, dst_ngh_edge_batch, dst_ngh_t_batch = self.model.neighbor_finder.get_temporal_neighbor(dst_nodes, edge_times, n_neighbors, find_future=False)
                            
                            src_ngh_mask = src_ngh_node_batch == 0
                            dst_ngh_mask = dst_ngh_node_batch == 0

                            src_ngh_node_batch_flat = src_ngh_node_batch.flatten().astype('int64')  #reshape(batch_size, -1)
                            src_ngh_t_batch_flat = src_ngh_t_batch.flatten().astype('int64')  #reshape(batch_size, -1)
                            dst_ngh_node_batch_flat = dst_ngh_node_batch.flatten().astype('int64')  #reshape(batch_size, -1)
                            dst_ngh_t_batch_flat = dst_ngh_t_batch.flatten().astype('int64')  #reshape(batch_size, -1)
                            
                            with torch.no_grad():
                                old_src_ngh_embs, _ = self.old_model.get_embeddings(src_ngh_node_batch_flat, None, src_ngh_edge_batch, src_ngh_t_batch_flat, n_neighbors)
                                old_dst_ngh_embs, _ = self.old_model.get_embeddings(dst_ngh_node_batch_flat, None, dst_ngh_edge_batch, dst_ngh_t_batch_flat, n_neighbors)

                            new_src_ngh_embs, _ = self.model.get_embeddings(src_ngh_node_batch_flat, None, src_ngh_edge_batch, src_ngh_t_batch_flat, n_neighbors)
                            new_dst_ngh_embs, _ = self.model.get_embeddings(dst_ngh_node_batch_flat, None, dst_ngh_edge_batch, dst_ngh_t_batch_flat, n_neighbors)


                        if self.args.emb_proj:
                            new_src_ngh_embs = self.emb_projector(new_src_ngh_embs)
                            new_dst_ngh_embs = self.emb_projector(new_dst_ngh_embs)

                        old_src_ngh_embs = old_src_ngh_embs.view(src_ngh_node_batch.shape[0], n_neighbors, -1)
                        old_dst_ngh_embs = old_dst_ngh_embs.view(dst_ngh_node_batch.shape[0], n_neighbors, -1)
                        new_src_ngh_embs = new_src_ngh_embs.view(src_ngh_node_batch.shape[0], n_neighbors, -1)
                        new_dst_ngh_embs = new_dst_ngh_embs.view(dst_ngh_node_batch.shape[0], n_neighbors, -1)

                        old_src_emb = old_src_emb.view(old_src_emb.shape[0], 1, -1)
                        old_dst_emb = old_dst_emb.view(old_dst_emb.shape[0], 1, -1)
                        new_src_emb = new_src_emb.view(new_src_emb.shape[0], 1, -1)
                        new_dst_emb = new_dst_emb.view(new_dst_emb.shape[0], 1, -1)
                        
                        old_src_similarity = self.similarity(old_src_emb, ngh_emb=old_src_ngh_embs, mask=src_ngh_mask)
                        old_dst_similarity = self.similarity(old_dst_emb, ngh_emb=old_dst_ngh_embs, mask=dst_ngh_mask)
                        new_src_similarity = self.similarity(new_src_emb, ngh_emb=new_src_ngh_embs, mask=src_ngh_mask)
                        new_dst_similarity = self.similarity(new_dst_emb, ngh_emb=new_dst_ngh_embs, mask=dst_ngh_mask)

                        if self.args.residual_distill:
                            # print("using redisual distill")
                            new_src_residual = self.residual_projector(new_src_emb, new_src_similarity)
                            new_dst_residual = self.residual_projector(new_dst_emb, new_dst_similarity)

                            new_src_similarity = new_src_similarity + new_src_residual
                            new_dst_similarity = new_dst_similarity + new_dst_residual

                        # The order of KLDivLoss can be reversed
                        src_struct_loss = self.distribution_difference(new_src_similarity, old_src_similarity)
                        dst_struct_loss = self.distribution_difference(new_dst_similarity, old_dst_similarity)
                        
                        struct_distill_loss = (src_struct_loss.mean() + dst_struct_loss.mean()) / 2

                        loss += struct_distill_loss * self.args.struct_distill_weight

                        data_dict['struct_distill_loss'] = struct_distill_loss.item()

                loss = loss * self.args.old_data_weight
                data_dict['loss'] = loss.item()
            else:
                loss = self.model(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx, src_avail_mask, dst_avail_mask)
                data_dict['loss'] = loss.item()
            
            if self.args.l2_norm:
                l2_reg = torch.tensor(0.).to(self.args.device)
                for param in self.model.parameters():
                    if param.requires_grad:
                        l2_reg += torch.norm(param)
                loss += self.args.l2_weight * l2_reg

                data_dict['l2_reg'] = l2_reg.item() * self.args.l2_weight

            self.optimizer.zero_grad()
            if self.args.distill and self.args.emb_proj:
                self.emb_projector_optimizer.zero_grad()
            if self.args.distill and self.args.residual_distill:
                self.residual_projector_optimizer.zero_grad()
            
            loss.backward()
            
            self.optimizer.step()
            if self.args.distill and self.args.emb_proj:
                self.emb_projector_optimizer.step()
            if self.args.distill and self.args.residual_distill:
                self.residual_projector_optimizer.step()
            
        elif self.args.supervision == 'semi-supervised':
            return NotImplementedError
        
        return data_dict

    def set_neighbor_finder(self, neighbor_finder):
        self.model.set_neighbor_finder(neighbor_finder)
    
    def set_sampler(self, rand_sampler):
        self.rand_sampler = rand_sampler

    def detach_memory(self):
        self.model.detach_memory()

    def end_epoch(self):
        self.scheduler.step()

    def get_logits(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx):
        src_logits, dst_logits = self.model(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx, return_logits=True)
        return src_logits, dst_logits
    
    def end_dataset(self, train_data, args, val_data=None):
        if not args.memory_replay:
            return
        elif args.select_mode == 'random':
            sampled_idx = np.random.choice(len(train_data.src), args.memory_size, replace=False)
            cur_memory = Data(train_data.src[sampled_idx], train_data.dst[sampled_idx], train_data.timestamps[sampled_idx], train_data.edge_idxs[sampled_idx], train_data.labels_src[sampled_idx], train_data.labels_dst[sampled_idx])
            self.memory.update_memory(cur_memory)

        elif args.select_mode == 'mean_emb':
            src_emb_bank = []
            dst_emb_bank = []
            with torch.no_grad():
                num_batch = math.ceil(len(train_data.src) / args.batch_size)

                for i in range(num_batch):

                    st_idx = i * args.batch_size
                    ed_idx = min((i + 1) * args.batch_size, len(train_data.src))

                    src_batch = train_data.src[st_idx:ed_idx]
                    dst_batch = train_data.dst[st_idx:ed_idx]
                    edge_batch = train_data.edge_idxs[st_idx:ed_idx]
                    timestamp_batch = train_data.timestamps[st_idx:ed_idx]

                    src_embeddings, dst_embeddings = self.model.get_embeddings(src_batch, dst_batch, edge_batch, timestamp_batch, args.num_neighbors)
                    src_emb_bank.append(src_embeddings)
                    dst_emb_bank.append(dst_embeddings)
                
                src_emb_bank = torch.cat(src_emb_bank, dim=0)
                dst_emb_bank = torch.cat(dst_emb_bank, dim=0)
            
            seen_task = len(self.memory.memory)
            cur_memory = None
            for c in range(seen_task * self.args.num_class_per_dataset, (seen_task + 1) * self.args.num_class_per_dataset):
                task_mp_src = torch.tensor([False for i in range(len(src_emb_bank))])
                task_mp_dst = torch.tensor([False for i in range(len(dst_emb_bank))])  
                task_mp_src = task_mp_src | (torch.tensor(train_data.labels_src) == c)
                task_mp_dst = task_mp_dst | (torch.tensor(train_data.labels_dst) == c)

                task_mp = task_mp_src | task_mp_dst

                total_emb_bank = torch.cat([src_emb_bank[task_mp_src], dst_emb_bank[task_mp_dst]], dim=0)
                mean_emb = torch.mean(total_emb_bank)

                dist_src = -torch.nn.CosineSimilarity()(src_emb_bank[task_mp], mean_emb)
                dist_dst = -torch.nn.CosineSimilarity()(dst_emb_bank[task_mp], mean_emb)

                total_dist = dist_src + dist_dst

                memory_size_class = int(args.memory_size / self.args.num_class_per_dataset)
                memory_size_class = min(memory_size_class, len(total_dist))
                _, sampled_idx_class = torch.sort(total_dist, stable=True)
                sampled_idx_class = sampled_idx_class[:memory_size_class].cpu()

                if cur_memory == None:
                    cur_memory = Data(train_data.src[task_mp][sampled_idx_class], train_data.dst[task_mp][sampled_idx_class], \
                                      train_data.timestamps[task_mp][sampled_idx_class], train_data.edge_idxs[task_mp][sampled_idx_class], \
                                      train_data.labels_src[task_mp][sampled_idx_class], train_data.labels_dst[task_mp][sampled_idx_class])
                else:
                    cur_memory.add_data(Data(train_data.src[task_mp][sampled_idx_class], train_data.dst[task_mp][sampled_idx_class], \
                                      train_data.timestamps[task_mp][sampled_idx_class], train_data.edge_idxs[task_mp][sampled_idx_class], \
                                      train_data.labels_src[task_mp][sampled_idx_class], train_data.labels_dst[task_mp][sampled_idx_class]))

            self.memory.update_memory(cur_memory)
            
        elif self.args.select_mode == 'mean_pred':
            src_pred_bank = []
            dst_pred_bank = []

            seen_task = len(self.memory.memory)

            with torch.no_grad():
                num_batch = math.ceil(len(train_data.src) / args.batch_size)

                for i in range(num_batch):

                    st_idx = i * args.batch_size
                    ed_idx = min((i + 1) * args.batch_size, len(train_data.src))

                    src_batch = train_data.src[st_idx:ed_idx]
                    dst_batch = train_data.dst[st_idx:ed_idx]
                    edge_batch = train_data.edge_idxs[st_idx:ed_idx]
                    timestamp_batch = train_data.timestamps[st_idx:ed_idx]

                    src_preds, dst_preds = self.model(src_batch, dst_batch, edge_batch, timestamp_batch, args.num_neighbors, seen_task, return_logits=True)
                    src_pred_bank.append(src_preds)
                    dst_pred_bank.append(dst_preds)
                
                src_pred_bank = torch.cat(src_pred_bank, dim=0)
                dst_pred_bank = torch.cat(dst_pred_bank, dim=0)
            
            cur_memory = None
            for c in range(seen_task * self.args.num_class_per_dataset, (seen_task + 1) * self.args.num_class_per_dataset):
                # current_class = c - seen_task * self.args.num_class_per_dataset
                task_mp_src = torch.tensor([False for i in range(len(src_pred_bank))])
                task_mp_dst = torch.tensor([False for i in range(len(dst_pred_bank))])
                task_mp_src = task_mp_src | (torch.tensor(train_data.labels_src) == c)
                task_mp_dst = task_mp_dst | (torch.tensor(train_data.labels_dst) == c)

                task_mp = task_mp_src | task_mp_dst

                total_pred_bank = torch.cat([src_pred_bank[task_mp_src], dst_pred_bank[task_mp_dst]], dim=0)
                mean_pred = torch.mean(total_pred_bank, dim=0)

                src_dist = (src_pred_bank[task_mp] - mean_pred).abs().sum(1)
                dst_dist = (dst_pred_bank[task_mp] - mean_pred).abs().sum(1)
                
                total_dist = src_dist + dst_dist

                memory_size_class = int(args.memory_size / self.args.num_class_per_dataset)
                memory_size_class = min(memory_size_class, len(total_dist))
                _, sampled_idx_class = torch.sort(total_dist, stable=True)
                sampled_idx_class = sampled_idx_class[:memory_size_class].cpu()

                print(sampled_idx_class.shape)

                if cur_memory == None:
                    cur_memory = Data(train_data.src[task_mp][sampled_idx_class], train_data.dst[task_mp][sampled_idx_class], \
                                      train_data.timestamps[task_mp][sampled_idx_class], train_data.edge_idxs[task_mp][sampled_idx_class], \
                                      train_data.labels_src[task_mp][sampled_idx_class], train_data.labels_dst[task_mp][sampled_idx_class])
                else:
                    cur_memory.add_data(Data(train_data.src[task_mp][sampled_idx_class], train_data.dst[task_mp][sampled_idx_class], \
                                      train_data.timestamps[task_mp][sampled_idx_class], train_data.edge_idxs[task_mp][sampled_idx_class], \
                                      train_data.labels_src[task_mp][sampled_idx_class], train_data.labels_dst[task_mp][sampled_idx_class]))

            self.memory.update_memory(cur_memory)
    
    def begin_task(self, args, train_data, task):
        new_class = [task * args.num_class_per_dataset + i for i in range(args.num_class_per_dataset)]
        new_src_mask = np.isin(train_data.labels_src, new_class)
        new_dst_mask = np.isin(train_data.labels_dst, new_class)
        new_mask = new_src_mask | new_dst_mask

        old_class = [i for i in range(task * args.num_class_per_dataset)]
        past_src_mask = np.isin(train_data.labels_src, old_class)
        past_dst_mask = np.isin(train_data.labels_dst, old_class)
        past_mask = past_src_mask | past_dst_mask

        past_train_data = deepcopy(train_data)
        past_train_data.apply_mask(past_mask, past_src_mask, past_dst_mask)

        if not args.memory_replay:
            return new_mask, new_src_mask, new_dst_mask
        
        if args.select_mode == 'random':
            
            # sampled_idx = np.random.choice(len(past_train_data.src), args.memory_size, replace=False)
            # cur_memory = Data(past_train_data.src[sampled_idx], past_train_data.dst[sampled_idx], past_train_data.timestamps[sampled_idx], past_train_data.edge_idxs[sampled_idx], past_train_data.labels_src[sampled_idx], past_train_data.labels_dst[sampled_idx])
            # self.memory.update_memory(cur_memory)

            cur_memory = None
            for c in range(task * self.args.num_class_per_dataset):
                task_mp_src = torch.tensor([False for i in range(len(past_train_data.src))])
                task_mp_dst = torch.tensor([False for i in range(len(past_train_data.src))])  
                task_mp_src = task_mp_src | (torch.tensor(past_train_data.labels_src) == c)
                task_mp_dst = task_mp_dst | (torch.tensor(past_train_data.labels_dst) == c)

                task_mp = task_mp_src | task_mp_dst

                sampled_idx_class = np.random.choice(len(past_train_data.src[task_mp]), args.memory_size, replace=False)

                if cur_memory == None:
                    cur_memory = Data(past_train_data.src[task_mp][sampled_idx_class], past_train_data.dst[task_mp][sampled_idx_class], \
                                      past_train_data.timestamps[task_mp][sampled_idx_class], past_train_data.edge_idxs[task_mp][sampled_idx_class], \
                                      past_train_data.labels_src[task_mp][sampled_idx_class], past_train_data.labels_dst[task_mp][sampled_idx_class])
                else:
                    cur_memory.add_data(Data(past_train_data.src[task_mp][sampled_idx_class], past_train_data.dst[task_mp][sampled_idx_class], \
                                      past_train_data.timestamps[task_mp][sampled_idx_class], past_train_data.edge_idxs[task_mp][sampled_idx_class], \
                                      past_train_data.labels_src[task_mp][sampled_idx_class], past_train_data.labels_dst[task_mp][sampled_idx_class]))
            
            self.memory = Memory()
            self.memory.update_memory(cur_memory)

            return new_mask, new_src_mask, new_dst_mask

        if args.select_mode == 'mean_emb':
            src_emb_bank = []
            dst_emb_bank = []
            with torch.no_grad():
                num_batch = math.ceil(len(past_train_data.src) / args.batch_size)

                for i in range(num_batch):

                    st_idx = i * args.batch_size
                    ed_idx = min((i + 1) * args.batch_size, len(past_train_data.src))

                    src_batch = past_train_data.src[st_idx:ed_idx]
                    dst_batch = past_train_data.dst[st_idx:ed_idx]
                    edge_batch = past_train_data.edge_idxs[st_idx:ed_idx]
                    timestamp_batch = past_train_data.timestamps[st_idx:ed_idx]

                    src_embeddings, dst_embeddings = self.model.get_embeddings(src_batch, dst_batch, edge_batch, timestamp_batch, args.num_neighbors)
                    src_emb_bank.append(src_embeddings)
                    dst_emb_bank.append(dst_embeddings)
                
                src_emb_bank = torch.cat(src_emb_bank, dim=0)
                dst_emb_bank = torch.cat(dst_emb_bank, dim=0)
            
            cur_memory = None
            for c in range(task * self.args.num_class_per_dataset):
                task_mp_src = torch.tensor([False for i in range(len(src_emb_bank))])
                task_mp_dst = torch.tensor([False for i in range(len(dst_emb_bank))])  
                task_mp_src = task_mp_src | (torch.tensor(past_train_data.labels_src) == c)
                task_mp_dst = task_mp_dst | (torch.tensor(past_train_data.labels_dst) == c)

                task_mp = task_mp_src | task_mp_dst

                total_emb_bank = torch.cat([src_emb_bank[task_mp_src], dst_emb_bank[task_mp_dst]], dim=0)
                mean_emb = torch.mean(total_emb_bank)

                dist_src = -torch.nn.CosineSimilarity()(src_emb_bank[task_mp], mean_emb)
                dist_dst = -torch.nn.CosineSimilarity()(dst_emb_bank[task_mp], mean_emb)

                total_dist = dist_src + dist_dst

                # memory_size_class = int(args.memory_size / self.args.num_class_per_dataset)
                # memory_size_class = min(memory_size_class, len(total_dist))
                _, sampled_idx_class = torch.sort(total_dist, stable=True)
                sampled_idx_class = sampled_idx_class[:args.memory_size].cpu()

                if cur_memory == None:
                    cur_memory = Data(past_train_data.src[task_mp][sampled_idx_class], past_train_data.dst[task_mp][sampled_idx_class], \
                                      past_train_data.timestamps[task_mp][sampled_idx_class], past_train_data.edge_idxs[task_mp][sampled_idx_class], \
                                      past_train_data.labels_src[task_mp][sampled_idx_class], past_train_data.labels_dst[task_mp][sampled_idx_class])
                else:
                    cur_memory.add_data(Data(past_train_data.src[task_mp][sampled_idx_class], past_train_data.dst[task_mp][sampled_idx_class], \
                                      past_train_data.timestamps[task_mp][sampled_idx_class], past_train_data.edge_idxs[task_mp][sampled_idx_class], \
                                      past_train_data.labels_src[task_mp][sampled_idx_class], past_train_data.labels_dst[task_mp][sampled_idx_class]))

            self.memory = Memory()
            self.memory.update_memory(cur_memory)

            return new_mask, new_src_mask, new_dst_mask

        if self.args.select_mode == 'error_min':
            # Compoute the loss for each node of each event, and store the loss values
            self.memory = Memory()
                
            old_src_loss_bank = []
            old_dst_loss_bank = []
                
            old_src_emb_bank = []
            old_dst_emb_bank = []
            with torch.no_grad():
                num_batch = math.ceil(len(past_train_data.src) / args.batch_size)

                for i in range(num_batch):

                    st_idx = i * args.batch_size
                    ed_idx = min((i + 1) * args.batch_size, len(past_train_data.src))

                    src_batch = past_train_data.src[st_idx:ed_idx]
                    dst_batch = past_train_data.dst[st_idx:ed_idx]
                    edge_batch = past_train_data.edge_idxs[st_idx:ed_idx]
                    timestamp_batch = past_train_data.timestamps[st_idx:ed_idx]
                
                    src_loss, dst_loss, src_embeddings, dst_embeddings = self.model(src_batch, dst_batch, edge_batch, timestamp_batch, args.num_neighbors, None, return_emb_loss=True)
                    old_src_loss_bank.append(src_loss)
                    old_dst_loss_bank.append(dst_loss)

                    old_src_emb_bank.append(src_embeddings)
                    old_dst_emb_bank.append(dst_embeddings)
                
                old_src_loss_bank = torch.cat(old_src_loss_bank, dim=0)
                old_dst_loss_bank = torch.cat(old_dst_loss_bank, dim=0)
                total_loss_bank = torch.cat([old_src_loss_bank, old_dst_loss_bank], dim=0)

                old_src_emb_bank = torch.cat(old_src_emb_bank, dim=0)
                old_dst_emb_bank = torch.cat(old_dst_emb_bank, dim=0)

            total_emb_bank = torch.cat([old_src_emb_bank, old_dst_emb_bank], dim=0)
            total_idx_bank = torch.cat([torch.tensor(past_train_data.edge_idxs), torch.tensor(past_train_data.edge_idxs)], dim=0)
            total_label_bank = torch.cat([torch.tensor(past_train_data.labels_src), torch.tensor(past_train_data.labels_dst)], dim=0)

            for y in range(task * args.num_class_per_dataset):
                print(len(total_label_bank))
                y_mask = np.isin(total_label_bank, [y])
                y_emb_bank = total_emb_bank[y_mask]
                y_loss_bank = total_loss_bank[y_mask]
                y_idx_bank = total_idx_bank[y_mask]

                if len(y_emb_bank) > 10000:
                    loss_choice = y_loss_bank.argsort()[:10000].cpu().numpy()
                    y_emb_bank = y_emb_bank[loss_choice]
                    y_idx_bank = y_idx_bank[loss_choice]
                    y_loss_bank = y_loss_bank[loss_choice]
                else:
                    loss_choice = None

                y_K = rbf_kernel(y_emb_bank)

                # calculate the l2 distance between each input embedding and the selected_new_emb
                if args.error_min_loss:
                    temp_y_loss_bank = args.error_min_loss_weight * (y_loss_bank - y_loss_bank.min()) / (y_loss_bank.max() - y_loss_bank.min())
                else:
                    y_loss_bank = torch.zeros_like(y_loss_bank)

                selected_mask = select_prototypes(y_K, temp_y_loss_bank, args.memory_size)
                selected_index = y_idx_bank[selected_mask].unique()
                selected_index_mask = torch.tensor([True if (idx in selected_index) else False for idx in past_train_data.edge_idxs])
            
                cur_memory = Data(past_train_data.src[selected_index_mask], past_train_data.dst[selected_index_mask], past_train_data.timestamps[selected_index_mask], \
                                past_train_data.edge_idxs[selected_index_mask], past_train_data.labels_src[selected_index_mask], past_train_data.labels_dst[selected_index_mask])
                self.memory.update_memory(cur_memory)
                
                del y_K, y_emb_bank, y_idx_bank, y_loss_bank, selected_mask, selected_index_mask, selected_index
            
            return new_mask, new_src_mask, new_dst_mask


    def get_acc(self, x, y):
        output = self.model(x)
        _, pred = torch.max(output, 1)
        correct = (pred == y).sum().item()
        return correct

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

    def similarity(self, emb, old_emb=None, ngh_emb=None, weight=None, mask=None):
        """
        emb input shape:        (batch_size, emb_dim)
        ngh_emb input shape:    (batch_size, n_neighbors, emb_dim)
        output shape:           (batch_size, n_neighbors)
        """
        if self.args.similarity_function == 'mse':
            return
        if self.args.similarity_function == 'cos':
            if old_emb is not None:
                if weight is not None:
                    similarity = (nn.CosineSimilarity(dim=-1)(emb, old_emb) * weight).mean()
                else:
                    similarity = nn.CosineSimilarity(dim=-1)(emb, old_emb).mean()
                similarity = ((similarity + 1) / 2).mean()
            elif ngh_emb is not None:
                similarity = nn.CosineSimilarity(dim=-1)(emb, ngh_emb)
                if mask is not None:
                    similarity[mask] = -1
                similarity = (similarity + 1) / 2
            
            return similarity
        
        if self.args.similarity_function == 'sigcos':
            if old_emb is not None:
                if weight is not None:
                    similarity = ((emb * old_emb) * weight)
                else:
                    similarity = (emb * old_emb)
                similarity = torch.sigmoid(torch.sum(similarity, dim=1)).mean()
            elif ngh_emb is not None:
                temp_emb = emb.expand(-1, ngh_emb.shape[1], -1)
                similarity = torch.sum(temp_emb * ngh_emb, dim=2)
                if mask is not None:
                    similarity[mask] = -float('inf')
                similarity = torch.sigmoid(similarity)

            return similarity
        
    def distribution_difference(self, new_distribution, old_distribution):
        if self.args.distribution_measure == 'KLDiv':
            old_distribution = F.log_softmax(old_distribution, dim=1)
            new_distribution = F.log_softmax(new_distribution, dim=1)
            loss = torch.nn.KLDivLoss(reduction='none', log_target=True)
        elif self.args.distribution_measure == 'MSE':
            loss = torch.nn.MSELoss(reduction='none')
        # elif self.args.distribution_measure == 'CE':
        #     loss = torch.nn.CrossEntropyLoss(reduction='none')

        return loss(new_distribution, old_distribution)



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
        if size > len(self.total_memory.src):
            size = len(self.total_memory.src)
        if mode == 'random':
            idx = np.random.choice(len(self.total_memory.src), size, replace=False)
            return self.total_memory.src[idx], self.total_memory.dst[idx], self.total_memory.edge_idxs[idx], self.total_memory.timestamps[idx]
        
    def get_full_data(self):
        return self.total_memory

    def set_memory(self, memory):
        self.total_memory = memory
    

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


def distillation_loss(old_logits, new_logits, T=2):
    with torch.no_grad():
        old_soft_logits = torch.pow(old_logits, 1/T)
        old_soft_logits = old_soft_logits / torch.sum(old_soft_logits, dim=1, keepdim=True)

    new_soft_logits = torch.pow(new_logits, 1/T)
    new_soft_logits = new_soft_logits / torch.sum(new_soft_logits, dim=1, keepdim=True)

    distill_loss = - torch.sum(old_soft_logits * torch.log(new_soft_logits), dim=1).mean()
    return distill_loss


def local_rbf_kernel(X:torch.Tensor, y:torch.Tensor, gamma:float=None):
    # todo make final representation sparse (optional)
    assert len(X.shape) == 2
    assert len(y.shape) == 1
    assert torch.all(y == y.sort()[0]), 'This function assumes the dataset is sorted by y'

    if gamma is None:
        gamma = default_gamma(X)
    K = torch.zeros((X.shape[0], X.shape[0]))
    y_unique = y.unique()
    for i in range(y_unique[-1] + 1): # compute kernel blockwise for each class
        ind = torch.where(y == y_unique[i])[0]
        start = ind.min()
        end = ind.max() + 1
        K[start:end, start:end] = rbf_kernel(X[start:end, :], gamma=gamma)
    return K

def default_gamma(X:torch.Tensor):
    gamma = 1.0 / X.shape[1]
    # print(f'Setting default gamma={gamma}')
    return gamma

def rbf_kernel(X:torch.Tensor, gamma:float=None):
    assert len(X.shape) == 2

    if gamma is None:
        gamma = default_gamma(X)
    K = torch.cdist(X, X, compute_mode='donot_use_mm_for_euclid_dist')

    mask = torch.eye(K.shape[0]).bool().to(K.device)
    K = K.masked_fill(mask, 0)
    # K = K.fill_diagonal(0) # avoid floating point error
    K = K.pow(2)
    K = K.mul(-gamma)
    K = K.exp()
    return K

def select_prototypes(K:torch.Tensor, loss = None, new_MMD = None, num_prototypes = 0):
    sample_indices = torch.arange(0, K.shape[0])
    num_samples = sample_indices.shape[0]

    colsum = 2 * K.sum(0) / num_samples
    is_selected = torch.zeros_like(sample_indices)
    selected = sample_indices[is_selected > 0]

    # print('colsum stat', colsum.mean(), colsum.max(), colsum.min())

    for i in range(num_prototypes):
        candidate_indices = sample_indices[is_selected == 0]
        s1 = colsum[candidate_indices]

        if selected.shape[0] == 0:
            s1 -= K.diagonal()[candidate_indices].abs()
        else:
            temp = K[selected, :][:, candidate_indices]
            s2 = temp.sum(0) * 2 + K.diagonal()[candidate_indices]
            s2 /= (selected.shape[0] + 1)
            s1 -= s2
        
        if (loss is not None) and (new_MMD is not None):
            s1 = s1 - loss[candidate_indices] - new_MMD[candidate_indices]

        best_sample_index = candidate_indices[s1.argmax()]
        is_selected[best_sample_index] = i + 1
        selected = sample_indices[is_selected > 0]

    return is_selected > 0

    selected_in_order = selected[is_selected[is_selected > 0].argsort()]
    return selected_in_order