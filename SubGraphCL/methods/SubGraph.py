import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

import math
from models.Backbone import TemporalGNNClassifier
from copy import deepcopy
from sklearn.cluster import KMeans, AgglomerativeClustering
import time
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
                # loss = self.model(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx)

                loss_src, loss_dst, src_emb, dst_emb = self.model(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx, return_emb_loss=True)

                loss = loss_src.mean() + loss_dst.mean()
                
                if self.args.distill:
                    
                    if self.args.error_min_distill:
                        # print("distribution regularization")
                        cur_labels = torch.cat([self.model.src_label[edges], self.model.dst_label[edges]], dim=0)
                        cur_emb = torch.cat([src_emb, dst_emb], dim=0)

                        if self.args.error_min_hash:
                            temp_src_nodes, temp_dst_nodes, temp_edges, temp_edge_times, temp_weight = self.new_memory.get_full_data()
                            temp_weight = torch.tensor(temp_weight,dtype=torch.float).to(self.args.device).detach()
                            temp_weight = torch.cat([temp_weight, temp_weight], dim=0)
                        else:
                            temp_src_nodes, temp_dst_nodes, temp_edges, temp_edge_times = self.new_memory.get_full_data()
                        temp_labels = torch.cat([self.model.src_label[temp_edges], self.model.dst_label[temp_edges]], dim=0)

                        with torch.no_grad():
                            temp_src_emb, temp_dst_emb = self.model.get_embeddings(temp_src_nodes, temp_dst_nodes, temp_edges, temp_edge_times, n_neighbors)
                        temp_emb = torch.cat([temp_src_emb, temp_dst_emb], dim=0).detach()

                        unique_labels = torch.unique(cur_labels)

                        distribution_loss = 0
                        for label in unique_labels:
                            if label >= dataset_idx * self.args.num_class_per_dataset:
                                continue
                            cur_mask = cur_labels == label
                            temp_mask = temp_labels == label

                            if len(cur_emb[cur_mask]) < int(self.args.batch_size * self.args.distill_size_threshold):
                                continue
                            
                            start_time = time.time()
                            # new_distribution_kernel = rbf_kernel(torch.cat([cur_emb[cur_mask], temp_emb[temp_mask]], dim=0), gamma=self.args.reg_gamma)
                            new_distribution_kernel = partial_rbf_kernel(cur_emb[cur_mask], temp_emb[temp_mask], gamma=self.args.reg_gamma)
                            # print("Kernel shape", new_distribution_kernel.shape)
                            # print("Kernel Time: ", time.time() - start_time)

                            n, m = len(cur_emb[cur_mask]), len(temp_emb[temp_mask])
                            
                            # XX = new_distribution_kernel[:n, :n]
                            # YY = new_distribution_kernel[n:, n:]
                            # XY = new_distribution_kernel[:n, n:]
                            # YX = new_distribution_kernel[n:, :n]
                            
                            start_time = time.time()
                            # if self.args.error_min_hash:
                            #     # print("hashing the number of similar data as weights")
                            #     # print(len(temp_weight), len(temp_mask))
                            #     weight = temp_weight[temp_mask]
                            #     weight_matrix = torch.diag(weight)
                            #     m = weight.sum()
                                
                            #     XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  # Source<->Source
                            #     XY = torch.mm(XY, weight_matrix)
                            #     XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) # Source<->Target
                                
                            #     YX = torch.mm(weight_matrix, YX)
                            #     YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) #Target<->Source
                            #     YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  # Target<->Target
                            # else:
                                # XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  # Source<->Source
                                # XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) # Source<->Target

                                # YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) #Target<->Source
                                # YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  # Target<->Target
                                
                            # distribution_loss += (XY).sum() + (YX).sum()
                            distribution_loss += 2 * torch.div(new_distribution_kernel, -n * m).sum(dim=1).view(1,-1).sum()

                            # print("Distribution Time: ", time.time() - start_time)
                        
                        loss += distribution_loss * self.args.emb_distribution_distill_weight
                        data_dict['dist_reg'] = float(distribution_loss)

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
            
            start_time = time.time()
            loss.backward()
            # print("Backward Time: ", time.time() - start_time)
            
            self.optimizer.step()
            if self.args.distill and self.args.emb_proj:
                self.emb_projector_optimizer.step()
            if self.args.distill and self.args.residual_distill:
                self.residual_projector_optimizer.step()

                            
            # if self.args.distill and self.args.error_min_distill and is_old_data:
            #     print('GPU Memory Usage After Loss Backward:')
            #     print('Allocated:', round(torch.cuda.memory_allocated(5)/1024**3,4), 'GB')
            #     print('Reserved: ', round(torch.cuda.memory_reserved(5)/1024**3,4), 'GB')
            
            #     del new_distribution_kernel
                
            #     print('GPU Memory Usage After Matrix Deletion:')
            #     print('Allocated:', round(torch.cuda.memory_allocated(5)/1024**3,4), 'GB')
            #     print('Reserved: ', round(torch.cuda.memory_reserved(5)/1024**3,4), 'GB')
                            
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
            self.new_memory = Memory()
                
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
                
                    src_loss, dst_loss, src_embeddings, dst_embeddings = self.model(src_batch, dst_batch, edge_batch, timestamp_batch, args.num_neighbors, task, return_emb_loss=True)
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
                y_mask = np.isin(total_label_bank, [y])
                y_emb_bank = total_emb_bank[y_mask]
                y_loss_bank = total_loss_bank[y_mask]
                y_idx_bank = total_idx_bank[y_mask]

                # num_parts = int(len(y_emb_bank) / 10000) + 1
                num_parts = int(len(y_emb_bank) / self.args.partition_size) + 1

                selected_index = []
                distribution_only_selected_index = []
                num_similar_embeddings_list = []

                if self.args.partition == 'none':
                    K = rbf_kernel(y_emb_bank)
                    if args.error_min_loss:
                        temp_cluster_loss = args.error_min_loss_weight * (y_loss_bank - y_loss_bank.min()) / (y_loss_bank.max() - y_loss_bank.min())
                    else:
                        temp_cluster_loss = torch.zeros_like(y_loss_bank)
                    
                    if args.error_min_distribution:
                        selected_mask = select_prototypes(K, temp_cluster_loss, num_prototypes=int(args.memory_size / num_parts))
                    elif args.error_min_loss:
                        selected_mask = torch.topk(temp_cluster_loss, int(args.memory_size / num_parts), largest=False)[1]
                    
                    selected_index.append(y_idx_bank[selected_mask].unique())

                if self.args.partition in ['kmeans', 'hierarchical']:
                    
                    if self.args.partition == 'kmeans':
                        y_cluster_idx = KMeans(n_clusters=num_parts, random_state=0, n_init="auto").fit_predict(y_emb_bank.cpu().numpy())
                    if self.args.partition == 'hierarchical':
                        y_cluster_idx = AgglomerativeClustering(n_clusters=num_parts, random_state=0).fit_predict(y_emb_bank.cpu().numpy())
                    
                    for cluster in range(num_parts):
                        cluster_mask = y_cluster_idx == cluster
                        cluster_emb = y_emb_bank[cluster_mask]
                        cluster_loss = y_loss_bank[cluster_mask]
                        cluster_idx = y_idx_bank[cluster_mask]

                        print("Cluster size: ", len(cluster_emb))
                        
                        cluster_K = rbf_kernel(cluster_emb)

                        # calculate the l2 distance between each input embedding and the selected_new_emb
                        if args.error_min_loss:
                            temp_cluster_loss = args.error_min_loss_weight * (cluster_loss - cluster_loss.min()) / (cluster_loss.max() - cluster_loss.min())
                        else:
                            temp_cluster_loss = torch.zeros_like(cluster_loss)
                        
                        if args.error_min_distribution:
                            selected_mask = select_prototypes(cluster_K, temp_cluster_loss, num_prototypes=int(args.memory_size / num_parts))
                        elif args.error_min_loss:
                            selected_mask = torch.topk(temp_cluster_loss, int(args.memory_size / num_parts), largest=False)[1]
                        
                        selected_index.append(cluster_idx[selected_mask].unique())

                        if args.error_min_distill:
                            # print("data selected for distillation")
                            distribution_only_selected_mask = select_prototypes(cluster_K, None, num_prototypes=int(args.replay_size / num_parts))
                            distribution_only_selected_index.append(cluster_idx[distribution_only_selected_mask].unique())
                
                elif self.args.partition == 'random':
                    temp_idx = np.array(range(len(y_emb_bank)))
                    np.random.shuffle(temp_idx)
    
                    # Calculate the number of elements in each part
                    part_size = len(y_idx_bank) // num_parts
                    # Split the data into parts
                    partitions = [temp_idx[i*part_size:(i+1)*part_size] for i in range(num_parts)]

                    for part in partitions:
                        cluster_emb = y_emb_bank[part]
                        cluster_loss = y_loss_bank[part]
                        cluster_idx = y_idx_bank[part]

                        cluster_K = rbf_kernel(cluster_emb)

                        if args.error_min_loss:
                            temp_cluster_loss = args.error_min_loss_weight * (cluster_loss - cluster_loss.min()) / (cluster_loss.max() - cluster_loss.min())
                        else:
                            temp_cluster_loss = torch.zeros_like(cluster_loss)

                        if args.error_min_distribution:
                            selected_mask = select_prototypes(cluster_K, temp_cluster_loss, num_prototypes=int(args.memory_size / num_parts))
                        elif args.error_min_loss:
                            selected_mask = torch.topk(temp_cluster_loss, int(args.memory_size / num_parts), largest=False)[1].cpu()
                        
                        selected_index.append(cluster_idx[selected_mask].unique())
                        
                        if args.error_min_distill:
                            # print("data selected for distillation")
                            distribution_only_selected_mask = select_prototypes(cluster_K, None, num_prototypes=int(args.replay_size / num_parts))
                            distribution_only_selected_index.append(cluster_idx[distribution_only_selected_mask].unique())

                            if args.error_min_hash:
                                # print("hashing the number of similar data")
                                num_similar_embeddings = []
                                
                                for idx in distribution_only_selected_index[-1]:
                                    loc = torch.where(cluster_idx == idx)[0][0]

                                    similarity_values = cluster_K[loc]

                                    # Define a similarity threshold
                                    similarity_threshold = args.error_min_hash_threshold

                                    # Find the indices where the similarity is above the threshold
                                    similar_indices = torch.where(similarity_values > similarity_threshold)

                                    # Count the number of similar embeddings
                                    num_similar_embeddings.append(len(similar_indices))
                                
                                num_similar_embeddings_list.extend(num_similar_embeddings)

                                # print(len(distribution_only_selected_index[-1]))
                                # print(len(num_similar_embeddings))
                                # print(num_similar_embeddings)

                selected_index = torch.cat(selected_index)
                selected_index_mask = torch.tensor([True if (idx in selected_index) else False for idx in past_train_data.edge_idxs])
            
                cur_memory = Data(past_train_data.src[selected_index_mask], past_train_data.dst[selected_index_mask], past_train_data.timestamps[selected_index_mask], \
                                past_train_data.edge_idxs[selected_index_mask], past_train_data.labels_src[selected_index_mask], past_train_data.labels_dst[selected_index_mask])
                self.memory.update_memory(cur_memory)

                if args.error_min_distill:
                    distribution_only_selected_index = torch.cat(distribution_only_selected_index)
                    distribution_only_selected_index_mask = torch.tensor([True if (idx in distribution_only_selected_index) else False for idx in past_train_data.edge_idxs])

                    if args.error_min_hash:
                        temp_num_similar_embeddings_list = np.array(num_similar_embeddings_list)
                        num_similar_embeddings_list = []
                        
                        for _, idx in enumerate(past_train_data.edge_idxs[distribution_only_selected_index_mask]):
                            if idx in distribution_only_selected_index:
                                loc = torch.where(distribution_only_selected_index == idx)[0][0]
                                num_similar_embeddings_list.append(temp_num_similar_embeddings_list[loc])
                        
                        num_similar_embeddings_list = np.array(num_similar_embeddings_list)

                        cur_memory = Data(past_train_data.src[distribution_only_selected_index_mask], past_train_data.dst[distribution_only_selected_index_mask], past_train_data.timestamps[distribution_only_selected_index_mask], \
                                    past_train_data.edge_idxs[distribution_only_selected_index_mask], past_train_data.labels_src[distribution_only_selected_index_mask], past_train_data.labels_dst[distribution_only_selected_index_mask], \
                                        weight=num_similar_embeddings_list)
                    else:
                        cur_memory = Data(past_train_data.src[distribution_only_selected_index_mask], past_train_data.dst[distribution_only_selected_index_mask], past_train_data.timestamps[distribution_only_selected_index_mask], \
                                past_train_data.edge_idxs[distribution_only_selected_index_mask], past_train_data.labels_src[distribution_only_selected_index_mask], past_train_data.labels_dst[distribution_only_selected_index_mask])
                    
                    self.new_memory.update_memory(cur_memory)
                
                # del y_K, y_emb_bank, y_idx_bank, y_loss_bank, selected_mask, selected_index_mask, selected_index
            
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

    def set_class_weight(self, class_weight):
        if class_weight is not None:
            self.model.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight).float().to(self.args.device), reduction='none')
        else:
            self.model.criterion = nn.CrossEntropyLoss(reduction='none')

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
        if self.total_memory.weight is not None:
            return self.total_memory.src, self.total_memory.dst, self.total_memory.edge_idxs, self.total_memory.timestamps, self.total_memory.weight
        else:    
            return self.total_memory.src, self.total_memory.dst, self.total_memory.edge_idxs, self.total_memory.timestamps
        # return self.total_memory

    def get_memory(self):
        return self.total_memory
    
    def set_memory(self, memory):
        self.total_memory = memory
    

class Data:
    def __init__(
        self, src, dst, timestamps, edge_idxs, labels_src, labels_dst, induct_nodes=None, weight=None
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

        self.weight = weight
        # self.dst_weight = dst_weight

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

        if x.weight is not None:
            self.weight = np.concatenate((self.weight, x.weight))


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

def partial_rbf_kernel(X:torch.Tensor, Y:torch.Tensor, gamma:float=None):
    assert len(X.shape) == 2

    if gamma is None:
        gamma = default_gamma(X)
    
    K = torch.cdist(X, Y, compute_mode='donot_use_mm_for_euclid_dist')

    # mask = torch.eye(K.shape[0]).bool().to(K.device)
    # K = K.masked_fill(mask, 0)
    # K = K.fill_diagonal(0) # avoid floating point error
    K = K.pow(2)
    K = K.mul(-gamma)
    K = K.exp()
    return K

def select_prototypes(K:torch.Tensor, loss = None, num_prototypes = 0):
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
        
        if (loss is not None):
            s1 = s1 - loss[candidate_indices]

        best_sample_index = candidate_indices[s1.argmax()]
        is_selected[best_sample_index] = i + 1
        selected = sample_indices[is_selected > 0]

    return is_selected > 0

    selected_in_order = selected[is_selected[is_selected > 0].argsort()]
    return selected_in_order