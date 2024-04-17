import os
import shutil
from random import sample
import random
import torch
import numpy as np
import math
from torch.nn import init
from tqdm import tqdm
import time
import pickle
import argparse
from pathlib import Path
from methods import get_model

from utils.data_processing import get_data, computer_time_statics, get_past_inductive_data
from utils.utils import get_neighbor_sampler, RandEdgeSampler, EarlyStopMonitor, str2bool
from utils.evaluation import eval_prediction
from utils.log_and_checkpoints import set_logger, get_checkpoint_path, get_mid_model_path
import matplotlib.pyplot as plt
import seaborn
from copy import deepcopy

import wandb

parser = argparse.ArgumentParser('TGCL')

# training settings

parser.add_argument('--debug_mode', type=int, default=0, help='debug mode')
parser.add_argument('--verbose', type=int, default=0, help='debug mode')

# general parameters

parser.add_argument('--dataset', type=str,default='yelp')
parser.add_argument('--model', type=str, default='TGAT', help='Model')
parser.add_argument('--method', type=str, default='Finetune', help='Continual learning method')

# parser.add_argument('--model', type=str, default='OTGNet', help='Model')
parser.add_argument('--batch_size', type=int, default=300, help='Batch_size')
# parser.add_argument('--n_degree', type=int, default=5, help='Number of neighbors to sample')
parser.add_argument('--n_epoch', type=int, default=500, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--select', type=str, default='reinforce', help='Policy select')
# parser.add_argument('--n_task', type=int, default=6, help='Number of tasks')
# parser.add_argument('--n_class', type=int, default=3, help='Classes per task')
parser.add_argument('--n_interval', type=int, default=3, help='Interval of RL training')
parser.add_argument('--n_mc', type=int, default=3, help='Number of MC Dropout')

parser.add_argument('--num_class', type=int, default=3, help='Number of classes')
parser.add_argument('--num_class_per_dataset', type=int, default=3, help='Number of classes per dataset')
parser.add_argument('--num_datasets', type=int, default=6, help='Number of datasets')

parser.add_argument('--num_neighbors', type=int, default=5, help='Number of neighbors to sample')

parser.add_argument('--supervision', type=str, default='supervised', help='Supervision type')
parser.add_argument('--task', type=str, default='nodecls', help='Task type')
parser.add_argument('--feature_type', type=str, default='embedded', help='The type of features used for node classification')

# Memory parameters

parser.add_argument('--memory_replay', type=str2bool, default=0, help='Use memory buffer or not')
parser.add_argument('--select_mode', type=str, default='random', help='How to select the data into the memory')
parser.add_argument('--memory_size', type=int, default=100, help='Size of memory buffer')
parser.add_argument('--memory_frac', type=float, default=-1, help='Size of memory buffer')
parser.add_argument('--memory_replay_weight', type=int, default=1, help='Weight for replaying memory')
parser.add_argument('--replay_select_mode', type=str, default='random', help='How to select the data from the memory')
parser.add_argument('--replay_size', type=int, default=500, help='Size of memory buffer for distribution regularization')
parser.add_argument('--error_min_hash', type=str2bool, default=0, help='Whether hash more data into the selected ones')
parser.add_argument('--error_min_hash_threshold', type=float, default=0.05, help='The similarity threshold for data selection')


parser.add_argument('--old_data_weight', type=float, default=1.0, help='The weight for the total old data loss')
parser.add_argument('--partition', type=str, default='random', help='How to separate the data')

# Learning based subgraph selection
parser.add_argument('--mid_model_path', type=str, default='', help='The storage path for mid model parameters')
parser.add_argument('--mid_epoch', type=int, default=10, help='The epoch for storing the mid model')
parser.add_argument('--param_importance', type=int, default=0, help='Learn the event weight by considering event weights')
parser.add_argument('--event_weight_epochs', type=int, default=50, help='The epochs for obtaining the event weight')
parser.add_argument('--event_weight_l1_weight', type=float, default=1.0, help='The weight for the l1 norm of event loss')
parser.add_argument('--weight_learning_method', type=str, default='param_diff', help='The method for learning the weight')
parser.add_argument('--weight_reg_method', type=str, default='l1', help='The method for regularizing the weight magnitude')

parser.add_argument('--error_min_distribution', type=str2bool, default=True, help='Whether to minimize the error with distribution difference')
# parser.add_argument('--error_min_distance_weight', type=float, default=1.0, help='The weight during selection when using new data')
parser.add_argument('--error_min_loss', type=str2bool, default=True, help='Whether to minimize the error with the loss')
parser.add_argument('--error_min_loss_weight', type=float, default=1.0, help='The weight during selection when using the loss')
# parser.add_argument('--error_min_new_data_kept_ratio', type=float, default=1.0, help='The ratio of new data kept during selection')
parser.add_argument('--error_min_distill', type=str2bool, default=True, help='Whether to require the distribution to be similar')

parser.add_argument('--old_emb_distribution_distill', type=str2bool, default=0, help='Whether to distill the old emb distribution')
# parser.add_argument('--old_emb_distribution_distill_weight', type=float, default=1.0, help='The weight for distilling the old emb distribution')
parser.add_argument('--new_emb_distribution_distill', type=str2bool, default=0, help='Whether to distill the new & old emb distribution')
parser.add_argument('--emb_distribution_distill_weight', type=float, default=1.0, help='The weight for distilling the new & old emb distribution')
parser.add_argument('--reg_gamma', type=float, default=0.1, help='The gamma value when calculating the kernel')

# Distillation parameters
parser.add_argument('--distill', type=str2bool, default=0, help='Distill or not')
parser.add_argument('--emb_distill', type=str2bool, default=0, help='Distill embeddings or not')
parser.add_argument('--emb_distill_weight', type=float, default=1.0, help='The weight for distilling the embeddings')
parser.add_argument('--emb_proj', type=str2bool, default=0, help='Project new embeddings to the old emb space or not')
parser.add_argument('--struct_distill', type=str2bool, default=0, help='Distill the structural information or not')
parser.add_argument('--struct_distill_weight', type=float, default=1.0, help='The weight for distilling the structural information')
parser.add_argument('--similarity_function', type=str, default='cos', help='choose the similarity function')
parser.add_argument('--future_neighbor', type=str2bool, default=0, help='Whether to aggregate the future neighbors')
parser.add_argument('--residual_distill', type=str2bool, default=0, help='Whether to distill the distribution via a residual method')
parser.add_argument('--distribution_measure', type=str, default='KLDiv', help='Which distribution measurement to use, KLDiv, MSE or CE')
parser.add_argument('--emb_residual', type=str2bool, default=0, help='Whether to distill the embeddings via residual method')

parser.add_argument('--rand_neighbor', type=str2bool, default=0, help='Whether to sample random neighbors instead of using past neighbors')

parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for distillation')
parser.add_argument('--ewc_weight', type=float, default=1.0, help='The weight for EWC loss')

# parser.add_argument('--explainer', type=str, default='PGExplainer', help='Explainer')
# parser.add_argument('--explainer_train_epoch', type=int, default=100, help='Number of epochs to train the explainer')
# parser.add_argument('--explainer_lr', type=float, default=0.001, help='Learning rate of the explainer')
# parser.add_argument('--explainer_batch_size', type=int, default=100, help='Batch size of the explainer')
# parser.add_argument('--explainer_reg_coefs', type=float, default=0.1, help='Regularization coefficient of the explainer')
# parser.add_argument('--explainer_level', type=str, default='node', help='the explanation level, node or graph')

# backbone model parameters

parser.add_argument('--use_feature', type=str, default='fg', help='Use node feature or not')
parser.add_argument('--use_time', type=int, default=5, help='Use time or not')
parser.add_argument('--mem_method', type=str, default='triad', help='Memory buffer sample method')
parser.add_argument('--filename_add', type=str, default='', help='Attachment to filename')
parser.add_argument('--device_id', type=int, default=0, help='Device id of cuda')
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
parser.add_argument('--mem_size', type=int, default=10, help='Size of memory slots')
parser.add_argument('--rp_times', type=int, default=1, help='repeat running times')
parser.add_argument('--is_r', type=str2bool, default=1, help='is_r')
parser.add_argument('--blurry', type=str2bool, default=1, help='blurry setting')
parser.add_argument('--online', type=str2bool, default=1, help='online setting')
parser.add_argument('--use_IB', type=str2bool, default=1, help='use IB')
parser.add_argument('--dis_IB', type=str2bool, default=1, help='dis IB')
parser.add_argument('--ch_IB', type=str, default='m', help='ch IB')
parser.add_argument('--pattern_rho', type=float, default=0.1, help='pattern_rho')

parser.add_argument('--multihead', type=str2bool, default=0, help='whether to use multihead classifiers for each data set')
parser.add_argument('--head_hidden_dim', type=int, default=100, help='Number of hidden dimensions of the head classifier')
parser.add_argument('--num_layers', type=int, default=2, help='Number of TGNN layers')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
parser.add_argument('--num_attn_heads', type=int, default=2, help='Number of attention heads')

parser.add_argument('--l2_norm', type=bool, default=True, help='L2 norm')
parser.add_argument('--l2_weight', type=float, default=0.001, help='L2 weight')

parser.add_argument('--node_init_dim', type=int, default=128, help='node initial feature dimension')
parser.add_argument('--node_embedding_dim', type=int, default=128, help='node embedding feature dimension')
parser.add_argument('--time_feat_dim', type=int, default=128, help='time feature dimension')
parser.add_argument('--channel_embedding_dim', type=int, default=50, help='channel embedding dimension')
parser.add_argument('--patch_size', type=int, default=2, help='the size of patch')
parser.add_argument('--time_gap', type=int, default=1000, help='the time gap for searching node neighbors')
parser.add_argument('--max_input_sequence_length', type=int, default=64, help='the max length of input sequence for each node')
# 

parser.add_argument('--feature_iter', type=int, default=1, help='feature_iter')
parser.add_argument('--patience', type=int, default=100, help='patience')
parser.add_argument('--radius', type=float, default=0, help='radius')
parser.add_argument('--beta', type=float, default=0, help='beta')
parser.add_argument('--gamma', type=float, default=0, help='gamma')
parser.add_argument('--uml', type=str2bool, default=0, help='uml')
parser.add_argument('--pmethod', type=str, default='knn', help='pseudo-label method')
parser.add_argument('--sk', type=int, default=1000, help='number of triads candidates')
parser.add_argument('--full_n', type=int, default=1, help='full_n')
parser.add_argument('--recover', type=int, default=1, help='recover')

# training setting

parser.add_argument('--class_balance', type=int, default=1, help='class balance')
parser.add_argument('--eval_avg', type=str, default='node', help='evaluation average')

parser.add_argument('--results_dir', type=str, default='.', help='results diretion')
parser.add_argument('--explainer_ckpt_dir', type=str, default='.', help='check point direction for the explainer')
parser.add_argument('--eval_metric', type=str, default='acc', help='evaluation metric')
parser.add_argument('--eval_scope', type=str, default='macro', help='the scope of the evaluation metric, macro or micro')


log_to_file = True
args = parser.parse_args()
args.num_class = args.num_datasets * args.num_class_per_dataset
# n_interval = args.n_interval
# n_mc = args.n_mc
# args.memory_replay = args.memory_replay==1
# args.multihead = args.multihead==1
# use_feature = args.use_feature
# use_time = args.use_time
# blurry = args.blurry==1
online = args.online
# is_r = args.is_r==1
mem_method = args.mem_method
mem_size = args.mem_size
rp_times = args.rp_times
# use_IB = args.use_IB==1
# dis_IB = args.dis_IB==1
ch_IB = args.ch_IB
pattern_rho = args.pattern_rho
# class_balance = args.class_balance
eval_avg = args.eval_avg
# feature_iter=args.feature_iter==1
# patience=args.patience
# radius = args.radius
# beta = args.beta
# gamma = args.gamma
uml = args.uml
# pmethod = args.pmethod
# sk = args.sk
# full_n = args.full_n==1
# recover = args.recover==1

# args.distill = args.distill
args.emb_distill = args.emb_distill and args.distill
# args.emb_proj = args.emb_proj==1
args.struct_distill = args.struct_distill and args.distill
# args.future_neighbor = args.future_neighbor==1
args.residual_distill = args.residual_distill and args.distill
# args.emb_residual = args.emb_residual==1
# args.rand_neighbor = args.rand_neighbor==1

# args.error_min_new_data = args.error_min_new_data==1
# args.error_min_loss = args.error_min_loss==1
args.old_emb_distribution_distill = args.old_emb_distribution_distill and args.distill
args.new_emb_distribution_distill = args.new_emb_distribution_distill and args.distill

avg_time_cost = []
avg_performance_all=[]
avg_forgetting_all=[]
task_acc_all=[0 for i in range(args.num_datasets)]
task_acc_vary=[[0]*args.num_datasets for i in range(args.num_datasets)]
task_acc_vary_cur=[[0]*args.num_datasets for i in range(args.num_datasets)]
test_acc_record = []

per_epoch_training_time = [[] for _ in range(args.num_datasets)]

if args.debug_mode > 0:
    f = open("./result/{}_debug.txt".format(args.dataset+args.filename_add),"a+")
else:
    f = open("./result/{}.txt".format(args.dataset+args.filename_add),"a+")
f.write("\n +++++++++++++++++++++++++++++++ \n")

run_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
init_args = deepcopy(args)
tags = [args.dataset, args.method, args.model]

for rp in range(rp_times):
    args = deepcopy(init_args)
    if args.debug_mode == 0:
        run = wandb.init(project="Temporal Graph Continual Learning", name=f"{args.dataset}_{args.method}_{args.model}_{run_time}", tags=tags)
        wandb.config.update(args)

    per_task_performance = []
    per_task_performance_matrix = np.zeros((args.num_datasets, args.num_datasets))

    start_time=time.time()
    logger, time_now = set_logger(args.model, args.dataset, args.select, log_to_file)
    Path("log/{}/{}/checkpoints".format(args.model, time_now)).mkdir(parents=True, exist_ok=True)
    Img_path = "log/{}/{}/checkpoints/result.png".format(args.model, time_now)
    Loss_path1 = "log/{}/{}/checkpoints/loss1.png".format(args.model, time_now)
    Loss_path2 = "log/{}/{}/checkpoints/loss2.png".format(args.model, time_now)
    loss_mem1 = []
    loss_mem2 = []
    # f = open("./result/{}.txt".format(args.dataset+args.filename_add),"a+")
    f.write(str(args))
    f.write("\n")
    f.write(time_now)
    f.write("\n")

    print(str(args))
    # data processing
    node_features, edge_features, full_data, train_data, val_data, test_data, all_data, _, _ = get_past_inductive_data(args.dataset,args.num_datasets,args.num_class_per_dataset, args.blurry)
    
    args.node_init_dim = node_features.shape[1]
    args.node_embedding_dim = node_features.shape[1]
    args.time_feat_dim = edge_features.shape[1]

    # Set the seeds
    seed=int(time.time())%100 # Note that this seed can't fix the results
    np.random.seed(seed)  # cpu vars
    torch.manual_seed(seed)  # cpu  vars
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.cuda.manual_seed_all(seed)  # gpu vars
    f.write("seed: %d\n"%seed)

    label_src = all_data.labels_src
    label_dst = all_data.labels_dst
    node_src = all_data.src
    node_dst = all_data.dst
    edge_timestamp = all_data.timestamps

    torch.cuda.set_device(args.device_id)
    device = 'cuda'
    args.device = 'cuda'
    logger.debug(str(args))

    # if args.debug_mode > 0:
    #     device = 'cpu'
    #     args.device = device

    g_time=0

    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    computer_time_statics(train_data[0].src, train_data[0].dst, train_data[0].timestamps)

    args.time_shift = {'mean_time_shift_src': mean_time_shift_src, 'std_time_shift_src': std_time_shift_src, 
                       'mean_time_shift_dst': mean_time_shift_dst, 'std_time_shift_dst': std_time_shift_dst}

    neighbor_finder = get_neighbor_sampler(all_data, 'recent')
    sgnn = get_model(args, neighbor_finder, node_features, edge_features, label_src, label_dst)
    
    sgnn.to(device)
    sgnn.reset_graph()

    logger.debug("./result/{}.txt".format(args.dataset+args.filename_add))
    LOSS = []
    val_acc, val_ap, val_f1 = [], [], []
    early_stopper = [EarlyStopMonitor(max_round=args.patience, higher_better=True) for i in range(args.num_datasets+1)]
    test_best=[0 for i in range(args.num_datasets)]

    if not os.path.exists(f'./checkpoints/{args.model}/'):
        os.makedirs(f'./checkpoints/{args.model}/')

    cur_train_data = None
    cur_test_data = None
    cur_val_data = None

    # for task in range(0,args.num_datasets):
    task_bar = tqdm(range(0,args.num_datasets), desc='Task', position=0, leave=True)
    for task in task_bar:

        args.mid_model_path = get_mid_model_path(args.model, time_now, task)
        
        if args.memory_frac > 0:
            args.memory_size = int(args.memory_frac * len(train_data[task].src))
            print("the memory size is", args.memory_size)

        if task == 0:
            cum_train_data = deepcopy(train_data[task])
        else:
            cum_train_data.add_data(train_data[task])
            
        cum_val_data = deepcopy(cum_train_data)
        cum_val_data.add_data(val_data[task])
        
        cum_test_data = deepcopy(cum_val_data)
        cum_test_data.add_data(test_data[task])

        train_neighbor_finder = get_neighbor_sampler(cum_train_data, 'recent')
        val_neighbor_finder = get_neighbor_sampler(cum_val_data, 'recent')
        test_neighbor_finder = get_neighbor_sampler(cum_test_data, 'recent')

        if args.rand_neighbor and args.method == 'SubGraph':
            rand_sampler = RandEdgeSampler(cum_train_data.src, cum_train_data.dst)
            sgnn.set_sampler(rand_sampler)

        sgnn.set_neighbor_finder(train_neighbor_finder)
        cur_train_data = deepcopy(train_data[task])

        selection_time = 0
        if hasattr(sgnn, 'begin_task') and task > 0:
            begin_time = time.time()
            if args.method == 'SubGraph':
                sgnn.set_class_weight(None)
            train_avail_mask, train_src_avail_mask, train_dst_avail_mask = sgnn.begin_task(args, train_data[task], task)
            selection_time = time.time() - begin_time
            cur_train_data.apply_mask(train_avail_mask, train_src_avail_mask, train_dst_avail_mask)

        # balance the weights for classes
        src_labels = cur_train_data.labels_src
        dst_labels = cur_train_data.labels_dst

        if task > 0:
            src_labels = src_labels[cur_train_data.src_avail_mask]
            dst_labels = dst_labels[cur_train_data.dst_avail_mask]

            if hasattr(sgnn, 'memory'):
                src_labels = np.concatenate([src_labels, sgnn.memory.get_memory().labels_src])
                dst_labels = np.concatenate([dst_labels, sgnn.memory.get_memory().labels_dst])

        class_stat = {i: 0 for i in range((task + 1) * args.num_class_per_dataset)}
        class_count = np.unique(np.concatenate([src_labels, dst_labels]), return_counts=True)
        for i, key in enumerate(class_count[0]):
            class_stat[key] += class_count[1][i]
        class_stat = list(class_stat.values())
        class_weight = np.array([np.sum(class_stat) / len(class_count[1]) / v if v > 0 else 1 for v in class_stat])

        sgnn.set_class_weight(class_weight)

        # print('current full training data class stat:', src_class_stat)
        print('available training data class stat: ', class_stat)

        print('available training data class weight: ', class_weight)
                
        epoch_bar = tqdm(range(args.n_epoch), desc='Epoch', position=1, leave=True)
        for e in epoch_bar:

            epoch_start_time = time.time()

            # print("task:",task,"epoch:",e)
            logger.debug('task {} , start {} epoch'.format(task,e))

            # loss_value = 0
            loss_dict = {'loss': 0, 'l2_reg': 0, 'dist_reg': 0}
            Reward = 0
            sgnn.reset_graph()
            sgnn.set_neighbor_finder(train_neighbor_finder)
            sgnn.train()

            # Learn the selected data first
            num_batch_old = 0
            if hasattr(sgnn, 'memory') and task > 0 and args.memory_replay:
                memory_data = sgnn.memory.get_memory()
                num_batch_old = math.ceil(len(memory_data.src) / args.batch_size)
                
                for i in range(num_batch_old):
                    st_idx = i * args.batch_size
                    ed_idx = min((i + 1) * args.batch_size, len(memory_data.src))

                    src_batch = memory_data.src[st_idx:ed_idx]
                    dst_batch = memory_data.dst[st_idx:ed_idx]
                    edge_batch = memory_data.edge_idxs[st_idx:ed_idx]
                    timestamp_batch = memory_data.timestamps[st_idx:ed_idx]

                    data_dict = sgnn(src_batch, dst_batch, edge_batch, timestamp_batch, args.num_neighbors, is_old_data=True, dataset_idx=task)

                    for key in loss_dict.keys():
                        if key in data_dict:
                            loss_dict[key] += data_dict[key]

                    # loss_dict['loss'] += data_dict['loss']
                    # loss_dict['l2_reg'] += data_dict['l2_reg']

            num_batch = math.ceil(len(cur_train_data.src) / args.batch_size)
            
            # Learn the accessible data next 
            for i in range(num_batch):
                st_idx = i * args.batch_size
                ed_idx = min((i + 1) * args.batch_size, len(cur_train_data.src))

                src_batch = cur_train_data.src[st_idx:ed_idx]
                dst_batch = cur_train_data.dst[st_idx:ed_idx]
                edge_batch = cur_train_data.edge_idxs[st_idx:ed_idx]
                timestamp_batch = cur_train_data.timestamps[st_idx:ed_idx]

                if args.method != 'Joint' and task > 0:
                    src_avail_mask = cur_train_data.src_avail_mask[st_idx:ed_idx]
                    dst_avail_mask = cur_train_data.dst_avail_mask[st_idx:ed_idx]
                else:
                    src_avail_mask = None
                    dst_avail_mask = None

                data_dict = sgnn(src_batch, dst_batch, edge_batch, timestamp_batch, args.num_neighbors, dataset_idx=task, src_avail_mask=src_avail_mask, dst_avail_mask=dst_avail_mask)

                for key in loss_dict.keys():
                    if key in data_dict:
                        loss_dict[key] += data_dict[key]

                # loss_dict['loss'] += data_dict['loss']
                # loss_dict['l2_reg'] += data_dict['l2_reg']
                        
            for key in loss_dict.keys():
                loss_dict[key] /= (num_batch + num_batch_old)

            # loss_dict['loss'] = loss_dict['loss'] / (num_batch + num_batch_old)
            # loss_dict['l2_reg'] = loss_dict['l2_reg'] / (num_batch + num_batch_old)
            # Obj=Obj / (num_batch + num_batch_old)
            sgnn.end_epoch()

            epoch_end_time = time.time()
            per_epoch_training_time[task].append(epoch_end_time - epoch_start_time)

            loss_mem1.append(loss_dict['loss'])
            # loss_mem2.append(Obj)
            # print("train loss: %.4f"%loss_value)
            # print("obj: %.4f"%(Obj))
            LOSS.append(loss_dict['loss'])
            logger.debug("loss in whole dataset = {}".format(loss_dict['loss']))
            
            # validation
            sgnn.eval()

            sgnn.reset_graph()
            _, train_result = eval_prediction(sgnn, train_data[task], task, task, args.batch_size, 'train', uml, eval_avg, args.multihead, args.num_class_per_dataset, within_task=True)

            sgnn.set_neighbor_finder(val_neighbor_finder)
            
            avg_val_result, task_val_result = eval_prediction(sgnn, val_data[task], task, task, args.batch_size, 'test', uml, eval_avg, args.multihead, args.num_class_per_dataset, within_task=True)

            if (hasattr(sgnn, "memory") and args.memory_replay) or args.method == 'Joint':
                val_result = avg_val_result[args.eval_metric]
            else:
                val_result = task_val_result[args.eval_metric][-1]

            train_memory_backup = sgnn.back_up_memory()
            
            if args.model=='OTGNet':
                train_IB_backup = sgnn.back_up_IB()
                if uml:
                    train_PGen_backup = sgnn.back_up_PGen()
                else:
                    train_PGen_backup =None
            else:
                train_IB_backup = None
                train_PGen_backup =None
            
            sgnn.set_neighbor_finder(test_neighbor_finder)
            
            avg_test_result, test_result = eval_prediction(sgnn, test_data[task], task, task, args.batch_size, 'test', uml, eval_avg, args.multihead, args.num_class_per_dataset, within_task=True)
            
            # print(f"train_{args.eval_metric}: {train_result[args.eval_metric][-1]:.2f}  val_{args.eval_metric}: {avg_val_result[args.eval_metric]:.2f}   test_{args.eval_metric}: {avg_test_result[args.eval_metric]:.2f}")
            logger.debug(f"train_{args.eval_metric}: {train_result[args.eval_metric][-1]:.2f}  val_{args.eval_metric}: {avg_val_result[args.eval_metric]:.2f}   test_{args.eval_metric}: {avg_test_result[args.eval_metric]:.2f}")

            epoch_bar.set_postfix({'loss': loss_dict['loss'], 'l2_reg': loss_dict['l2_reg'], f'train_{args.eval_metric}': train_result[args.eval_metric][-1], f'val_{args.eval_metric}': task_val_result[args.eval_metric][-1], f'test_{args.eval_metric}': test_result[args.eval_metric][-1]})
            if args.debug_mode == 0:
                log_dict = {}
                for key in loss_dict.keys():
                    log_dict[key+f'_period{task+1}'] = loss_dict[key]
                log_dict.update({'epoch': e, 'period': task+1})
                # log_dict = {f'loss_period{task+1}': loss_dict['loss'], f'l2_reg_period{task+1}': loss_dict['l2_reg'], 'epoch': e, 'period': task+1}
                for metric in ['acc', 'ap', 'f1']:
                    for t in range(task+1):
                        log_dict[f'train_{metric}_task{t+1}_period{task+1}'] = train_result[metric][t]
                        log_dict[f'val_{metric}_task{t+1}_period{task+1}'] = task_val_result[metric][t]
                        log_dict[f'test_{metric}_task{t+1}_period{task+1}'] = test_result[metric][t]
                    log_dict[f'avg_val_{metric}_period{task+1}'] = avg_val_result[metric]
                    log_dict[f'avg_test_{metric}_period{task+1}'] = avg_test_result[metric]
                wandb.log(log_dict)

            sgnn.restore_memory(train_memory_backup)

            if online:
                for k in range(task+1):
                    if not args.multihead:
                        test_n_acc, test_n_ap, test_n_f1, test_m_acc = eval_prediction(sgnn, test_data[k], task, k, args.batch_size, 'test', uml, eval_avg, args.multihead, args.num_class_per_dataset)
                    else:
                        test_n_acc, test_n_ap, test_n_f1, test_m_acc = eval_prediction(sgnn, test_data[k], k, k, args.batch_size, 'test', uml, eval_avg, args.multihead, args.num_class_per_dataset)
                    test_best[k]=max(test_best[k], test_n_acc)
                    task_acc_vary[k][task]+=test_n_acc
                break
            else:
                if early_stopper[task].early_stop_check(val_result, sgnn, args.model, train_memory_backup, time_now, task, train_IB_backup, train_PGen_backup) or e == args.n_epoch - 1:
                    logger.info(f'Early stop at {early_stopper[task].max_round} epochs, loading the best model at epoch {early_stopper[task].best_epoch}')
                    # logger.info(f'Loading the best model at epoch {early_stopper[task].best_epoch}')
                    best_model_path, _, _, _ = get_checkpoint_path(args.model, time_now, task, uml)
                    sgnn = torch.load(best_model_path)
                    sgnn.set_features(node_features, edge_features)
                    # logger.info(f'Loaded the best model at epoch {early_stopper[task].best_epoch} for inference')
                    sgnn.eval()

                    sgnn.set_neighbor_finder(test_neighbor_finder)

                    best_model_path, best_mem_path, best_IB_path, best_PGen_path = get_checkpoint_path(args.model, time_now, task, uml)
                    if args.memory_replay:
                        best_mem = torch.load(best_mem_path)
                        sgnn.restore_memory(best_mem)

                    if args.model=='OTGNet':
                        if not args.dis_IB:
                            best_IB = torch.load(best_IB_path)
                            if args.dataset != 'reddit' and args.dataset != 'yelp':
                                sgnn.restore_IB(best_IB)
                        if uml:
                            best_PGen = torch.load(best_PGen_path)
                            sgnn.restore_PGen(best_PGen)

                    test_result, task_test_result = eval_prediction(sgnn, test_data[task], task, task, args.batch_size, 'test', uml, eval_avg, args.multihead, args.num_class_per_dataset, within_task=True)

                    print(f"current all task {args.eval_metric}: ", test_result[args.eval_metric])
                    print(f"per task {args.eval_metric}", task_test_result[args.eval_metric])
                    per_task_performance.append(task_test_result[args.eval_metric])
                    per_task_performance_matrix[task, :task+1] = np.array(task_test_result[args.eval_metric])
                    test_acc_record.append(test_result[args.eval_metric])

                    break   
        
            sgnn.set_features(node_features, edge_features)
            
        if hasattr(sgnn, "old_model"):
            sgnn.old_model = deepcopy(sgnn.model)

        # if args.memory_replay:
        #     if args.select_mode == 'event_weight' and args.method == 'SubGraph':
        #         sgnn.end_dataset(train_data[task], args, val_data[task])
        #     else:
        #         sgnn.end_dataset(train_data[task], args)

        if args.debug_mode == 0:
            # wandb.log({'final_avg_performance': test_acc_record[-1], 'period': (task + 1)})
            wandb.log({'final_avg_performance': test_acc_record[-1], 'final_avg_f1': test_result['f1'], 'final_avg_ap': test_result['ap'], 'final_avg_acc': test_result['acc'], 'period': (task + 1), 'selection_time': selection_time})

    per_task_performance_matrix_str = np.array2string(per_task_performance_matrix, precision=2, separator='\t', suppress_small=True)
    per_task_performance_matrix_str = per_task_performance_matrix_str.replace('[', '').replace(']', '')
    print('Performance List: ', test_acc_record)
    # print("Average performance: %.2f"%(np.array(avg_performance).mean()))
    print("Average performance: %.2f"%(test_acc_record[-1]))
    print(f"Per Task Performance: \n {per_task_performance_matrix_str}\n")

    avg_performance_all.append(test_acc_record[-1])

    f.write("Performance List: "+str(test_acc_record))
    f.write("\n")
    f.write("Average performance: %.2f"%(test_acc_record[-1]))
    f.write("\n")
    f.write(f"Per Task Performance: \n {per_task_performance_matrix_str}\n")
    f.write("\n")
    # f.write("Average forgetting: %.2f"%(np.array(avg_forgetting[:-1]).mean()))
    # f.write("\n")
    if mem_method=='triad':
        print("greedy_time: ", g_time/3600)
        f.write("greedy_time: "+str(g_time/3600))
        f.write("\n")
    all_time=time.time()-start_time
    avg_time_cost.append(all_time)
    print("all_time: ", all_time/3600)
    f.write("all_time: "+str(all_time/3600))
    f.write("\n")

    if args.debug_mode == 0:
        run.finish()

f.write(str(args))
f.write("\n")
f.write(time_now)
f.write("\n")
print("Overall AP: %.2f (%.2f)"%(np.array(avg_performance_all).mean(), np.array(avg_performance_all).std()))
# print("Overall AF: %.2f (%.2f)"%(np.array(avg_forgetting_all).mean(), np.array(avg_performance_all).std()))

f.write(f"Backbone:{args.model}, Method:{args.method}\n")
f.write(f"Select Mode:{args.select_mode}, Memory Size:{args.memory_size}, Memory Weight:{args.old_data_weight}\n")
f.write(f"Distill:{args.distill}, Old Dist Distill:{args.old_emb_distribution_distill}, New Dist Distill:{args.new_emb_distribution_distill}\n")
f.write(f"Dist Distill Weight:{args.emb_distribution_distill_weight}, Reg Gamma:{args.reg_gamma}, \n")
f.write(f"Error Min Loss:{args.error_min_loss}, Error Min Loss Weight:{args.error_min_loss_weight}, Error Min Distribution:{args.error_min_distribution}\n")
f.write(f"Emb Distill:{args.emb_distill}, Struct Distill:{args.struct_distill}, Emb Proj:{args.emb_proj}, Residual Distill:{args.residual_distill}, Rand Neighbor:{args.rand_neighbor}\n")
f.write(f"Emb Distill Weight:{args.emb_distill_weight}, Struct Distill Weight:{args.struct_distill_weight}, Similarity Function:{args.similarity_function}, Distribution Measure:{args.distribution_measure}\n")
f.write(f"Weight Learning Method:{args.weight_learning_method}, Weight Reg Method:{args.weight_reg_method}, Weight Training Epoch:{args.event_weight_epochs}\n")
f.write("Overall AP: %.2f (%.2f)"%(np.array(avg_performance_all).mean(), np.array(avg_performance_all).std()))
f.write("\n")
# f.write("Overall AF: %.2f (%.2f)"%(np.array(avg_forgetting_all).mean(), np.array(avg_forgetting_all).std()))
# f.write("\n")
# for i in range(args.num_datasets):
#     print("Overall task %d performance: %.2f"%(i,task_acc_all[i]/rp_times))
#     f.write("Overall task %d performance: %.2f"%(i,task_acc_all[i]/rp_times))
#     f.write("\n")
f.write(f"Average Time Cost: {(np.array(avg_time_cost) / 3600).mean():.2f}±{(np.array(avg_time_cost) / 3600).std():.2f}\n")
for i in range(args.num_datasets):
    f.write(f"Average Training Time of Task {i}: {np.array(per_epoch_training_time[i]).mean():.2f}±{np.array(per_epoch_training_time[i]).std():.2f}\n")
    print(f"Average Training Time of Task {i}: {np.array(per_epoch_training_time[i]).mean():.2f}±{np.array(per_epoch_training_time[i]).std():.2f}\n")

c_list=['tomato','golden','pea','leaf','jade','bluish','violet','strawberry']
for i in range(args.num_datasets):
    for j in range(i,args.num_datasets):    
        task_acc_vary[i][j]/=rp_times
    f.write("task %d: "%(i)+str(task_acc_vary[i][i:]))
    f.write("\n")

f.write("\n ========================= \n")
f.close()

shutil.rmtree("./log/{}/{}/".format(args.model, time_now))