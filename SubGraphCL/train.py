import os
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

from utils.data_processing import get_data, computer_time_statics
from utils.utils import get_neighbor_finder, RandEdgeSampler, EarlyStopMonitor
from utils.evaluation import eval_prediction
from utils.log_and_checkpoints import set_logger, get_checkpoint_path
import matplotlib.pyplot as plt
import seaborn
from copy import deepcopy

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
parser.add_argument('--feature_type', type=str, default='both', help='The type of features used for node classification')

# continual learning method parameters

parser.add_argument('--multihead', type=int, default=1, help='whether to use multihead classifiers for each data set')
parser.add_argument('--head_hidden_dim', type=int, default=100, help='Number of hidden dimensions of the head classifier')
parser.add_argument('--num_layer', type=int, default=1, help='Number of TGNN layers')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')

# Memory parameters

parser.add_argument('--memory_replay', type=int, default=0, help='Use memory buffer or not')
parser.add_argument('--select_mode', type=str, default='random', help='How to select the data into the memory')
parser.add_argument('--memory_size', type=int, default=50, help='Size of memory buffer')
parser.add_argument('--memory_replay_weight', type=int, default=1, help='Weight for replaying memory')
parser.add_argument('--replay_select_mode', type=str, default='random', help='How to select the data from the memory')
parser.add_argument('--replay_size', type=int, default=100, help='The number of data to replay')

parser.add_argument('--explainer', type=str, default='PGExplainer', help='Explainer')
parser.add_argument('--explainer_train_epoch', type=int, default=100, help='Number of epochs to train the explainer')
parser.add_argument('--explainer_lr', type=float, default=0.001, help='Learning rate of the explainer')
parser.add_argument('--explainer_batch_size', type=int, default=100, help='Batch size of the explainer')
parser.add_argument('--explainer_reg_coefs', type=float, default=0.1, help='Regularization coefficient of the explainer')
parser.add_argument('--explainer_level', type=str, default='node', help='the explanation level, node or graph')

# backbone model parameters

parser.add_argument('--use_feature', type=str, default='fg', help='Use node feature or not')
parser.add_argument('--use_time', type=int, default=5, help='Use time or not')
parser.add_argument('--mem_method', type=str, default='triad', help='Memory buffer sample method')
parser.add_argument('--filename_add', type=str, default='', help='Attachment to filename')
parser.add_argument('--device', type=int, default=0, help='Device of cuda')
parser.add_argument('--mem_size', type=int, default=10, help='Size of memory slots')
parser.add_argument('--rp_times', type=int, default=1, help='repeat running times')
parser.add_argument('--is_r', type=int, default=1, help='is_r')
parser.add_argument('--blurry', type=int, default=1, help='blurry setting')
parser.add_argument('--online', type=int, default=1, help='online setting')
parser.add_argument('--use_IB', type=int, default=1, help='use IB')
parser.add_argument('--dis_IB', type=int, default=1, help='dis IB')
parser.add_argument('--ch_IB', type=str, default='m', help='ch IB')
parser.add_argument('--pattern_rho', type=float, default=0.1, help='pattern_rho')
parser.add_argument('--num_attn_heads', type=int, default=2, help='Number of attention heads')

parser.add_argument('--node_init_dim', type=int, default=128, help='node initial feature dimension')
parser.add_argument('--node_embedding_dim', type=int, default=128, help='node embedding feature dimension')
# 

parser.add_argument('--feature_iter', type=int, default=1, help='feature_iter')
parser.add_argument('--patience', type=int, default=100, help='patience')
parser.add_argument('--radius', type=float, default=0, help='radius')
parser.add_argument('--beta', type=float, default=0, help='beta')
parser.add_argument('--gamma', type=float, default=0, help='gamma')
parser.add_argument('--uml', type=int, default=0, help='uml')
parser.add_argument('--pmethod', type=str, default='knn', help='pseudo-label method')
parser.add_argument('--sk', type=int, default=1000, help='number of triads candidates')
parser.add_argument('--full_n', type=int, default=1, help='full_n')
parser.add_argument('--recover', type=int, default=1, help='recover')

# training setting

parser.add_argument('--class_balance', type=int, default=1, help='class balance')
parser.add_argument('--eval_avg', type=str, default='node', help='evaluation average')

parser.add_argument('--results_dir', type=str, default='.', help='results diretion')
parser.add_argument('--explainer_ckpt_dir', type=str, default='.', help='check point direction for the explainer')


log_to_file = True
args = parser.parse_args()
args.dataset = args.dataset
args.model = args.model
args.select = args.select
args.n_epoch = args.n_epoch
args.batch_size = args.batch_size
args.num_neighbors = args.num_neighbors
args.lr = args.lr
args.num_datasets = args.num_datasets
args.num_class_per_dataset = args.num_class_per_dataset
args.num_class = args.num_datasets * args.num_class_per_dataset
n_interval = args.n_interval
n_mc = args.n_mc
args.memory_replay = args.memory_replay==1
args.multihead = args.multihead==1
use_feature = args.use_feature
use_time = args.use_time
blurry = args.blurry==1
online = args.online==1
is_r = args.is_r==1
mem_method = args.mem_method
mem_size = args.mem_size
rp_times = args.rp_times
use_IB = args.use_IB==1
dis_IB = args.dis_IB==1
ch_IB = args.ch_IB
pattern_rho = args.pattern_rho
class_balance = args.class_balance
eval_avg = args.eval_avg
feature_iter=args.feature_iter==1
patience=args.patience
radius = args.radius
beta = args.beta
gamma = args.gamma
uml = args.uml==1
pmethod = args.pmethod
sk = args.sk
full_n = args.full_n==1
recover = args.recover==1

avg_performance_all=[]
avg_forgetting_all=[]
task_acc_all=[0 for i in range(args.num_datasets)]
task_acc_vary=[[0]*args.num_datasets for i in range(args.num_datasets)]
task_acc_vary_cur=[[0]*args.num_datasets for i in range(args.num_datasets)]



for rp in range(rp_times):
    start_time=time.time()
    logger, time_now = set_logger(args.model, args.dataset, args.select, log_to_file)
    Path("log/{}/{}/checkpoints".format(args.model, time_now)).mkdir(parents=True, exist_ok=True)
    Img_path = "log/{}/{}/checkpoints/result.png".format(args.model, time_now)
    Loss_path1 = "log/{}/{}/checkpoints/loss1.png".format(args.model, time_now)
    Loss_path2 = "log/{}/{}/checkpoints/loss2.png".format(args.model, time_now)
    loss_mem1 = []
    loss_mem2 = []
    f = open("./result/{}.txt".format(args.dataset+args.filename_add),"a+")
    f.write(str(args))
    f.write("\n")
    f.write(time_now)
    f.write("\n")

    print(str(args))
    # data processing
    node_features, edge_features, full_data, train_data, val_data, test_data, all_data, re_train_data, re_val_data = get_data(args.dataset,args.num_datasets,args.num_class_per_dataset,blurry)
    
    args.node_init_dim = node_features.shape[1]
    args.node_embedding_dim = node_features.shape[1]

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
    
    node_label = [-1 for i in range(all_data.n_unique_nodes + 1)]
    for i in range(len(label_src)):
        node_label[all_data.src[i]]=label_src[i]
        node_label[all_data.dst[i]]=label_dst[i]

    torch.cuda.set_device(args.device)
    device = 'cuda'
    logger.debug(str(args))

    # device = 'cpu'
    # args.device = device

    g_time=0

    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    computer_time_statics(train_data[0].src, train_data[0].dst, train_data[0].timestamps)

    args.time_shift = {'mean_time_shift_src': mean_time_shift_src, 'std_time_shift_src': std_time_shift_src, 
                       'mean_time_shift_dst': mean_time_shift_dst, 'std_time_shift_dst': std_time_shift_dst}

    neighbor_finder = get_neighbor_finder(all_data, False)
    sgnn = get_model(args, neighbor_finder, node_features, edge_features, label_src, label_dst)
    
    sgnn.to(device)
    
    # what is reset_graph?
    sgnn.reset_graph()

    logger.debug("./result/{}.txt".format(args.dataset+args.filename_add))
    LOSS = []
    val_acc, val_ap, val_f1 = [], [], []
    early_stopper = [EarlyStopMonitor(max_round=patience) for i in range(args.num_datasets+1)]
    test_best=[0 for i in range(args.num_datasets)]
    test_neighbor_finder=[]

    if not os.path.exists(f'./checkpoints/{args.model}/'):
        os.makedirs(f'./checkpoints/{args.model}/')

    cur_train_data = None
    cur_test_data = None
    cur_val_data = None

    for task in range(0,args.num_datasets):
        # initialize temporal graph
        if (task == 0) or (args.method != 'Joint'):
            cur_train_data = deepcopy(train_data[task])
            cur_test_data = deepcopy(test_data[task])
            cur_val_data = deepcopy(val_data[task])
        else:
            cur_train_data.add_data(train_data[task])
            cur_test_data.add_data(test_data[task])
            cur_val_data.add_data(val_data[task])

        train_neighbor_finder = get_neighbor_finder(cur_train_data, False)
        test_neighbor_finder.append(get_neighbor_finder(all_data, False, mask=test_data[task]))
        full_neighbor_finder = get_neighbor_finder(all_data, False)

        for e in range(args.n_epoch):
            print("task:",task,"epoch:",e)
            logger.debug('task {} , start {} epoch'.format(task,e))
            num_batch = math.ceil(len(cur_train_data.src) / args.batch_size)
            loss_value = 0
            Obj = 0
            Reward = 0
            sgnn.reset_graph()
            sgnn.set_neighbor_finder(train_neighbor_finder)
            sgnn.train()

            for i in range(num_batch):

                st_idx = i * args.batch_size
                ed_idx = min((i + 1) * args.batch_size, len(cur_train_data.src))

                src_batch = cur_train_data.src[st_idx:ed_idx]
                dst_batch = cur_train_data.dst[st_idx:ed_idx]
                edge_batch = cur_train_data.edge_idxs[st_idx:ed_idx]
                timestamp_batch = cur_train_data.timestamps[st_idx:ed_idx]

                data_dict = sgnn(src_batch, dst_batch, edge_batch, timestamp_batch, args.num_neighbors, dataset_idx=task)

                loss_value += data_dict['loss']
            
            sgnn.end_epoch()

            loss_value=loss_value / num_batch
            Obj=Obj / num_batch
            loss_mem1.append(loss_value)
            loss_mem2.append(Obj)
            print("train loss: %.4f"%loss_value)
            print("obj: %.4f"%(Obj))
            LOSS.append(loss_value)
            logger.debug("loss in whole dataset = {}".format(loss_value))

            # validation
            sgnn.eval()

            # sgnn.reset_graph(full_data[task].unique_nodes)
            sgnn.reset_graph()
            train_n_acc, train_n_ap, train_n_f1, train_m_acc = eval_prediction(sgnn, train_data[task], task, task, args.batch_size, 'train', uml, eval_avg, args.multihead, args.num_class_per_dataset)
            # train_n_acc, train_n_ap, train_n_f1, train_m_acc = eval_prediction(sgnn, cur_train_data, task, task, args.batch_size, 'train', uml, eval_avg, args.multihead, args.num_class_per_dataset)

            if full_n:
                sgnn.set_neighbor_finder(full_neighbor_finder)
            else:
                sgnn.set_neighbor_finder(test_neighbor_finder[task])
            val_n_acc, val_n_ap, val_n_f1, val_m_acc = eval_prediction(sgnn, val_data[task], task, task, args.batch_size, 'val', uml, eval_avg, args.multihead, args.num_class_per_dataset)


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

            test_n_acc, test_n_ap, test_n_f1, test_m_acc = eval_prediction(sgnn, test_data[task], task, task, args.batch_size, 'test', uml, eval_avg, args.multihead, args.num_class_per_dataset)
            print("train_acc:%.2f   val_acc:%.2f   test_acc:%.2f"%(train_n_acc,val_n_acc,test_n_acc))
            logger.debug("train_acc:%.2f   val_acc:%.2f   test_acc:%.2f"%(train_n_acc,val_n_acc,test_n_acc))
            val_acc.append(val_n_acc)
            val_ap.append(val_n_ap)
            val_f1.append(val_n_f1)
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
                if early_stopper[task].early_stop_check(val_n_ap, sgnn, args.model, train_memory_backup, time_now, task, train_IB_backup, train_PGen_backup) or e == args.n_epoch - 1:
                    logger.info('No improvement over {} epochs, stop training'.format(early_stopper[task].max_round))
                    logger.info(f'Loading the best model at epoch {early_stopper[task].best_epoch}')
                    best_model_path, _, _, _ = get_checkpoint_path(args.model, time_now, task, uml)
                    sgnn = torch.load(best_model_path)
                    logger.info(f'Loaded the best model at epoch {early_stopper[task].best_epoch} for inference')
                    sgnn.eval()
                    for k in range(task+1):
                        if full_n:
                            sgnn.set_neighbor_finder(full_neighbor_finder)
                        else:
                            sgnn.set_neighbor_finder(test_neighbor_finder[k])
                        best_model_path, best_mem_path, best_IB_path, best_PGen_path = get_checkpoint_path(args.model, time_now, k, uml)
                        if args.memory_replay:
                            best_mem = torch.load(best_mem_path)
                            sgnn.restore_memory(best_mem)
                        if args.model=='OTGNet':
                            if not dis_IB:
                                best_IB = torch.load(best_IB_path)
                                if args.dataset != 'reddit' and args.dataset != 'yelp':
                                    sgnn.restore_IB(best_IB)
                            if uml:
                                best_PGen = torch.load(best_PGen_path)
                                sgnn.restore_PGen(best_PGen)
                        if not args.multihead:
                            test_n_acc, test_n_ap, test_n_f1, test_m_acc = eval_prediction(sgnn, test_data[k], task, k, args.batch_size, 'test', uml, eval_avg, args.multihead, args.num_class_per_dataset)
                        else:
                            test_n_acc, test_n_ap, test_n_f1, test_m_acc = eval_prediction(sgnn, test_data[k], k, k, args.batch_size, 'test', uml, eval_avg, args.multihead, args.num_class_per_dataset)
                        test_best[k]=max(test_best[k], test_n_acc)
                        task_acc_vary[k][task]+=test_n_acc
                        task_acc_vary_cur[k][task]=test_n_acc
                        print("task %d: "%(k)+str(task_acc_vary_cur[k][task]))
                    break   
            
        if args.memory_replay:
            sgnn.end_dataset(train_data[task], args)
            # modify later! with specific operations
            # time_cost = sgnn.end_task()
            # g_time += time_cost


        
        # temp_dict = vars(args)
        # time_shift = temp_dict.pop('time_shift')
        # # args_str = vars(deepcopy(args))
        # torch.save(sgnn.state_dict(), f'./checkpoints/{args.model}/' + str(temp_dict))
        # args.time_shift = time_shift

    # test
    print('best performance: ',test_best)
    sgnn.eval()
    sgnn.set_neighbor_finder(full_neighbor_finder)
    avg_performance=[]
    avg_forgetting=[]
    for i in range(args.num_datasets):
        print("task:", i)
        test_acc, test_ap, test_f1 = task_acc_vary_cur[i][args.num_datasets-1], task_acc_vary_cur[i][args.num_datasets-1], task_acc_vary_cur[i][args.num_datasets-1]
        avg_performance.append(test_acc)
        avg_forgetting.append(test_best[i]-test_acc)
        task_acc_all[i]+=test_acc
        logger.debug("in test, acc = {}".format(test_acc))
        print("in test, acc = {}".format(test_acc))
        f.write("task %d, test_acc=%.2f, test_ap = %.2f, test_f1=%.2f"%(i, test_acc, test_ap, test_f1))
        f.write("\n")
    print('avg performance: ',avg_performance)
    print("Average performance: %.2f"%(np.array(avg_performance).mean()))
    print("Average forgetting: %.2f"%(np.array(avg_forgetting[:-1]).mean()))
    avg_performance_all.append(np.array(avg_performance).mean())
    avg_forgetting_all.append(np.array(avg_forgetting[:-1]).mean())
    f.write("Average performance: %.2f"%(np.array(avg_performance).mean()))
    f.write("\n")
    f.write("Average forgetting: %.2f"%(np.array(avg_forgetting[:-1]).mean()))
    f.write("\n")
    if mem_method=='triad':
        print("greedy_time: ", g_time/3600)
        f.write("greedy_time: "+str(g_time/3600))
        f.write("\n")
    all_time=time.time()-start_time
    print("all_time: ", all_time/3600)
    f.write("all_time: "+str(all_time/3600))
    f.write("\n")

    f.write('train loss:'+str(loss_mem1))
    f.write("\n")
    f.write('IB loss:'+str(loss_mem2))
    f.write("\n")
    plt.plot(list(range(0,len(loss_mem1))), loss_mem1, c='b')
    plt.savefig(Loss_path1)
    plt.show()
    plt.clf()
    plt.plot(list(range(0,len(loss_mem2))), loss_mem2, c='b')
    plt.savefig(Loss_path2)
    plt.show()
    plt.clf()

f.write(str(args))
f.write("\n")
f.write(time_now)
f.write("\n")
print("Overall AP: %.2f (%.2f)"%(np.array(avg_performance_all).mean(), np.array(avg_performance_all).std()))
print("Overall AF: %.2f (%.2f)"%(np.array(avg_forgetting_all).mean(), np.array(avg_performance_all).std()))
f.write("Overall AP: %.2f (%.2f)"%(np.array(avg_performance_all).mean(), np.array(avg_performance_all).std()))
f.write("\n")
f.write("Overall AF: %.2f (%.2f)"%(np.array(avg_forgetting_all).mean(), np.array(avg_forgetting_all).std()))
f.write("\n")
for i in range(args.num_datasets):
    print("Overall task %d performance: %.2f"%(i,task_acc_all[i]/rp_times))
    f.write("Overall task %d performance: %.2f"%(i,task_acc_all[i]/rp_times))
    f.write("\n")

c_list=['tomato','golden','pea','leaf','jade','bluish','violet','strawberry']
for i in range(args.num_datasets):
    for j in range(i,args.num_datasets):    
        task_acc_vary[i][j]/=rp_times
    f.write("task %d: "%(i)+str(task_acc_vary[i][i:]))
    f.write("\n")
    plt.plot(list(range(i,args.num_datasets)), task_acc_vary[i][i:], c=seaborn.xkcd_rgb[c_list[i]], marker='X', label='task%d'%(i))

plt.legend()
plt.savefig(Img_path)
plt.show()

f.write("\n ========================= \n")
f.close()
