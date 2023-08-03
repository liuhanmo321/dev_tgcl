import os
import sys
from pathlib import Path


# dataset = sys.argv[1]
#dataset = 'reddit'
dataset = 'yelp'
#dataset = 'taobao'
model = 'TGAT' 
method = 'SubGraph'
n_epoch = 400

device=1
rp_times=3

select = 'none' 
Path("./result/").mkdir(parents=True, exist_ok=True)
bs = 300

if dataset=='taobao':
    bs = 600

num_neighbors=5
if dataset=='reddit':
    lr=1e-4
elif dataset=='yelp':
    lr=5e-3
elif dataset=='taobao':
    lr=1e-3

if dataset == 'yelp':
    num_datasets=5
else:
    num_datasets=6
num_class_per_dataset=3

if dataset == 'taobao':
    num_datasets=3
    num_class_per_dataset=30
    n_epoch=100

select_mode = 'random'
memory_size = 100

n_interval=5
n_mc=0
mem_size=10
use_feature='fg' 
use_memory=1
use_time=5
mem_method = 'triad' 
filename_add = 'test'
filename_add += ("_"+model)

# os['CUDA_VISIBLE_DEVICES']=device

is_r=0
blurry=0
online=0
use_IB=0
dis_IB=0
ch_IB = 'm' 
pattern_rho=0
class_balance=1
eval_avg='node'
multihead=True
feature_iter = 1
patience = 100
radius=0
beta=0.3
gamma=20
uml=0
pmethod='knn' 
sk=1000
full_n=0
recover=1



cmd = "python train.py --batch_size {} --dataset {} --num_neighbors {} --n_epoch {} --lr {} --select {} --num_datasets {} --num_class_per_dataset {} --n_interval {} --n_mc {}".format(bs, dataset, num_neighbors,
n_epoch, lr, select, num_datasets, num_class_per_dataset,n_interval, n_mc)
cmd += " --model {}".format(model)
cmd += " --method {}".format(method)
cmd += " --memory_replay {}".format(use_memory)
cmd += " --use_feature {}".format(use_feature)
cmd += " --use_time {}".format(use_time)
cmd += " --mem_method {}".format(mem_method)
cmd += " --filename_add {}".format(filename_add)
cmd += " --device {}".format(device)
cmd += " --mem_size {}".format(mem_size)
cmd += " --rp_times {}".format(rp_times)
cmd += " --is_r {}".format(is_r)
cmd += " --blurry {}".format(blurry)
cmd += " --online {}".format(online)
cmd += " --use_IB {}".format(use_IB)
cmd += " --pattern_rho {}".format(pattern_rho)
cmd += " --class_balance {}".format(class_balance)
cmd += " --eval_avg {}".format(eval_avg)
cmd += " --feature_iter {}".format(feature_iter)
cmd += " --patience {}".format(patience)
cmd += " --radius {}".format(radius)
cmd += " --beta {}".format(beta)
cmd += " --gamma {}".format(gamma)
cmd += " --uml {}".format(uml)
cmd += " --sk {}".format(sk)
cmd += " --full_n {}".format(full_n)
cmd += " --recover {}".format(recover)
cmd += " --pmethod {}".format(pmethod)
cmd += " --dis_IB {}".format(dis_IB)
cmd += " --ch_IB {}".format(ch_IB)
cmd += " --select_mode {}".format(select_mode)
cmd += " --memory_size {}".format(memory_size)
os.system(cmd)