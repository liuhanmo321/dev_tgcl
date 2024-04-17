import os
import sys
from pathlib import Path


# dataset = sys.argv[1]
#dataset = 'reddit'
# dataset = 'yelp'
# dataset = 'taobao'

for dataset in ['amazon']:
# for dataset in ['amazon', 'reddit']:
# for dataset in ['amazon', 'yelp', 'reddit']:
    # model = 'DyGFormer' 
    model = 'TGAT'
    method = 'SubGraph'

    device = 0
    rp_times = 3
    debug_mode = 1

    select = 'none' 
    Path("./result/").mkdir(parents=True, exist_ok=True)

    if dataset=='taobao':
        lr=1e-3
        num_datasets=3
        num_class_per_dataset=30
        n_epoch=100
        bs = 600
        memory_size = 2000
    elif dataset=='yelp':
        lr=1e-5
        num_datasets=5        
        num_class_per_dataset=3
        n_epoch=100
        # n_epoch=1
        bs = 600
        memory_size = 1000
        replay_size = 500
    elif dataset=='reddit':
        lr=1e-5
        num_datasets=3
        num_class_per_dataset=5
        n_epoch = 100
        bs = 600
        memory_size = 1000
        replay_size = 500
    elif dataset=='amazon':
        lr=1e-5
        num_datasets=3
        num_class_per_dataset=3
        n_epoch = 100
        bs = 600
        memory_size = 1000
        replay_size = 500


    select_mode = 'error_min'
    error_min_distribution = 1
    error_min_loss = 1
    # error_min_distance_weight = 1.0
    error_min_loss_weight = 0.25
    # error_min_new_data_kept_ratio = 0.1
    error_min_distill = 1
    emb_distribution_distill_weight = 1
    error_min_hash = 0
    error_min_hash_threshold = 0.95

    partition = 'random'

    old_emb_distribution_distill = 0
    new_emb_distribution_distill = 1
    
    reg_gamma = 0.1

    distill = 1

    weight_learning_method = 'pred_diff'
    weight_reg_method = 'acc'

    eval_metric = 'ap'

    num_neighbors = 10

    event_weight_epochs = 50


    emb_proj = 0
    emb_distill = 0
    struct_distill = 0
    residual_distill = 0
    similarity_function = 'cos'

    distribution_measure = 'KLDiv'
    future_neighbor = 0
    old_data_weight = 1.0
    rand_neighbor = 1

    emb_distill_weight = 2.0
    struct_distill_weight = 2.0

    n_interval=5
    n_mc=0
    mem_size=10
    use_feature='fg' 
    use_memory=1
    use_time=5
    mem_method = 'triad' 
    
    filename_add = '_' + '_'.join([model, method])
    # filename_add += ("_"+model)

    # os['CUDA_VISIBLE_DEVICES']=device

    is_r=0
    blurry=0
    online=0
    use_IB=0
    dis_IB=0
    ch_IB = 'm' 
    pattern_rho=0
    class_balance=1
    eval_avg='edge'
    multihead=0
    feature_iter = 1
    patience = 20
    radius=0
    beta=0.3
    gamma=20
    uml=0
    pmethod='knn' 
    sk=1000
    full_n=0
    recover=1

    if debug_mode > 0:
        n_epoch = 1
        rp_times = 1
        event_weight_epochs = 2



    cmd = "python pi_train.py --batch_size {} --dataset {} --num_neighbors {} --n_epoch {} --lr {} --select {} --num_datasets {} --num_class_per_dataset {} --n_interval {} --n_mc {}".format(bs, dataset, num_neighbors,
    n_epoch, lr, select, num_datasets, num_class_per_dataset,n_interval, n_mc)
    cmd += " --model {}".format(model)
    cmd += " --method {}".format(method)
    cmd += " --memory_replay {}".format(use_memory)
    cmd += " --use_feature {}".format(use_feature)
    cmd += " --use_time {}".format(use_time)
    cmd += " --mem_method {}".format(mem_method)
    cmd += " --filename_add {}".format(filename_add)
    cmd += " --device_id {}".format(device)
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
    cmd += " --debug_mode {}".format(debug_mode)
    cmd += " --distill {}".format(distill)
    cmd += " --emb_proj {}".format(emb_proj)
    cmd += " --emb_distill {}".format(emb_distill)
    cmd += " --struct_distill {}".format(struct_distill)
    cmd += " --future_neighbor {}".format(future_neighbor)
    cmd += " --old_data_weight {}".format(old_data_weight)
    cmd += " --emb_distill_weight {}".format(emb_distill_weight)
    cmd += " --struct_distill_weight {}".format(struct_distill_weight)
    cmd += " --distribution_measure {}".format(distribution_measure)
    cmd += " --residual_distill {}".format(residual_distill)
    cmd += " --rand_neighbor {}".format(rand_neighbor)
    cmd += " --event_weight_epochs {}".format(event_weight_epochs)
    cmd += " --weight_learning_method {}".format(weight_learning_method)
    cmd += " --weight_reg_method {}".format(weight_reg_method)
    cmd += " --similarity_function {}".format(similarity_function)
    cmd += " --eval_metric {}".format(eval_metric)
    cmd += " --error_min_distribution {}".format(error_min_distribution)
    cmd += " --error_min_loss {}".format(error_min_loss)
    cmd += " --error_min_loss_weight {}".format(error_min_loss_weight)
    cmd += " --old_emb_distribution_distill {}".format(old_emb_distribution_distill)
    cmd += " --new_emb_distribution_distill {}".format(new_emb_distribution_distill)
    cmd += " --emb_distribution_distill_weight {}".format(emb_distribution_distill_weight)
    cmd += " --reg_gamma {}".format(reg_gamma)
    cmd += " --multihead {}".format(multihead)
    cmd += " --partition {}".format(partition)
    cmd += " --error_min_distill {}".format(error_min_distill)
    cmd += " --replay_size {}".format(replay_size)
    cmd += " --error_min_hash {}".format(error_min_hash)
    cmd += " --error_min_hash_threshold {}".format(error_min_hash_threshold)
    os.system(cmd)



    # if dataset=='reddit':
    #     lr=1e-4
    # elif dataset=='yelp':
    #     lr=1e-4
    # elif dataset=='taobao':
    #     lr=1e-3

    # if dataset == 'yelp':
    #     num_datasets=5
    # else:
    #     num_datasets=6
    # num_class_per_dataset=3

    # if dataset == 'taobao':
    #     num_datasets=3
    #     num_class_per_dataset=30
    #     n_epoch=100

    # if dataset == 'reddit':
    #     memory_size = 1000
    # elif dataset == 'yelp':
    #     memory_size = 100
    # else:
    #     memory_size = 2000