import numpy as np
import pandas as pd
import random
from copy import deepcopy
import pickle
# import torch
# import torch.nn.functional as F


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

        self.src_avail_mask = None
        self.dst_avail_mask = None

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
    
    def apply_mask(self, mask, src_mask, dst_mask):
        self.src = self.src[mask]
        self.dst = self.dst[mask]

        self.timestamps = self.timestamps[mask]
        self.edge_idxs = self.edge_idxs[mask]
        self.labels_src = self.labels_src[mask]
        self.labels_dst = self.labels_dst[mask]

        self.n_interactions = len(self.src)
        self.unique_nodes = set(self.src) | set(self.dst)
        self.n_unique_nodes = len(self.unique_nodes)

        self.src_avail_mask = src_mask[mask]
        self.dst_avail_mask = dst_mask[mask]


def get_past_inductive_data(DatasetName, n_task, n_class, blurry, train_frac = 0.8):

    random.seed(42)

    train_data = []
    val_data = []
    test_data = []
    full_data = []
    re_train_data = []
    re_val_data = []

    tmp_all_data = None

    if DatasetName == "reddit":
        prefix = "./data/{}_20day_200d_past_inductive_".format(DatasetName)
    elif DatasetName == 'amazon':
        prefix = "./data/{}_24day_200d_past_inductive_".format(DatasetName)
    elif DatasetName == 'yelp':
        prefix = "./data/{}_200d_past_inductive_".format(DatasetName)
    elif DatasetName == 'reddit_large':
        prefix = "./data/reddit_32topics_200d_past_inductive_"
    elif DatasetName == 'reddit_long':
        prefix = "./data/reddit_long_period_200d_past_inductive_"

    for i in range(n_task):
        tmp_full_graph = pd.read_csv("{}{}.csv".format(prefix, i))

        full_data.append(tmp_full_graph)

        if tmp_all_data is None:
            tmp_all_data = tmp_full_graph
        else:
            tmp_all_data = pd.concat([tmp_all_data, tmp_full_graph])
    
    class_train_node = []
    class_val_node = []
    class_test_node = []
    class_no_train_node = []

    for i in range(n_task * n_class):
        class_full_data_src = tmp_all_data[(tmp_all_data.label_u == i)]
        class_full_data_dst = tmp_all_data[(tmp_all_data.label_i == i)]

        class_unique_nodes = set(class_full_data_src.u.values) | set(class_full_data_dst.i.values)

        tmp_train_node_set = set(
            random.sample(class_unique_nodes, int(train_frac * len(class_unique_nodes)))
        )
        tmp_no_train_node_set = class_unique_nodes - tmp_train_node_set
        tmp_val_node_set = set(
            random.sample(tmp_no_train_node_set, int(0.5 * len(tmp_no_train_node_set)))
        )
        tmp_test_node_set = tmp_no_train_node_set - tmp_val_node_set

        class_train_node.append(tmp_train_node_set)
        class_val_node.append(tmp_val_node_set)
        class_test_node.append(tmp_test_node_set)
        class_no_train_node.append(tmp_no_train_node_set)
    
    total_edges = 0
    for i in range(n_task):
        src = full_data[i].u.values
        dst = full_data[i].i.values
        edge_idxs = full_data[i].idx.values
        labels_src = full_data[i].label_u.values
        labels_dst = full_data[i].label_i.values
        timestamps = full_data[i].ts.values

        total_edges += len(src)

        task_train_node_set = set.union(*class_train_node[: (i + 1) * n_class])
        task_val_node_set = set.union(*class_val_node[: (i + 1) * n_class])
        task_test_node_set = set.union(*class_test_node[: (i + 1) * n_class])
        print((i+1) * n_class)
        # task_no_train_node_set = set.union(*class_no_train_node[: (i + 1) * n_class])

        tmp_train_mask = [
            (src[j] in task_train_node_set and dst[j] in task_train_node_set)
            for j in range(len(src))
        ]
        tmp_val_mask = [
            ((src[j] in task_val_node_set and dst[j] not in task_test_node_set) \
             or dst[j] in task_val_node_set and src[j] not in task_test_node_set)
            for j in range(len(src))
        ]
        tmp_test_mask = [
            ((src[j] in task_test_node_set and dst[j] not in task_val_node_set) \
             or (dst[j] in task_test_node_set and src[j] not in task_val_node_set))
            for j in range(len(src))
        ]

        train_mask = tmp_train_mask
        val_mask = tmp_val_mask
        test_mask = tmp_test_mask
        
        train_data.append(
            Data(
                src[train_mask],
                dst[train_mask],
                timestamps[train_mask],
                edge_idxs[train_mask],
                labels_src[train_mask],
                labels_dst[train_mask],
            )
        )

        val_data.append(
            Data(
                src[val_mask],
                dst[val_mask],
                timestamps[val_mask],
                edge_idxs[val_mask],
                labels_src[val_mask],
                labels_dst[val_mask],
            )
        )

        test_data.append(
            Data(
                src[test_mask],
                dst[test_mask],
                timestamps[test_mask],
                edge_idxs[test_mask],
                labels_src[test_mask],
                labels_dst[test_mask],
            )
        )

        full_data[i] = Data(
            src,
            dst,
            timestamps,
            edge_idxs,
            labels_src,
            labels_dst,
        )

        # val_data[-1].induct_nodes = (
        #     val_data[-1].unique_nodes
        #     - train_data[-1].unique_nodes
        #     - test_data[-1].unique_nodes
        # )

        val_data[-1].induct_nodes = (
            val_data[-1].unique_nodes & task_val_node_set
        )

        # print(len(val_data[-1].induct_nodes))

        # test_data[-1].induct_nodes = (
        #     test_data[-1].unique_nodes
        #     - train_data[-1].unique_nodes
        #     - val_data[-1].unique_nodes
        # )

        test_data[-1].induct_nodes = (
            test_data[-1].unique_nodes & task_test_node_set
        )

        # print(len(test_data[-1].induct_nodes))

        print("Task", i, end=" ### ")
        print(
            "unique nodes:",
            "full",
            len(full_data[i].unique_nodes),
            "train",
            len(train_data[i].unique_nodes),
            "val",
            len(val_data[i].unique_nodes),
            "test",
            len(test_data[i].unique_nodes),
            "###",
            "interactions:",
            "full",
            full_data[i].n_interactions,
            "train",
            train_data[i].n_interactions,
            "val",
            val_data[i].n_interactions,
            "test",
            test_data[i].n_interactions,
        )

        label_set = sorted(set(train_data[i].labels_src) | set(test_data[i].labels_src) | set(test_data[i].labels_dst))
        for label in label_set:
            train_label_count = len(train_data[i].labels_src[train_data[i].labels_src == label]) + len(train_data[i].labels_dst[train_data[i].labels_dst == label])
            val_label_count = len(val_data[i].labels_src[val_data[i].labels_src == label]) + len(val_data[i].labels_dst[val_data[i].labels_dst == label])
            test_label_count = len(test_data[i].labels_src[test_data[i].labels_src == label]) + len(test_data[i].labels_dst[test_data[i].labels_dst == label])
            print("train label %d: %d" % (label, train_label_count), "val label %d: %d" % (label, val_label_count), "test label %d: %d" % (label, test_label_count))

    all_data = Data(
        tmp_all_data.u.values,
        tmp_all_data.i.values,
        tmp_all_data.ts.values,
        tmp_all_data.idx.values,
        tmp_all_data.label_u.values,
        tmp_all_data.label_i.values,
    )

    if DatasetName == 'reddit_large':
        node_features = np.load("./data/reddit_32topics_200d_past_inductive_node_feat.npy")
    elif DatasetName == 'reddit_long':
        node_features = np.load("./data/reddit_long_period_200d_past_inductive_node_feat.npy")
    else:
        node_features = np.load("./data/{}_past_inductive_node_feat.npy".format(DatasetName))
    edge_features = np.zeros((len(all_data.src), node_features.shape[1]))

    num_all_nodes = len(set(all_data.src) | set(all_data.dst))
    num_all_edges = len(all_data.src)

    print(num_all_nodes, total_edges)

    return (
        node_features,
        edge_features,
        full_data,
        train_data,
        val_data,
        test_data,
        all_data,
        re_train_data,
        re_val_data,
    )

def get_data(DatasetName, n_task, n_class, blurry):

    train_data = []
    val_data = []
    test_data = []
    full_data = []
    re_train_data = []
    re_val_data = []

    for i in range(n_task):
        train_graph = pd.read_csv("./data/{}_train_{}.csv".format(DatasetName, i))
        val_graph = pd.read_csv("./data/{}_val_{}.csv".format(DatasetName, i))
        test_graph = pd.read_csv("./data/{}_test_{}.csv".format(DatasetName, i))

        train_unique_nodes = pickle.load(open("./data/{}_unique_train_nodes_{}.pkl".format(DatasetName, i), "rb"))
        val_unique_nodes = pickle.load(open("./data/{}_unique_val_nodes_{}.pkl".format(DatasetName, i), "rb"))
        test_unique_nodes = pickle.load(open("./data/{}_unique_test_nodes_{}.pkl".format(DatasetName, i), "rb"))

        train_data.append(
            Data(
                train_graph.u.values,
                train_graph.i.values,
                train_graph.ts.values,
                train_graph.idx.values,
                train_graph.label_u.values,
                train_graph.label_i.values,
                train_unique_nodes
            )
        )

        val_data.append(
            Data(
                val_graph.u.values,
                val_graph.i.values,
                val_graph.ts.values,
                val_graph.idx.values,
                val_graph.label_u.values,
                val_graph.label_i.values,
                val_unique_nodes
            )
        )

        test_data.append(
            Data(
                test_graph.u.values,
                test_graph.i.values,
                test_graph.ts.values,
                test_graph.idx.values,
                test_graph.label_u.values,
                test_graph.label_i.values,
                test_unique_nodes
            )
        )

        temp_full_data = deepcopy(train_data[i])
        temp_full_data.add_data(val_data[i])
        temp_full_data.add_data(test_data[i])
        full_data.append(
            temp_full_data
        )
        
        if i == 0:
            re_train_data.append(train_data[i])
            re_val_data.append(val_data[i])
        else:
            re_train_data.append(deepcopy(re_train_data[i - 1]))
            re_val_data.append(deepcopy(re_val_data[i - 1]))
            re_train_data[i].add_data(train_data[i])
            re_val_data[i].add_data(val_data[i])

        print("Task", i, end=" ### ")
        print(
            "unique nodes:",
            "full",
            len(full_data[-1].unique_nodes),
            "train",
            len(train_data[-1].unique_nodes),
            "val",
            len(val_data[-1].unique_nodes),
            "test",
            len(test_data[-1].unique_nodes),
            "###",
            "interactions:",
            "full",
            full_data[i].n_interactions,
            "train",
            train_data[i].n_interactions,
            "val",
            val_data[i].n_interactions,
            "test",
            test_data[i].n_interactions,
        )

        label_set = sorted(set(train_data[i].labels_src) | set(test_data[i].labels_src) | set(test_data[i].labels_dst))
        for label in label_set:
            train_label_count = len(train_data[i].labels_src[train_data[i].labels_src == label]) + len(train_data[i].labels_dst[train_data[i].labels_dst == label])
            val_label_count = len(val_data[i].labels_src[val_data[i].labels_src == label]) + len(val_data[i].labels_dst[val_data[i].labels_dst == label])
            test_label_count = len(test_data[i].labels_src[test_data[i].labels_src == label]) + len(test_data[i].labels_dst[test_data[i].labels_dst == label])
            print("train label %d: %d" % (label, train_label_count), "val label %d: %d" % (label, val_label_count), "test label %d: %d" % (label, test_label_count))

    all_data = deepcopy(full_data[0])
    for i in range(1, n_task):
        all_data.add_data(full_data[i])
    
    # for i in range(len(class_num)):
    #     if i % n_class == 0:
    #         print("task %d:" % (i / n_class))
    #     print("interactions of class %d: %d" % (i, class_num[i]))
        
    node_features = np.load("./data/{}_node_feat.npy".format(DatasetName))
    edge_features = np.zeros((len(all_data.src), node_features.shape[1]))

    return (
        node_features,
        edge_features,
        full_data,
        train_data,
        val_data,
        test_data,
        all_data,
        re_train_data,
        re_val_data,
    )


def computer_time_statics(src, dst, timestamps):
    last_timestamp_src = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(src)):
        src_id = src[k]
        dst_id = dst[k]
        cur_timestamp = timestamps[k]
        if src_id not in last_timestamp_src.keys():
            last_timestamp_src[src_id] = 0
        if dst_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dst_id] = 0
        all_timediffs_src.append(cur_timestamp - last_timestamp_src[src_id])
        all_timediffs_dst.append(cur_timestamp - last_timestamp_dst[dst_id])
        last_timestamp_src[src_id] = cur_timestamp
        last_timestamp_dst[dst_id] = cur_timestamp
    assert len(all_timediffs_src) == len(src)
    assert len(all_timediffs_dst) == len(dst)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)
    return (
        mean_time_shift_src,
        std_time_shift_src,
        mean_time_shift_dst,
        std_time_shift_dst,
    )


# def get_data(DatasetName, n_task, n_class, blurry):
#     graph = pd.read_csv("./data/{}.csv".format(DatasetName))
#     # edge_features = np.load("./data/{}.npy".format(DatasetName))
#     node_features = np.load("./data/{}_node.npy".format(DatasetName))

#     src = graph.u.values
#     dst = graph.i.values
#     edge_idxs = graph.idx.values
#     labels_src = graph.label_u.values
#     labels_dst = graph.label_i.values
#     timestamps = graph.ts.values
#     timestamps = timestamps - timestamps[0] + 1

#     all_data = Data(src, dst, timestamps, edge_idxs, labels_src, labels_dst)
#     edge_features = np.zeros((len(edge_idxs), node_features.shape[1]))
#     # edge_features = np.array([[0] * 300 for i in range(len(edge_idxs))])

#     node_set = set(src) | set(dst)

#     print(
#         "The dataset has {} interactions, involving {} different nodes".format(
#             len(src), len(node_set)
#         )
#     )

#     task_full_mask = [[False] * len(src) for i in range(n_task)]
#     tmp = 0

#     heter = 0
#     node_first_time = {}
#     cos = 0
#     for i in range(len(src)):
#         mx_label = max(labels_src[i], labels_dst[i])
#         if blurry:
#             tmp = max(tmp, mx_label // n_class)
#         else:
#             tmp = mx_label // n_class
#         if tmp >= n_task:
#             break
#         task_full_mask[tmp][i] = True
#         cur_time = timestamps[i]
#         if src[i] not in node_first_time:
#             node_first_time[src[i]] = cur_time
#         if dst[i] not in node_first_time:
#             node_first_time[dst[i]] = cur_time

#     full_data = [
#         Data(
#             src[task_full_mask[i]],
#             dst[task_full_mask[i]],
#             timestamps[task_full_mask[i]],
#             edge_idxs[task_full_mask[i]],
#             labels_src[task_full_mask[i]],
#             labels_dst[task_full_mask[i]],
#         )
#         for i in range(n_task)
#     ]

#     task_full_node_set = [
#         set(src[task_full_mask[i]]) | set(dst[task_full_mask[i]]) for i in range(n_task)
#     ]

#     random.seed(2020)

#     train_mask = [[False] * len(src) for i in range(n_task)]
#     val_mask = [[False] * len(src) for i in range(n_task)]
#     test_mask = [[False] * len(src) for i in range(n_task)]

#     train_data = []
#     val_data = []
#     test_data = []

#     re_train_data = []
#     re_val_data = []

#     for i in range(n_task):
#         tmp_train_node_set = set(
#             random.sample(task_full_node_set[i], int(0.8 * len(task_full_node_set[i])))
#         )
#         tmp_no_train_node_set = set(task_full_node_set[i]) - tmp_train_node_set
#         tmp_val_node_set = set(
#             random.sample(tmp_no_train_node_set, int(0.5 * len(tmp_no_train_node_set)))
#         )
#         tmp_test_node_set = tmp_no_train_node_set - tmp_val_node_set

#         tmp_train_mask = [
#             (src[j] in tmp_train_node_set or dst[j] in tmp_train_node_set)
#             for j in range(len(src))
#         ]
#         tmp_val_mask = [
#             (src[j] in tmp_val_node_set or dst[j] in tmp_val_node_set)
#             for j in range(len(src))
#         ]
#         tmp_test_mask = [
#             (src[j] in tmp_test_node_set or dst[j] in tmp_test_node_set)
#             for j in range(len(src))
#         ]

#         tmp_no_train_src_mask = graph.u.map(lambda x: x in tmp_no_train_node_set).values
#         tmp_no_train_dst_mask = graph.i.map(lambda x: x in tmp_no_train_node_set).values
#         tmp_observed_edges_mask = np.logical_and(
#             ~tmp_no_train_src_mask, ~tmp_no_train_dst_mask
#         )

#         train_mask[i] = np.logical_and(tmp_train_mask, tmp_observed_edges_mask)
#         train_mask[i] = np.logical_and(train_mask[i], task_full_mask[i])
#         val_mask[i] = np.logical_and(tmp_val_mask, task_full_mask[i])
#         test_mask[i] = np.logical_and(tmp_test_mask, task_full_mask[i])

#         train_data.append(
#             Data(
#                 src[train_mask[i]],
#                 dst[train_mask[i]],
#                 timestamps[train_mask[i]],
#                 edge_idxs[train_mask[i]],
#                 labels_src[train_mask[i]],
#                 labels_dst[train_mask[i]],
#             )
#         )

#         val_data.append(
#             Data(
#                 src[val_mask[i]],
#                 dst[val_mask[i]],
#                 timestamps[val_mask[i]],
#                 edge_idxs[val_mask[i]],
#                 labels_src[val_mask[i]],
#                 labels_dst[val_mask[i]],
#             )
#         )

#         test_data.append(
#             Data(
#                 src[test_mask[i]],
#                 dst[test_mask[i]],
#                 timestamps[test_mask[i]],
#                 edge_idxs[test_mask[i]],
#                 labels_src[test_mask[i]],
#                 labels_dst[test_mask[i]],
#             )
#         )

#         val_data[-1].induct_nodes = (
#             val_data[-1].unique_nodes
#             - train_data[-1].unique_nodes
#             - test_data[-1].unique_nodes
#         )
#         test_data[-1].induct_nodes = (
#             test_data[-1].unique_nodes
#             - train_data[-1].unique_nodes
#             - val_data[-1].unique_nodes
#         )
#         print("Task", i, end=" ### ")
#         print(
#             "unique nodes:",
#             "full",
#             len(task_full_node_set[i]),
#             "train",
#             len(tmp_train_node_set),
#             "val",
#             len(tmp_val_node_set),
#             "test",
#             len(tmp_test_node_set),
#             "###",
#             "interactions:",
#             "full",
#             full_data[i].n_interactions,
#             "train",
#             train_data[i].n_interactions,
#             "val",
#             val_data[i].n_interactions,
#             "test",
#             test_data[i].n_interactions,
#         )

#         if i == 0:
#             re_train_data.append(train_data[i])
#             re_val_data.append(val_data[i])
#         else:
#             re_train_data.append(deepcopy(re_train_data[i - 1]))
#             re_val_data.append(deepcopy(re_val_data[i - 1]))
#             re_train_data[i].add_data(train_data[i])
#             re_val_data[i].add_data(val_data[i])

#         print(len(re_train_data[i].src))
#         re_val_data[-1].induct_nodes = (
#             re_val_data[-1].unique_nodes - re_train_data[-1].unique_nodes
#         )

#     class_num = [0 for i in range(n_task * n_class)]
#     tmp = 0
#     for i in range(len(labels_src)):
#         mx_label = max(labels_src[i], labels_dst[i])
#         if blurry:
#             tmp = max(tmp, mx_label // n_class)
#         else:
#             tmp = mx_label // n_class
#         if tmp >= n_task:
#             break
#         class_num[labels_src[i]] += 1
#         class_num[labels_dst[i]] += 1
#     for i in range(len(class_num)):
#         if i % n_class == 0:
#             print("task %d:" % (i / n_class))
#         print("interactions of class %d: %d" % (i, class_num[i]))

#     return (
#         node_features,
#         edge_features,
#         full_data,
#         train_data,
#         val_data,
#         test_data,
#         all_data,
#         re_train_data,
#         re_val_data,
#     )