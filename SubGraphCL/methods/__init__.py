import torch
import torch.nn as nn
from typing import List

from . import Finetune, SubGraph, Joint, LwF, EWC

import importlib

module_list = {
    'Finetune': Finetune,
    'LwF': LwF,
    'SubGraph': SubGraph,
    'Joint': Joint,
    'EWC': EWC
}

def get_model(args, neighbor_finder, node_features, edge_features, src_label, dst_label):
    try:
        # module = importlib.import_module(args.method)
        # module = globals()[args.method]
        # module = Finetune
        method = getattr(module_list[args.method], args.method)
        return method(args, neighbor_finder, node_features, edge_features, src_label, dst_label)
    except ImportError:
        print(f"Module '{args.method}' not found or class is incorrect.")
    return None