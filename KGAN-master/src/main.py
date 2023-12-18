import argparse
import numpy as np
import json
from data_loader import load_data
from train import train

np.random.seed(2020)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--dim', type=int, default=32, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
parser.add_argument('--n_relations', type=int, default=25, help='numbers of relations')
parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=20, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=32, help='size of aggregate set for each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
parser.add_argument('--show_topk', type=bool, default=True, help='top_k')
parser.add_argument('--show_loss', type=bool, default=False, help='loss')


args = parser.parse_args()

data_info = load_data(args)
train(args, data_info, args.show_loss)


