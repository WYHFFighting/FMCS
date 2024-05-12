import os

import torch
from functions import f1_score_calculation, load_query_n_gt, cosin_similarity, get_gt_legnth, \
    coo_matrix_to_nx_graph_efficient, evaluation
from search_functions import mwg_subgraph_heuristic_fast
import argparse
import numpy as np
from tqdm import tqdm
from numpy import *
import time


def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()
    # main parameters
    parser.add_argument('--dataset', type = str, default = 'acm', help = 'dataset name')
    parser.add_argument('--embedding_tensor_name', type = str, default = 'acm_Node_', help = 'embedding tensor name')
    parser.add_argument('--EmbeddingPath', type = str, default = './pretrain_result/ACM_NoSVD_NoLossCorrAndContra/',
                        help = 'embedding path')
    parser.add_argument('--topk', type = int, default = 400, help = 'the number of nodes selected.')

    return parser.parse_args()


def run():
    args = parse_args()
    print(args)

    if args.dataset.lower() in ['acm', 'imdb']:
        args.num_view = 2

    # 设置 embedding_tensor_name 的默认值
    if args.embedding_tensor_name is None:
        args.embedding_tensor_name = args.dataset

    common_embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.embedding_tensor_name + 'common.npy'))
    # common_embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.embedding_tensor_name + f'view0.npy'))
    private_embedding_tensor = []
    for i in range(args.num_view):
        private_embedding_tensor.append(
            torch.from_numpy(np.load(args.EmbeddingPath + args.embedding_tensor_name + f'view{i}.npy'))
        )
    private_embedding_tensor = torch.stack(private_embedding_tensor)
    # embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.embedding_tensor_name + '.npy'))

    # load queries and labels
    query, labels = load_query_n_gt("./dataset/", args.dataset, common_embedding_tensor.shape[0])
    gt_length = get_gt_legnth("./dataset/", args.dataset)

    # load adj
    # if args.dataset in {"photo", "cs"}:
    #     file_path = './dataset/' + args.dataset + '_dgl.pt'
    # else:
    #     file_path = './dataset/' + args.dataset + '_pyg.pt'
    # data_list = torch.load(file_path)
    # adj = data_list[0]

    adj = torch.from_numpy(np.load(os.path.join(args.EmbeddingPath, f'{args.dataset}_global_adj.npy')))
    graph = coo_matrix_to_nx_graph_efficient(adj)

    start = time.time()

    query_feature_common = torch.mm(query, common_embedding_tensor)
    query_num = torch.sum(query, dim = 1)
    query_feature_common = torch.div(query_feature_common, query_num.view(-1, 1))

    query_feature_private = []
    for i in range(args.num_view):
        query_feature = torch.mm(query, private_embedding_tensor[i])
        query_feature_private.append(torch.div(query_feature, query_num.view(-1, 1)))

    # query_feature = torch.mm(query, embedding_tensor)  # (query_num, embedding_dim)
    # query_num = torch.sum(query, dim = 1)
    # query_feature = torch.div(query_feature, query_num.view(-1, 1))

    # cosine similarity
    query_score_common = cosin_similarity(query_feature_common, common_embedding_tensor)
    query_score_common = torch.nn.functional.normalize(query_score_common, dim = 1, p = 1)

    query_score_private = []
    for i in range(args.num_view):
        query_score = cosin_similarity(query_feature_private[i], private_embedding_tensor[i])  # (query_num, node_num)
        query_score = torch.nn.functional.normalize(query_score, dim = 1, p = 1)
        # query_score_private.append([item.tolist() for item in query_score])
        query_score_private.append(query_score.tolist())

    print("query_score.shape: ", query_score_common.shape)
    # query_score_private = [item.tolist() for item in query_score_private]
    # query_score_common = query_score_common.tolist()
    y_pred = torch.zeros_like(query_score_common)
    for i in tqdm(range(query_score_common.shape[0])):
        query_index = (torch.nonzero(query[i]).squeeze()).reshape(-1)
        # selected_candidates = mwg_subgraph_heuristic(query_index.tolist(), query_score[i].tolist(), graph)
        selected_candidates = mwg_subgraph_heuristic_fast(
            query_index.tolist(), query_score_common[i].tolist(), [t[i] for t in query_score_private], graph, args
        )
        for j in range(len(selected_candidates)):
            y_pred[i][selected_candidates[j]] = 1

    end = time.time()
    print("The local search using time: {:.4f}".format(end - start))
    print("The local search using time (one query): {:.4f}".format((end - start) / query_feature.shape[0]))
    f1_score = f1_score_calculation(y_pred.int(), labels.int())

    print("F1 score by maximum weight gain (local search): {:.4f}".format(f1_score))

    nmi, ari, jac = evaluation(y_pred.int(), labels.int())

    print("NMI score by maximum weight gain (local search): {:.4f}".format(nmi))
    print("ARI score by maximum weight gain (local search): {:.4f}".format(ari))
    print("JAC score by maximum weight gain (local search): {:.4f}".format(jac))


if __name__ == "__main__":
    '''
    只用 common
    recall:  tensor(0.7032) pre:  tensor(0.7669)
    F1 score by maximum weight gain (local search): 0.7337
    NMI score by maximum weight gain (local search): 0.2979
    ARI score by maximum weight gain (local search): 0.4262
    JAC score by maximum weight gain (local search): 0.5794

    '''
    run()



