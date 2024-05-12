import os

import torch
from functions import f1_score_calculation, load_query_n_gt, cosin_similarity, get_gt_legnth, \
    coo_matrix_to_nx_graph_efficient, evaluation
from search_functions import *
import argparse
import numpy as np
from tqdm import tqdm
from numpy import *
import time
import pickle
from EM import generate_sample_data
from EM import run as EMRun
from main import setup_seed, get_args


def parse_args():
    """
    Generate a parameters parser.
    """
    main_args = get_args('', '', outside = True)
    # with open('embedding_path.txt', 'r') as fr:
    #     embedding_path = fr.read()
    # parse parameters
    parser = argparse.ArgumentParser()
    # main parameters
    parser.add_argument('--dataset', type = str, default = 'acm', help = 'dataset name')
    parser.add_argument('--embedding_tensor_name', type = str, default = 'acm', help = 'embedding tensor name')
    parser.add_argument('--EmbeddingPath', type = str, default = main_args.embedding_path,
    # parser.add_argument('--EmbeddingPath', type = str, default = embedding_path,
                        help = 'embedding path')
    parser.add_argument('--topk', type = int, default = 400, help = 'the number of nodes selected.')
    parser.add_argument('--lammbda', type = float, default = 0.5, help = 'coef for private feature')
    parser.add_argument('--savefile', type = str, default = '_res.txt')

    return parser.parse_args()


def run(args):
    # args = parse_args()
    # print(args)

    if args.dataset.lower() in ['acm', 'imdb']:
        args.num_view = 2

    # 设置 embedding_tensor_name 的默认值
    if args.embedding_tensor_name is None:
        args.embedding_tensor_name = args.dataset

    # common_embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.embedding_tensor_name + 'common.npy'))
    common_embedding_tensor = []
    private_embedding_tensor = []
    for i in range(args.num_view):
        common_embedding_tensor.append(
            torch.from_numpy(np.load(args.EmbeddingPath + args.embedding_tensor_name + f'_com{i}.npy'))
        )
        private_embedding_tensor.append(
            torch.from_numpy(np.load(args.EmbeddingPath + args.embedding_tensor_name + f'_pri{i}.npy'))
        )
    common_embedding_tensor = torch.stack(common_embedding_tensor)
    private_embedding_tensor = torch.stack(private_embedding_tensor)
    # embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.embedding_tensor_name + '.npy'))

    # load queries and labels
    query, labels = load_query_n_gt("./dataset/", args.dataset, common_embedding_tensor[0].shape[0])
    gt_length = get_gt_legnth("./dataset/", args.dataset)

    # load adj
    # if args.dataset in {"photo", "cs"}:
    #     file_path = './dataset/' + args.dataset + '_dgl.pt'
    # else:
    #     file_path = './dataset/' + args.dataset + '_pyg.pt'
    # data_list = torch.load(file_path)
    # adj = data_list[0]
    graph_list = []
    for i in range(args.num_view):
        with open(os.path.join(args.EmbeddingPath, args.embedding_tensor_name + '_g{}.pickle'.format(i)), 'rb') as f:
            graph_list.append(pickle.load(f))

    start = time.time()
    # query_feature_common = torch.mm(query, common_embedding_tensor)
    query_num = torch.sum(query, dim = 1)
    # query_feature_common = torch.div(query_feature_common, query_num.view(-1, 1))

    query_feature_common = []
    query_feature_private = []
    for i in range(args.num_view):
        query_feature_c = torch.mm(query, common_embedding_tensor[i])
        query_feature_common.append(torch.div(query_feature_c, query_num.view(-1, 1)))

        query_feature = torch.mm(query, private_embedding_tensor[i])
        query_feature_private.append(torch.div(query_feature, query_num.view(-1, 1)))

    # query_feature = torch.mm(query, embedding_tensor)  # (query_num, embedding_dim)
    # query_num = torch.sum(query, dim = 1)
    # query_feature = torch.div(query_feature, query_num.view(-1, 1))

    # cosine similarity
    # query_score_common = cosin_similarity(query_feature_common, common_embedding_tensor)
    # query_score_common = torch.nn.functional.normalize(query_score_common, dim = 1, p = 1)

    query_score_common = []
    query_score_private = []
    for i in range(args.num_view):
        # query_score_c 解释:
        # 维度 (150, 3025) 其中, 第一行的 3025 个值, 表示第一个 query 和 3025 个节点分别求 cosine_similarity 的结果
        query_score_c = cosin_similarity(query_feature_common[i], common_embedding_tensor[i])  # (query_num, node_num)
        query_score_c = torch.nn.functional.normalize(query_score_c, dim = 1, p = 1)
        # query_score_private.append([item.tolist() for item in query_score])
        query_score_common.append(query_score_c.tolist())

        query_score = cosin_similarity(query_feature_private[i], private_embedding_tensor[i])  # (query_num, node_num)
        query_score = torch.nn.functional.normalize(query_score, dim = 1, p = 1)
        # query_score_private.append([item.tolist() for item in query_score])
        query_score_private.append(query_score.tolist())

    # print("query_score.shape: ", query_score_common[0].shape)
    # query_score_private = [item.tolist() for item in query_score_private]
    # query_score_common = query_score_common.tolist()

    # 使用 common + private 特征
    query_score_comPlusPri = np.array(query_score_common) + np.array(query_score_private) * args.lammbda

    use_pretrain = False
    if not use_pretrain:
        y_pred_list = []
        for v in range(args.num_view):
            y_pred = torch.zeros_like(query)
            for i in tqdm(range(query.shape[0])):
                query_index = (torch.nonzero(query[i]).squeeze()).reshape(-1)
                # selected_candidates = mwg_subgraph_heuristic(query_index.tolist(), query_score[i].tolist(), graph)
                # selected_candidates = greedy_search(
                #     query_index.tolist(), query_score_common[v][i], graph_list[v], args
                # )
                selected_candidates = greedy_search(
                    query_index.tolist(), query_score_comPlusPri[v][i], graph_list[v], args
                )
                    # selected_candidates = mwg_subgraph_heuristic_fast(
                    #     query_index.tolist(), [c[i] for c in query_score_common], [t[i] for t in query_score_private], graph, args
                    # )
                for j in range(len(selected_candidates)):
                    y_pred[i][selected_candidates[j]] = 1
            y_pred_list.append(y_pred)
        # np.load(args.EmbeddingPath + args.embedding_tensor_name + f'_pri{i}.npy')
        np.save(args.EmbeddingPath + args.embedding_tensor_name + f'_res.npy', torch.stack(y_pred_list).numpy())
    else:
        y_pred_list = torch.from_numpy(
            np.load(args.EmbeddingPath + args.embedding_tensor_name + f'_res.npy'))

    end = time.time()
    use_EM = True
    if use_EM:
        print('start EM...')
        start = time.time()
        '''
        类别未取反结果
        EM time: 114.09735488891602 s
        EM iter out num: 1
        recall:  tensor(0.0174) pre:  tensor(0.1147)
        F1 score by maximum weight gain (local search): 0.0303
        '''

        '''
        类别取反
        EM time: 113.28575801849365 s
        EM iter out num: 1
        recall:  tensor(0.9725) pre:  tensor(0.7451)
        F1 score by maximum weight gain (local search): 0.8437
        '''
        max_iter = 350
        print('max_iter:', max_iter)
        y_pred_list = [1 - item for item in y_pred_list]
        y_pred_list = torch.stack(y_pred_list)

        responses_list = generate_sample_data(y_pred_list)
        y_pred = torch.zeros_like(query)
        iter_out_number = 0
        for i, resp in enumerate(responses_list):
            # resp = 1 - resp
            proba_distri, max_iter_out = EMRun(resp, max_iter = max_iter)
            type_distri = np.argmax(proba_distri, axis = 1)
            y_pred[i] = torch.from_numpy(type_distri)
            iter_out_number += int(max_iter_out)
        end = time.time()
        print('EM time: {} s'.format(end - start))
        print('EM iter out num:', iter_out_number)
        labels = 1 - labels
        f1_score = f1_score_calculation(y_pred.int(), labels.int())
        print('common / private: 1 / {}'.format(args.lammbda))
        print("F1 score by maximum weight gain (local search): {:.4f}".format(f1_score))
        with open(args.EmbeddingPath + args.embedding_tensor_name + args.savefile, 'a') as fa:
            fa.write('common / private: 1 / {}, max iter: {}, max iter out: {}\n'.format(args.lammbda, max_iter, iter_out_number))
            fa.write("F1 score by maximum weight gain (local search): {:.4f}\n\n".format(f1_score))
        '''
        EM iter out num: 1
        recall:  tensor(0.2698) pre:  tensor(0.8853)
        F1 score by maximum weight gain (local search): 0.4136
        '''
    else:
        for v in range(args.num_view):
            print('view {}'.format(v))
            print("The local search using time: {:.4f}".format(end - start))
            print("The local search using time (one query): {:.4f}".format((end - start) / query_feature.shape[0]))
            f1_score = f1_score_calculation(y_pred_list[v].int(), labels.int())

            print("F1 score by maximum weight gain (local search): {:.4f}".format(f1_score))

            nmi, ari, jac = evaluation(y_pred_list[v].int(), labels.int())

            print("NMI score by maximum weight gain (local search): {:.4f}".format(nmi))
            print("ARI score by maximum weight gain (local search): {:.4f}".format(ari))
            print("JAC score by maximum weight gain (local search): {:.4f}".format(jac))
            print('*******************************************************')
        '''
        view 0
        The local search using time: 0.4983
        The local search using time (one query): 0.0033
        recall:  tensor(0.5356) pre:  tensor(0.6891)
        F1 score by maximum weight gain (local search): 0.6027
        NMI score by maximum weight gain (local search): 0.1595
        ARI score by maximum weight gain (local search): 0.2618
        JAC score by maximum weight gain (local search): 0.4313
        *******************************************************
        view 1
        The local search using time: 0.4983
        The local search using time (one query): 0.0033
        recall:  tensor(0.4333) pre:  tensor(0.8533)
        F1 score by maximum weight gain (local search): 0.5748
        NMI score by maximum weight gain (local search): 0.2213
        ARI score by maximum weight gain (local search): 0.2940
        JAC score by maximum weight gain (local search): 0.4033
        *******************************************************
        '''

if __name__ == "__main__":
    '''
    只用 common
    recall:  tensor(0.7032) pre:  tensor(0.7669)
    F1 score by maximum weight gain (local search): 0.7337
    NMI score by maximum weight gain (local search): 0.2979
    ARI score by maximum weight gain (local search): 0.4262
    JAC score by maximum weight gain (local search): 0.5794
    '''
    args = parse_args()
    beta_list = list(range(10))
    for beta in beta_list:
        if beta == 4:
            continue
        common_name = 'RetainU_NoCommonLinkLoss_CorrLossCoefBeta0{}'.format(int(beta))
        args.EmbeddingPath = f'./pretrain_result/ACM_{common_name}/'
        # lb_range = np.linspace(0.1, 1, 10)
        lb_range = np.linspace(-2, 2, 41)
        lb_range = np.linspace(0, 1, 11)
        # lb_range = np.linspace(-1.1, 2, 32)
        # lb_range = [0, 0.3, 0.4]
        # lb_range = [0.1, 0.2, 0.5, 0.6, 0.7]
        for lb in lb_range:
            args.lammbda = lb
            run(args)



