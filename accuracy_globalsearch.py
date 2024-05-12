import torch
from functions import f1_score_calculation, load_query_n_gt, cosin_similarity, get_gt_legnth, evaluation
import argparse
import numpy as np
from tqdm import tqdm
from numpy import *
import time
import pickle
from EM import generate_sample_data
from EM import run as EMRun
from accelerate_EM import run as AccEMRun
import os
import cProfile
from main import get_dataset_view
import pandas as pd


def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters

    # embedding_path = r'E:\wyh\论文\安泰\PU_CS-main\DMG\pretrain_result\acm\RetainU_NoComLinkLoss_CorrLossCoefBeta10_LinkLosCoef0_NoContraLoss'
    # embedding_path = r'./pretrain_result/rm/RetainU_NoComLinkLoss_CorrLossCoefBeta10_LinkLosCoef0_NoContraLoss'
    # embedding_path = r'.\pretrain_result\rm\RetainU_NoComLinkLoss_CorrLossCoefBeta10_LinkLosCoef0'

    # embedding_path = r"E:\wyh\论文\安康\PU_CS-main\DMG\pretrain_result\ACM_RetainU_NoCommonLinkLoss_CorrLossCoefBeta010_LinkLosCoef00_addContraLossABBA"
    # 0.8732 lambda = 3 时, f1 提升了 0.7 ％
    # 8426 - 8501
    parser = argparse.ArgumentParser()
    parser.add_argument('-pt', '--pattern', type = str, default = 'both_std',
                        choices = ['com', 'pri', 'both', 'both_std', 'both_std_scores'])
    # main parameters
    parser.add_argument('--dataset', '-d', type=str, default='acm', help='dataset name')
    with open(f'./dataset_embedding_path/embedding_path_{parser.parse_args().dataset}.txt', 'r') as fr:
        embedding_path = fr.read()
    parser.add_argument('--embedding_tensor_name', type=str, help='embedding tensor name')
    parser.add_argument('--EmbeddingPath', type=str, default=embedding_path, help='embedding path')
    parser.add_argument('--topk', type=int, default=400, help='the number of nodes selected.')
    parser.add_argument('--lammbda', type = float, default = 0.5, help = 'coef for private feature')
    # parser.add_argument('--pattern', type = str, default = 'both_std_scores', choices = ['com', 'pri', 'both', 'both_std', 'both_std_scores'])
    # parser.add_argument('--pattern', type = str, default = 'both', choices = ['com', 'pri', 'both', 'both_std'])
    # parser.add_argument('-pt', '--pattern', type = str, default = 'both_std', choices = ['com', 'pri', 'both', 'both_std'])
    # parser.add_argument('--pattern', type = str, default = 'com', choices = ['com', 'pri', 'both', 'both_std'])
    # parser.add_argument('--pattern', type = str, default = 'pri', choices = ['com', 'pri', 'both', 'both_std'])

    parser.add_argument('--max_iter', type = int, default = 12, help = 'EM max iter')
    parser.add_argument('--write_result', type = bool, default = True)
    temp = parser.parse_args()
    parser.add_argument('--savefile', type = str, default = '_globalsearch_{}_maxIter{}.txt'.format(
        temp.pattern, temp.max_iter))

    return parser.parse_args()


def main():
    args = parse_args()

    # pattern_list = ['both_std_scores', 'com', 'pri', 'both', 'both_std', ]
    pattern_list = ['both_std_scores', 'com', 'pri']
    result_dict = {
        # 'both': [],
        # 'both_std': [],
        'both_std_scores': []
    }
    for pt in pattern_list:
        args.pattern = pt
        args.savefile = '_globalsearch_{}_maxIter{}.txt'.format(args.pattern, args.max_iter)
        print('embedding path:', args.EmbeddingPath)
        print('pattern:', args.pattern)
        lb_range = np.linspace(-2, 2, 41)
        lb_range = [-1, 0]
        if 'both' in args.pattern:
            # lb_range = np.linspace(0, 1, 11)
            # lb_range = np.linspace(-2, 2, 41)
            for lb in lb_range:
                args.lammbda = lb
                print(args.dataset)
                res = run(args)
                result_dict[pt].append(res.item())
        else:
            # lb_range = [0]
            args.lammbda = 0
            result_dict[pt] = ['' for _ in range(len(lb_range))]
            result_dict[pt][len(lb_range) // 2] = run(args).item()

        index = lb_range

        # for lb in lb_range:
        #     args.lammbda = lb
        #     res = run(args)
        #     result_dict[pt].append(res)

    df = pd.DataFrame(result_dict, index = index)
    # df = df.rename(columns = {'both': 'simple addition', 'both_std': 'standardize features',
    #                           'both_std_scores': 'standardize scores'})
    df = df.rename(columns = {'both_std_scores': 'standardize scores'})
    df.to_excel(os.path.join(args.EmbeddingPath, f'aaa_{args.dataset}.xlsx'))



def subgraph_density_controled(candidate_score, graph_score):
    
    weight_gain = (sum(candidate_score)-sum(graph_score)*(len(candidate_score)**1)/(len(graph_score)**1))/(len(candidate_score)**0.50)
    return weight_gain

def GlobalSearch(query_index, graph_score, graph, args):

    candidates = query_index
    selected_candidate = candidates

    graph_score=np.array(graph_score)
    max2min_index = np.argsort(-graph_score)
    
    startpoint = 0
    endpoint = int(0.50*len(max2min_index))
    if endpoint >= 10000:
        endpoint = 10000
    
    while True:
        candidates_half = query_index+[max2min_index[i] for i in range(0, int((startpoint+endpoint)/2))]
        candidate_score_half = [graph_score[i] for i in candidates_half]
        candidates_density_half = subgraph_density_controled(candidate_score_half, graph_score)

        candidates = query_index+[max2min_index[i] for i in range(0, endpoint)]
        candidate_score = [graph_score[i] for i in candidates]
        candidates_density = subgraph_density_controled(candidate_score, graph_score)

        if candidates_density >= candidates_density_half:
            startpoint = int((startpoint+endpoint)/2)
            endpoint = endpoint
        else:
            startpoint = startpoint
            endpoint = int((startpoint+endpoint)/2)
        
        if startpoint == endpoint or startpoint+1 == endpoint:
            break

    selected_candidate = query_index+[max2min_index[i] for i in range(0, startpoint)] 
    
    return selected_candidate


def run(args, save_result = True):
    # args = parse_args()
    # print(args)

    get_dataset_view(args)
    # if args.dataset.lower() in ['acm', 'imdb']:
    #     args.num_view = 2
    # elif args.dataset.lower() in ['dblp', 'freebase']:
    #     args.num_view = 3
    # elif args.dataset in ['terrorist']:
    #     args.num_view = 14
    # elif args.dataset in ['rm']:
    #     args.num_view = 10
    # 设置 embedding_tensor_name 的默认值
    if args.embedding_tensor_name is None:
        args.embedding_tensor_name = args.dataset

    common_embedding_tensor = []
    private_embedding_tensor = []
    for i in range(args.num_view):
        common_embedding_tensor.append(
            torch.from_numpy(np.load(
                os.path.join(args.EmbeddingPath, args.embedding_tensor_name + f'_com{i}.npy')))
            # np.load(args.EmbeddingPath + args.embedding_tensor_name + f'_com{i}.npy')
        )
        private_embedding_tensor.append(
            torch.from_numpy(np.load(
                os.path.join(args.EmbeddingPath, args.embedding_tensor_name + f'_pri{i}.npy')))
            # np.load(args.EmbeddingPath + args.embedding_tensor_name + f'_pri{i}.npy')
        )
    common_embedding_tensor = torch.stack(common_embedding_tensor)
    private_embedding_tensor = torch.stack(private_embedding_tensor)
    # common_embedding_tensor = np.stack(common_embedding_tensor)
    # private_embedding_tensor = np.stack(private_embedding_tensor)
    # if args.pattern == 'both_std':
    if args.pattern == 'both_std':
        for i in range(args.num_view):
            common_embedding_tensor[i] = torch.from_numpy(standardize_columns(common_embedding_tensor[i].numpy()))
            private_embedding_tensor[i] = torch.from_numpy(standardize_columns(private_embedding_tensor[i].numpy()))
            # private_embedding_tensor[i] = standardize_columns(private_embedding_tensor[i])

    # load queries and labels
    query, labels = load_query_n_gt("./dataset/", args.dataset, common_embedding_tensor[0].shape[0])
    gt_length = get_gt_legnth("./dataset/", args.dataset)
    # torch.Size([150, 3025])

    graph_list = []
    for i in range(args.num_view):
        with open(os.path.join(args.EmbeddingPath, args.embedding_tensor_name + '_g{}.pickle'.format(i)), 'rb') as f:
            graph_list.append(pickle.load(f))

    start = time.time()
    # 所有的 query 节点特征求和
    query_num = torch.sum(query, dim=1)

    query_feature_common = []
    query_feature_private = []
    for i in range(args.num_view):
        query_feature_c = torch.mm(query, common_embedding_tensor[i])
        query_feature_common.append(torch.div(query_feature_c, query_num.view(-1, 1)))

        query_feature = torch.mm(query, private_embedding_tensor[i])
        query_feature_private.append(torch.div(query_feature, query_num.view(-1, 1)))

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

    if args.pattern == 'both_std_scores':
        for i in range(args.num_view):
            query_score_common[i] = torch.from_numpy(standardize_rows(np.array(query_score_common[i])))
            query_score_private[i] = torch.from_numpy(standardize_rows(np.array(query_score_private[i])))

    # print("query_score.shape: ", query_score_common[0].shape)
    # query_score_private = [item.tolist() for item in query_score_private]
    # query_score_common = query_score_common.tolist()

    # 使用 common + private 特征
    if args.pattern == 'com':
        query_score_result = np.array(query_score_common)
    elif args.pattern == 'pri':
        query_score_result = np.array(query_score_private)
    elif 'both' in args.pattern:
        query_score_result = np.array(query_score_common) + np.array(query_score_private) * args.lammbda
    # query_score_comPlusPri = np.array(query_score_common) + np.array(query_score_private) * args.lammbda

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
                selected_candidates = GlobalSearch(
                    query_index.tolist(), query_score_result[v][i], graph_list[v], args
                )
                    # selected_candidates = mwg_subgraph_heuristic_fast(
                    #     query_index.tolist(), [c[i] for c in query_score_common], [t[i] for t in query_score_private], graph, args
                    # )
                for j in range(len(selected_candidates)):
                    y_pred[i][selected_candidates[j]] = 1
            y_pred_list.append(y_pred)
        # np.load(args.EmbeddingPath + args.embedding_tensor_name + f'_pri{i}.npy')
        np.save(os.path.join(
            args.EmbeddingPath, args.embedding_tensor_name + f'_res.npy'), torch.stack(y_pred_list).numpy())

        # y_pred = torch.zeros_like(query_score)
        # for i in tqdm(range(query_score.shape[0])):
        #     query_index = (torch.nonzero(query[i]).squeeze()).reshape(-1)
        #
        #     selected_candidates = GlobalSearch(query_index.tolist(), query_score[i].tolist())
        #     for j in range(len(selected_candidates)):
        #         y_pred[i][selected_candidates[j]] = 1
    end = time.time()
    use_EM = True
    if use_EM:
        print('start EM...')

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
        max_iter = args.max_iter
        print('max_iter:', max_iter)
        y_pred_list = [1 - item for item in y_pred_list]
        y_pred_list = torch.stack(y_pred_list)

        srtc = time.time()
        responses_list = generate_sample_data(y_pred_list)
        ertc = time.time()
        print('generate data: {} s'.format(ertc - srtc))

        y_pred = torch.zeros_like(query)
        iter_out_number = 0
        start = time.time()
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
        print("F1 score by maximum weight gain (global search): {:.4f}".format(f1_score))
        if args.write_result:
            with open(os.path.join(
                    args.EmbeddingPath, args.embedding_tensor_name + args.savefile), 'a') as fa:
                fa.write('common / private: 1 / {}, max iter: {}, max iter out: {}\n'.format(args.lammbda, max_iter,
                                                                                             iter_out_number))
                fa.write("F1 score by maximum weight gain (global search): {:.4f}\n\n".format(f1_score))

    return f1_score

    # print("The global search using time: {:.4f}".format(end-start))
    # print("The global search using time (one query): {:.4f}".format((end-start)/query_feature.shape[0]))
    # f1_score = f1_score_calculation(y_pred.int(), labels.int())
    #
    # print("F1 score by maximum weight gain: {:.4f}".format(f1_score))
    #
    # nmi, ari, jac = evaluation(y_pred.int(), labels.int())
    #
    # print("NMI score by maximum weight gain: {:.4f}".format(nmi))
    # print("ARI score by maximum weight gain: {:.4f}".format(ari))
    # print("JAC score by maximum weight gain: {:.4f}".format(jac))


def standardize_rows(arr):
    # 计算每列的均值和标准差
    means = np.mean(arr, axis = 1)
    stds = np.std(arr, axis = 1)

    # 防止除以零的情况，对标准差为零的列进行处理
    stds[stds == 0] = 1

    # 对每一列进行标准化
    standardized_arr = (arr - means[:, np.newaxis]) / stds[:, np.newaxis]

    return standardized_arr


def standardize_columns(arr):
    # 计算每列的均值和标准差
    means = np.mean(arr, axis = 0)
    stds = np.std(arr, axis = 0)

    # 防止除以零的情况，对标准差为零的列进行处理
    stds[stds == 0] = 1

    # 对每一列进行标准化
    standardized_arr = (arr - means) / stds

    return standardized_arr

def measure_time():
    lb = 1
    args = parse_args()
    print(args.EmbeddingPath)
    args.lammbda = lb
    run(args, save_result = False)


if __name__ == '__main__':
    main()

    # measure time
    # cProfile.run('measure_time()')

    # run one
    # lb = 1
    # args = parse_args()
    # print(args.EmbeddingPath)
    # args.lammbda = lb
    # run(args)
