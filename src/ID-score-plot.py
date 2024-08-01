

import os
import re
import copy
import torch
import random
import argparse
import numpy as np
import seaborn as sns
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    AutoTokenizer,
    LlamaForCausalLM
)

from utils import indexing
from utils import utils
from utils.utils import set_seed, ReadLineFromFile, load_pickle
from utils.score import Diversity_Score, Memorization_Score

def get_dict_from_lines(lines):
    index_map = dict()
    for line in lines:
        info = line.split(" ")
        index_map[info[0]] = info[1]
    return index_map

def get_emb(emb_dict, tokenizer, asin2id, method='mean'):
    '''
    emb_dict: [num_of_token, emb_dim]
    asin2id: Dict{asin: id}
    Return: Dict{asin: emb}
    '''
    result_emb_dict = {}
    for asin, id_val in asin2id.items():
        tokens = tokenizer.encode(id_val)
        # Look up embeddings for each token
        token_embeddings = [emb_dict[token] for token in tokens]

        # If no embeddings found, skip this entry
        if not token_embeddings:
            print(asin, id_val)
            continue
        # Combine embeddings based on the specified method
        if method == 'mean':
            combined_emb = np.mean(token_embeddings, axis=0)
        elif method == 'sum':
            combined_emb = np.sum(token_embeddings, axis=0)
        else:
            raise ValueError("Invalid method. Use 'mean' or 'sum'.")

        result_emb_dict[asin] = combined_emb
    return result_emb_dict

def get_new_tokens(args):
    # load user sequence data
    user_sequence = ReadLineFromFile(os.path.join(args.data_path, args.dataset, 'user_sequence.txt'))
    user_sequence_dict = indexing.construct_user_sequence_dict(user_sequence)

    if args.item_indexing == 'sequential':
        _, args.item_map, args.user_map = indexing.sequential_indexing(args.data_path, args.dataset, user_sequence_dict, args.sequential_order)
    elif args.item_indexing == 'random':
        _, args.item_map, args.user_map = indexing.random_indexing(args.data_path, args.dataset, user_sequence_dict)
    if args.item_indexing == 'collaborative':
        _, args.item_map, args.user_map = indexing.collaborative_indexing(args.data_path, args.dataset, user_sequence_dict,
                                            args.collaborative_token_size, args.collaborative_cluster,
                                            args.collaborative_last_token, args.collaborative_float32)
        args.new_token = []
        for idx in list(args.item_map.values()):
            args.new_token += re.findall(r'\<.*?\>', idx)
    if args.item_indexing == 'independent':
        _, args.item_map, args.user_map = indexing.independent_indexing(args.data_path, args.dataset, user_sequence_dict)
        args.new_token = []
        for idx in list(args.item_map.values()):
            args.new_token += re.findall(r'\<.*?\>', idx)

    elif args.item_indexing == 'metapath':
        _, args.item_map, args.user_map = indexing.metapath_indexing(args.data_path, args.dataset, user_sequence_dict,
                                        args.metapath_cluster_method, args.metapath_cluster_num,
                                        args.item_indexing, args.user_indexing)
        args.new_token = []
        for idx in list(args.item_map.values()):
            args.new_token += re.findall(r'\<.*?\>', idx)
        for idx in list(args.user_map.values()):
            args.new_token += re.findall(r'\<.*?\>', idx)

def find_index(item_asins, asin1, asin2):
    i = item_asins.index(asin1)
    j = item_asins.index(asin2)
    
    # Ensure i < j to match how the similarity matrix is filled
    if i > j:
        i, j = j, i
    
    num_items = len(item_asins)
    k = (i * (2 * num_items - i - 1)) // 2 + (j - i - 1)
    return k


def find_specific_user_item_subsets(rating_dict, target_items=[]):
    # Check each combination to find a valid subset
    valid_subsets = []
    for user, items in rating_dict.items():
        flag = 1
        for i in target_items:
            if i not in items.keys():
                flag = 0
                break
        if flag == 1:
            valid_subsets.append(user)

    # Return all found subsets or indicate none found
    return valid_subsets



def plot_pca(emb, labels, file='tmp.png', method='tsne', plot_color=['#DAE3F3', '#F9C756']):
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE

    concatenated_embeddings = np.concatenate(emb)
    labels_name = labels

    tsne = TSNE(n_components=2, random_state=0)
    embedded_tokens_2d = tsne.fit_transform(concatenated_embeddings)
    # from sklearn.decomposition import PCA
    # scaler = StandardScaler()
    # concatenated_embeddings = scaler.fit_transform(concatenated_embeddings)
    # pca = PCA(n_components=2)
    # embedded_tokens_2d = pca.fit_transform(concatenated_embeddings)
    # # Access the eigenvalues and eigenvectors
    # eigenvalues = pca.explained_variance_
    # print(eigenvalues)

    # Separate the embeddings for visualization
    labels = []
    for i in range(len(labels_name)):
        labels.append(i * np.ones((emb[i].shape[0])))
    # Concatenate labels
    labels = np.concatenate(labels)

    # Plot the embeddings in 2D
    plt.figure(figsize=(8, 6))

    for i, label in enumerate(labels_name):
        indices = labels == i
        plt.scatter(
            embedded_tokens_2d[indices, 0],
            embedded_tokens_2d[indices, 1],
            label=label,
            c=plot_color[i],
            # alpha=0.7  # Adjust alpha for overlap
            )

    plt.savefig(file, dpi=1200)


def plot_heatmap(item_asins, user_asins, norm_exp_rating_matrix, rating_dict, prefix='./plt/'):

    last_num = 5

    user_sequence = ReadLineFromFile(os.path.join(args.data_path, args.dataset, 'user_sequence.txt'))
    user_sequence_dict = indexing.construct_user_sequence_dict(user_sequence)

    sampled_user_asins_sid = ['A1A5TCGIV52I1E', 'A1A8CBN2HWT4VP', 'A1AJWJGB89GSIL', 'A1AW460LGI1QER', 'A1AXDE60UT3N9L', 'A1AWL9JASMG904']
    sampled_item_asins_sid = [user_sequence_dict[i][-last_num:] for i in sampled_user_asins_sid]


    sampled_ground_truth_similarity = np.zeros((len(sampled_user_asins_sid), last_num))
    sampled_experimental_similarity = np.zeros((len(sampled_user_asins_sid), last_num))


    for index_user, user in enumerate(sampled_user_asins_sid):
        for index_item, item in enumerate(sampled_item_asins_sid[index_user]):
            sampled_ground_truth_similarity[index_user][index_item] = rating_dict[user].get(item, -1)
            sampled_experimental_similarity[index_user][index_item] = norm_exp_rating_matrix[user_asins.index(user)][item_asins.index(item)] # np.dot(item_emb[item], user_emb[user].T)


    plt.clf()
    plt.figure(figsize=(10, 5))  # Adjust the total figure size to accommodate both heatmaps

    # Plot the first heatmap
    # plt.subplot(2, 1, 1)  
    sns.heatmap(sampled_ground_truth_similarity, annot=True, cmap='coolwarm', fmt=".2f" ,yticklabels=sampled_user_asins_sid)
    plt.title("GT")
    plt.savefig(f'{prefix}/a_combined_heatmaps_gt.png')
    plt.clf()
    # Plot the second heatmap
    # plt.subplot(2, 1, 2) 
    def norm_0_1_list_per_row(matrix):
        for index, i in enumerate(matrix):
            matrix[index] = (i - min(i)) / (max(i) - min(i))
        return matrix
    normalized_exp_similarity = norm_0_1_list_per_row(sampled_experimental_similarity) * 5
    
    sns.heatmap(normalized_exp_similarity, annot=True, cmap='coolwarm', fmt=".2f",yticklabels=sampled_user_asins_sid)
    plt.title("Exp")

    # Save the entire figure
    plt.tight_layout()  # Adjust layout to prevent overlap
    import pathlib
    pathlib.Path(prefix).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{prefix}/a_combined_heatmaps_exp.png')


    return



def plot_heatmap_sim(item_asins, experimental_similarity_items, ground_truth_similarity_items, prefix='./plt/', k=500, seed=0, cmap='coolwarm'):
    sampled_item_asins_sid = random.sample(item_asins, k)

    
    def normalize_list(lst):
        arr = np.array(lst)
        min_val = arr.min()
        max_val = arr.max()
        range_val = max_val - min_val
        return (arr - min_val) / range_val
    ground_truth_similarity_items = normalize_list(ground_truth_similarity_items)

    sampled_ground_truth_similarity = np.zeros((len(sampled_item_asins_sid), len(sampled_item_asins_sid)))
    sampled_experimental_similarity = np.zeros((len(sampled_item_asins_sid), len(sampled_item_asins_sid)))

    sampled_index = []
    for i_index, i in enumerate(sampled_item_asins_sid):
        for j_index, j in enumerate(sampled_item_asins_sid):
            if not i_index == j_index:
                index = find_index(item_asins, i, j)
                sampled_index.append(index)
                sampled_ground_truth_similarity[i_index][j_index] = ground_truth_similarity_items[index]
                sampled_experimental_similarity[i_index][j_index] = experimental_similarity_items[index]

    diag_value1 = np.max([ground_truth_similarity_items[index] for index in sampled_index])
    diag_value2 = np.max([experimental_similarity_items[index] for index in sampled_index])
    for i_index, i in enumerate(sampled_item_asins_sid):
            sampled_ground_truth_similarity[i_index][i_index] = diag_value1
            sampled_experimental_similarity[i_index][i_index] = diag_value2
    # sampled_experimental_similarity = normalize_data(sampled_experimental_similarity)
    # sampled_ground_truth_similarity = normalize_data(sampled_ground_truth_similarity)
    # Plotting the heatmap

    for cmap in ['viridis','coolwarm','vlag','winter','twilight_shifted','tab20c','tab20c_r','rainbow','seismic','rocket','ocean','magma','mako','icefire','gnuplot','gist_heat']:
        

        plt.figure(figsize=(10, 8))
        sns.heatmap((sampled_ground_truth_similarity - np.min(sampled_ground_truth_similarity)) / (np.max(sampled_ground_truth_similarity) - np.min(sampled_ground_truth_similarity)), cmap=cmap)#(sampled_ground_truth_similarity - np.mean(sampled_ground_truth_similarity)) / (np.max(sampled_ground_truth_similarity) - np.min(sampled_ground_truth_similarity))
        plt.title("Heatmap of Normalized Item Similarities")
        plt.savefig(f'{prefix}/heatmaps_gt_{k}_{seed}_{cmap}.png')
        plt.clf()
        plt.figure(figsize=(10, 8))
        sns.heatmap((sampled_experimental_similarity - np.min(sampled_experimental_similarity)) / (np.max(sampled_experimental_similarity) - np.min(sampled_experimental_similarity)), cmap=cmap)  # (sampled_experimental_similarity - np.mean(sampled_experimental_similarity)) / (np.max(sampled_experimental_similarity) - np.min(sampled_experimental_similarity)), annot=False, , fmt=".2f", xticklabels=sampled_item_asins_sid, yticklabels=sampled_item_asins_sid
        plt.title("Heatmap of Normalized Item Similarities")
        plt.savefig(f'{prefix}/heatmaps_exp_{k}_{seed}_{cmap}.png')
        plt.clf()

def main(args):
    get_new_tokens(args)
    print(args.dataset, args.item_indexing)
    if hasattr(args, 'new_token'):
        print(len(set(args.new_token)))

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    config = T5Config.from_pretrained(args.backbone)

    # load model, tokenizer
    if args.item_indexing in ['sequential', 'random', 'collaborative', 'independent']:
        from P5 import P5
        tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        config = T5Config.from_pretrained(args.backbone)
        model = P5.from_pretrained(args.backbone, config=config)
        model.resize_token_embeddings(len(tokenizer))
    elif 't5' in args.backbone.lower():
        config = T5Config.from_pretrained(args.backbone)
        
        if args.linear:
            from T5_Linear import CustomT5ForConditionalGeneration
            model = CustomT5ForConditionalGeneration(config)
        else:
            model = T5ForConditionalGeneration.from_pretrained(args.backbone, config=config)
        tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    
    print('Loading ', os.path.join(args.checkpoint_path, args.checkpoint_name))
   
    if args.item_indexing == 'metapath' and args.linear:
        new_token_embeddings = load_pickle(f'{args.data_path}/{args.dataset}/centroid_metapath_indexing_{args.metapath_cluster_method}_{args.metapath_cluster_num}_ag.pkl')
        extra_tokens = sorted(args.new_token)
        tokenizer.add_tokens(extra_tokens)
        model.resize_token_embeddings(len(tokenizer))
        extra_token_ids = []
        fixed_matrix = {}
        for i in set(extra_tokens):
            token_id = tokenizer.convert_tokens_to_ids(i)
            extra_token_ids.append(token_id)
            if 'CT' not in i:
                fixed_matrix[token_id] = np.random.rand(len(new_token_embeddings['<CT0>']))
            else:
                fixed_matrix[token_id] = new_token_embeddings[i]            
        model.init_for_linear(fixed_matrix, extra_token_ids, tokenizer=tokenizer)

    elif args.item_indexing == 'collaborative':
        new_tokens = sorted(args.new_token)
        tokenizer.add_tokens(sorted(new_tokens))
        model.resize_token_embeddings(len(tokenizer))
    elif args.item_indexing == 'independent':
        new_tokens = sorted(args.new_token)
        tokenizer.add_tokens(sorted(new_tokens))
        model.resize_token_embeddings(len(tokenizer))
    print('Token num:', len(tokenizer))
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.checkpoint_name)))
    # tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)


    
    DS=False
    MS=False
    PLOT=True
    if DS:

        args.user_map = {i:'user_' + j for i, j in args.user_map.items()}# 'user_' + j
        args.item_map = {i:'item_' + j for i, j in args.item_map.items()}

        emb_dict = copy.deepcopy(model.encoder.embed_tokens.weight).detach().numpy()
        user_emb = get_emb(emb_dict, tokenizer, args.user_map)
        item_emb = get_emb(emb_dict, tokenizer, args.item_map)

        item_matrix = np.array(list(item_emb.values()))    
        mean_kl_div = Diversity_Score(item_matrix, args.sample_nums)
        print('Dataset', args.dataset)
        print('Item-indexing', args.item_indexing)
        print(f'Average KL-Div for items: {mean_kl_div}')
        with open('DS-score.txt', 'a') as f:
            f.write(f'{args.dataset}\t{args.item_indexing}\t{args.sample_nums}\t{args.seed}\t{mean_kl_div}\n')

    if MS:
        args.user_map = {i:j for i, j in args.user_map.items()}# 'user_' + j
        args.item_map = {i:j for i, j in args.item_map.items()}

        emb_dict = copy.deepcopy(model.encoder.embed_tokens.weight).detach().numpy()
        user_emb = get_emb(emb_dict, tokenizer, args.user_map)
        item_emb = get_emb(emb_dict, tokenizer, args.item_map)
        a = load_pickle(f'./data/ratings/rating_mixed_test_{args.dataset}.pkl')
        rating_dict = defaultdict(dict)
        print(len(a['train']))
        for i in a['train']:
            rating_dict[i[0]][i[1]] = i[2]
        for i in a['val']:
            rating_dict[i[0]][i[1]] = i[2]
        for i in a['test']:
            rating_dict[i[0]][i[1]] = i[2]

        ground_truth_similarity_items = None
        ground_truth_similarity_users = None
        if os.path.exists(f'./data/ratings/ground_truth_similarity_{args.dataset}.pkl'):
            ground_truth_similarity_items = load_pickle(f'./data/ratings/ground_truth_similarity_{args.dataset}.pkl')
        if os.path.exists(f'./data/ratings/ground_truth_similarity_{args.dataset}_user.pkl'):
            ground_truth_similarity_users = load_pickle(f'./data/ratings/ground_truth_similarity_{args.dataset}_user.pkl')



        for _type in ['cosine']:
            _, experimental_similarity_items, average_ms_items = Memorization_Score(rating_dict, item_emb, ground_truth_similarity_items, prefix=args.dataset, exp_type=_type)
            _, experimental_similarity_users, average_ms_users = Memorization_Score(rating_dict, user_emb, ground_truth_similarity_users, prefix=args.dataset, exp_type=_type, ms_item=False)

            print('Dataset', args.dataset)
            print('Item-indexing', args.item_indexing)
            print(f'Average MS for all item pairs: {1 - average_ms_items}')
            print(f'Average MS for all user pairs: {1 - average_ms_users}')
            # with open('MS-score.txt', 'a') as f:
            #     f.write(f'{args.dataset}\t{args.item_indexing}\t{args.seed}\t{1 - ((average_ms_items + average_ms_users) / 2)}\n')

    if PLOT:
        heatmap=False
        if heatmap:
            plot_heatmap_sim(list(item_emb.keys()), experimental_similarity_items, ground_truth_similarity_items, prefix=f'./plt-heatmap/{args.item_indexing}', k=args.sample_nums, seed=args.seed, cmap=args.cmap)

        emb_dict = copy.deepcopy(model.encoder.embed_tokens.weight).detach().numpy()
        plot_color = []
        if args.item_indexing == 'random':
            plot_color = ['#DAE3F3', '#F9C756']
        elif args.item_indexing == 'sequential':
            plot_color = ['#DAE3F3', '#7F98D1']
        elif args.item_indexing == 'collaborative':
            plot_color = ['#DAE3F3', '#F4B183']
        else:
            plot_color = ['#DAE3F3', '#D17173']
        args.user_map = {i:'user_' + j for i, j in args.user_map.items()}# 'user_' + j
        args.item_map = {i:'item_' + j for i, j in args.item_map.items()}
        user_item_map = {**args.user_map, **args.item_map}
        user_item_emb = get_emb(emb_dict, tokenizer, user_item_map)

        labels = ["Other tokens", "IDs Tokens"]
        plot_pca([emb_dict, np.array(list(user_item_emb.values()))], labels=labels, file=f'{args.item_indexing}_{args.dataset}.png', plot_color=plot_color)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--backbone", type=str, default='t5-small', help='The model name')
    parser.add_argument("--data_path", type=str, default='./data', help="data directory")
    parser.add_argument("--dataset", type=str, default='Beauty', help="Dataset names, separate by comma")
    parser.add_argument("--checkpoint_path", type=str, default='../model/20240331143339_Beauty_t5-small_metapath_2023')
    parser.add_argument("--checkpoint_name", type=str, default='20240331143339_Beauty_t5-small_metapath_2023.pth')

    # Diversity Score
    parser.add_argument("--sample_nums", type=int, default=10000, help="Random sample number for Diversity Score")

    # arguments related to indexing methods
    parser.add_argument("--item_indexing", type=str, default='sequential', help="item indexing method: random, sequential, collaborative, metapath")
    parser.add_argument("--user_indexing", type=str, default='sequential', help="user indexing method: sequential, metapath")
    
    parser.add_argument("--sequential_order", type=str, default='original', help='The rank of user history during ')
    parser.add_argument("--collaborative_token_size", type=int, default=200, help='the number of tokens used for indexing')
    parser.add_argument("--collaborative_cluster", type=int, default=20, help='the number of clusters in each level for collaborative indexing.')
    parser.add_argument("--collaborative_last_token", type=str, default='sequential', help='how to assign the last token to items in a cluster, random or sequential')
    parser.add_argument("--collaborative_float32", type=int, default=0, help='1 for use float32 during indexing, 0 for float64.')

    parser.add_argument("--metapath_cluster_num", type=int, default=10)
    parser.add_argument("--metapath_cluster_method", type=str, default='kmcos')
    parser.add_argument("--linear", action='store_true')
    parser.add_argument("--cmap", type=str, default='coolwarm')

    args = parser.parse_args()
    set_seed(args.seed)
    main(args)




