import numpy as np
import random
from itertools import combinations
from sklearn.cluster import SpectralClustering
from utils import utils
from collections import defaultdict
import os
from scipy.sparse import csr_matrix
import pdb
import pandas as pd


def independent_indexing(data_path, dataset, user_sequence_dict):
    user_index_file = os.path.join(data_path, dataset, 'user_indexing.txt')
    item_index_file = os.path.join(data_path, dataset, 'item_independent_indexing.txt')
    reindex_sequence_file = os.path.join(data_path, dataset, f'user_sequence_independent_indexing.txt')

    if os.path.exists(reindex_sequence_file):
        user_sequence = utils.ReadLineFromFile(reindex_sequence_file)
        
        item_info = utils.ReadLineFromFile(item_index_file)
        item_map = get_dict_from_lines(item_info)
        user_info = utils.ReadLineFromFile(user_index_file)
        user_map = get_dict_from_lines(user_info)
        return construct_user_sequence_dict(user_sequence), item_map, user_map

    if os.path.exists(user_index_file):
        user_info = utils.ReadLineFromFile(user_index_file)
        user_map = get_dict_from_lines(user_info)
    else:
        user_map = generate_user_map(user_sequence_dict)
        utils.WriteDictToFile(user_index_file, user_map)

    if os.path.exists(item_index_file):
        item_info = utils.ReadLineFromFile(item_index_file)
        item_map = get_dict_from_lines(item_info)
    else:
        raise NotImplementedError
        
    reindex_user_sequence_dict = reindex(user_sequence_dict, user_map, item_map)
    utils.WriteDictToFile(reindex_sequence_file, reindex_user_sequence_dict)
    return reindex_user_sequence_dict, item_map, user_map



def sequential_indexing(data_path, dataset, user_sequence_dict, order):
    """
    Use sequential indexing method to index the given user seuqnece dict.
    """
    user_index_file = os.path.join(data_path, dataset, 'user_indexing.txt')
    item_index_file = os.path.join(data_path, dataset, f'item_sequential_indexing_{order}.txt')
    reindex_sequence_file = os.path.join(data_path, dataset, f'user_sequence_sequential_indexing_{order}.txt')
    
    if os.path.exists(reindex_sequence_file):
        user_sequence = utils.ReadLineFromFile(reindex_sequence_file)
        
        item_info = utils.ReadLineFromFile(item_index_file)
        item_map = get_dict_from_lines(item_info)
        user_info = utils.ReadLineFromFile(user_index_file)
        user_map = get_dict_from_lines(user_info)
        return construct_user_sequence_dict(user_sequence), item_map, user_map
    
    # For user index, load from txt file if already exists, otherwise generate from user sequence and save.
    if os.path.exists(user_index_file):
        user_info = utils.ReadLineFromFile(user_index_file)
        user_map = get_dict_from_lines(user_info)
    else:
        user_map = generate_user_map(user_sequence_dict)
        utils.WriteDictToFile(user_index_file, user_map)
        
        
    # For item index, load from txt file if already exists, otherwise generate from user sequence and save.
    if os.path.exists(item_index_file):
        item_info = utils.ReadLineFromFile(item_index_file)
        item_map = get_dict_from_lines(item_info)
    else:
        item_map = dict()
        if order == 'original':
            user_list = user_sequence_dict.keys()
        elif order == 'short2long':
            user_list = sorted(user_sequence_dict, key=lambda x: len(user_sequence_dict[x]), reverse=False)
        elif order == 'long2short':
            user_list = sorted(user_sequence_dict, key=lambda x: len(user_sequence_dict[x]), reverse=True)
            
        for user in user_list:
            items = user_sequence_dict[user][:-2]
            for item in items:
                if item not in item_map:
                    item_map[item] = str(len(item_map) + 1001)
        for user in user_list:
            items = user_sequence_dict[user][-2:]
            for item in items:
                if item not in item_map:
                    item_map[item] = str(len(item_map) + 1001)
        utils.WriteDictToFile(item_index_file, item_map)
        
    reindex_user_sequence_dict = reindex(user_sequence_dict, user_map, item_map)
    utils.WriteDictToFile(reindex_sequence_file, reindex_user_sequence_dict)
    return reindex_user_sequence_dict, item_map, user_map
        


def random_indexing(data_path, dataset, user_sequence_dict):
    """
    Use random indexing method to index the given user seuqnece dict.
    """
    user_index_file = os.path.join(data_path, dataset, 'user_indexing.txt')
    item_index_file = os.path.join(data_path, dataset, 'item_random_indexing.txt')
    reindex_sequence_file = os.path.join(data_path, dataset, f'user_sequence_random_indexing.txt')
    
    if os.path.exists(reindex_sequence_file):
        user_sequence = utils.ReadLineFromFile(reindex_sequence_file)
        
        item_info = utils.ReadLineFromFile(item_index_file)
        item_map = get_dict_from_lines(item_info)
        user_info = utils.ReadLineFromFile(user_index_file)
        user_map = get_dict_from_lines(user_info)
        return construct_user_sequence_dict(user_sequence), item_map, user_map
    
    # For user index, load from txt file if already exists, otherwise generate from user sequence and save.
    if os.path.exists(user_index_file):
        user_info = utils.ReadLineFromFile(user_index_file)
        user_map = get_dict_from_lines(user_info)
    else:
        user_map = generate_user_map(user_sequence_dict)
        utils.WriteDictToFile(user_index_file, user_map)
        
        
    # For item index, load from txt file if already exists, otherwise generate from user sequence and save.
    if os.path.exists(item_index_file):
        item_info = utils.ReadLineFromFile(item_index_file)
        item_map = get_dict_from_lines(item_info)
    else:
        item_map = dict()
        items = set()
        for user in user_sequence_dict:
            items.update(user_sequence_dict[user])
        items = list(items)
        random.shuffle(items)
        for item in items:
            if item not in item_map:
                item_map[item] = str(len(item_map) + 1001)
        utils.WriteDictToFile(item_index_file, item_map)
        
    reindex_user_sequence_dict = reindex(user_sequence_dict, user_map, item_map)
    utils.WriteDictToFile(reindex_sequence_file, reindex_user_sequence_dict)
    return reindex_user_sequence_dict, item_map, user_map

def collaborative_indexing(data_path, dataset, user_sequence_dict, token_size, cluster_num, last_token, float32):
    """
    Use collaborative indexing method to index the given user seuqnece dict.
    """
    user_index_file = os.path.join(data_path, dataset, 'user_indexing.txt')
    item_index_file = os.path.join(data_path, dataset, f'item_collaborative_indexing_{token_size}_{cluster_num}_{last_token}.txt')
    reindex_sequence_file = os.path.join(data_path, dataset, f'user_sequence_collaborative_indexing_{token_size}_{cluster_num}_{last_token}.txt')
    
    if os.path.exists(reindex_sequence_file):
        user_sequence = utils.ReadLineFromFile(reindex_sequence_file)
        
        item_info = utils.ReadLineFromFile(item_index_file)
        item_map = get_dict_from_lines(item_info)
        user_info = utils.ReadLineFromFile(user_index_file)
        user_map = get_dict_from_lines(user_info)
        return construct_user_sequence_dict(user_sequence), item_map, user_map
    
    # For user index, load from txt file if already exists, otherwise generate from user sequence and save.
    if os.path.exists(user_index_file):
        user_info = utils.ReadLineFromFile(user_index_file)
        user_map = get_dict_from_lines(user_info)
    else:
        user_map = generate_user_map(user_sequence_dict)
        utils.WriteDictToFile(user_index_file, user_map)
        
        
    # For item index, load from txt file if already exists, otherwise generate from user sequence and save.
    if os.path.exists(item_index_file):
        item_info = utils.ReadLineFromFile(item_index_file)
        item_map = get_dict_from_lines(item_info)
    else:
        item_map = generate_collaborative_id(user_sequence_dict, token_size, cluster_num, last_token, float32)
        utils.WriteDictToFile(item_index_file, item_map)
        
    reindex_user_sequence_dict = reindex(user_sequence_dict, user_map, item_map)
    utils.WriteDictToFile(reindex_sequence_file, reindex_user_sequence_dict)
    return reindex_user_sequence_dict, item_map, user_map
        
def generate_collaborative_id(user_sequence_dict, token_size, cluster_num, last_token, float32):
    """
    Generate collaborative index for items.
    """
    # get the items in training data and all data.
    all_items = set()
    train_items = set()
    for user in user_sequence_dict:
        all_items.update(set(user_sequence_dict[user]))
        train_items.update(set(user_sequence_dict[user][:-2]))
        
    # reindex all training items for calculating the adjacency matrix
    item2id = dict()
    id2item = dict()
    for item in train_items:
        item2id[item] = len(item2id)
        id2item[len(id2item)] = item
        
    
    # calculate the co-occurrence of items in the training data as an adjacency matrix
    if float32 > 0:
        adj_matrix = np.zeros((len(item2id), len(item2id)), dtype=np.float32)
    else:
        adj_matrix = np.zeros((len(item2id), len(item2id)))
    for user in user_sequence_dict:
        interactions = user_sequence_dict[user][:-2]
        for pairs in combinations(interactions, 2):
            adj_matrix[item2id[pairs[0]]][item2id[pairs[1]]] += 1
            adj_matrix[item2id[pairs[1]]][item2id[pairs[0]]] += 1
    
    
    # get the clustering results for the first layer
    clustering = SpectralClustering(
        n_clusters=cluster_num,
        assign_labels="cluster_qr",
        random_state=0,
        affinity="precomputed",
    ).fit(adj_matrix)
    labels = clustering.labels_.tolist()
    
    # count the clustering results
    grouping = defaultdict(list)
    for i in range(len(labels)):
        grouping[labels[i]].append((id2item[i],i))
    
    item_map = dict()
    index_now = 0
    
    # add current clustering information into the item indexing results.
    item_map, index_now = add_token_to_indexing(item_map, grouping, index_now, token_size)
    
    # add current clustering info into a queue for BFS
    queue = []
    for group in grouping:
        queue.append(grouping[group])
    
    # apply BFS to further use spectral clustering for large groups (> token_size)
    while queue:
        group_items = queue.pop(0)
        
        # if current group is small enough, add the last token to item indexing
        if len(group_items) <= token_size:
            item_list = [items[0] for items in group_items]
            if last_token == 'sequential':
                item_map = add_last_token_to_indexing_sequential(item_map, item_list, token_size)
            elif last_token == 'random':
                item_map = add_last_token_to_indexing_random(item_map, item_list, token_size)
        else:
            # calculate the adjacency matrix for current group
            if float32 > 0:
                sub_adj_matrix = np.zeros((len(group_items), len(group_items)), dtype=np.float32)
            else:
                sub_adj_matrix = np.zeros((len(group_items), len(group_items)))
            for i in range(len(group_items)):
                for j in range(i+1, len(group_items)):
                    sub_adj_matrix[i][j] = adj_matrix[group_items[i][1]][group_items[j][1]]
                    sub_adj_matrix[j][i] = adj_matrix[group_items[j][1]][group_items[i][1]]
                    
            # get the clustering results for current group        
            clustering = SpectralClustering(
                n_clusters=cluster_num,
                assign_labels="cluster_qr",
                random_state=0,
                affinity="precomputed",
            ).fit(sub_adj_matrix)
            labels = clustering.labels_.tolist()
            
            # count current clustering results
            grouping = defaultdict(list)
            for i in range(len(labels)):
                grouping[labels[i]].append(group_items[i])
                
            # add current clustering information into the item indexing results.
            item_map, index_now = add_token_to_indexing(item_map, grouping, index_now, token_size)
            
            # push current clustering info into the queue
            for group in grouping:
                queue.append(grouping[group])
                
    # if some items are not in the training data, assign an index for them
    remaining_items = list(all_items - train_items)
    if len(remaining_items) > 0:
        if last_token == 'sequential':
            item_map = add_last_token_to_indexing_sequential(item_map, remaining_items, token_size)
        elif last_token == 'random':
            item_map = add_last_token_to_indexing_random(item_map, remaining_items, token_size)
                
    return item_map


def rqvae_indexing(data_path, dataset, user_sequence_dict, item_indexing, user_indexing):
    """
    Use collaborative indexing method to index the given user seuqnece dict.
    """
    user_index_file = os.path.join(data_path, dataset, 'user_indexing.txt')
    item_index_file = os.path.join(data_path, dataset, f'{dataset}.txt')
    reindex_sequence_file = os.path.join(data_path, dataset, 'user_sequence_indexing.txt')
    
    if os.path.exists(reindex_sequence_file):
        user_sequence = utils.ReadLineFromFile(reindex_sequence_file)
        
        item_info = utils.ReadLineFromFile(item_index_file)
        item_map = get_dict_from_lines(item_info)
        
        return construct_user_sequence_dict(user_sequence), item_map
    
    # For user index, load from txt file if already exists, otherwise generate from user sequence and save.
    if os.path.exists(user_index_file):
        user_info = utils.ReadLineFromFile(user_index_file)
        user_map = get_dict_from_lines(user_info)
    else:
        user_map = generate_user_map(user_sequence_dict)
        utils.WriteDictToFile(user_index_file, user_map)
        
        
    # For item index, load from txt file if already exists, otherwise generate from user sequence and save.
    if os.path.exists(item_index_file):
        item_info = utils.ReadLineFromFile(item_index_file)
        item_map = get_dict_from_lines(item_info)
    else:
        raise NotImplementedError


    all_items = set()
    for user in user_sequence_dict:
        all_items.update(set(user_sequence_dict[user]))

    prefix = '<a_256><b_0><c_0>'
    tmp_index = 0
    for item in all_items:
        if item not in item_map.keys():
            item_map[item] = f'{prefix}<d_{tmp_index}>'
            tmp_index += 1
            print(item, item_map[item])


    reindex_user_sequence_dict = reindex(user_sequence_dict, user_map, item_map)
    utils.WriteDictToFile(reindex_sequence_file, reindex_user_sequence_dict)
    return reindex_user_sequence_dict, item_map


def metapath_indexing(data_path, dataset, user_sequence_dict, metapath_cluster_method, metapath_cluster_num, item_indexing, user_indexing):
    """
    Use metapath indexing method to index the given user seuqnece dict.
    """
    if user_indexing == 'metapath':
        user_index_file = os.path.join(data_path, dataset, f'user_{user_indexing}_indexing_{metapath_cluster_method}_{metapath_cluster_num}_ag.txt')
    else:
        user_index_file = os.path.join(data_path, dataset, 'user_indexing.txt')

    if item_indexing == 'metapath':
        item_index_file = os.path.join(data_path, dataset, f'item_{item_indexing}_indexing_{metapath_cluster_method}_{metapath_cluster_num}_ag.txt')
    else:
        item_index_file = os.path.join(data_path, dataset, 'item_indexing.txt')

    reindex_sequence_file = os.path.join(data_path, dataset, f'user_sequence_user_{user_indexing}_item_{item_indexing}_indexing_{metapath_cluster_method}_{metapath_cluster_num}_ag.txt')
    print(reindex_sequence_file)
    

    if os.path.exists(reindex_sequence_file):
        user_sequence = utils.ReadLineFromFile(reindex_sequence_file)

        item_info = utils.ReadLineFromFile(item_index_file)
        item_map = get_dict_from_lines(item_info)

        user_info = utils.ReadLineFromFile(user_index_file)
        user_map = get_dict_from_lines(user_info)

        return construct_user_sequence_dict(user_sequence), item_map, user_map

    # For user index, load from txt file if already exists, otherwise generate from user sequence and save.
    if os.path.exists(user_index_file):
        user_info = utils.ReadLineFromFile(user_index_file)
        user_map = get_dict_from_lines(user_info)
    else:
        if user_indexing == 'metapath':
            user_map = generate_metapath_id(data_path, dataset, metapath_cluster_method, metapath_cluster_num, is_item=False)
        else:
            user_map = generate_user_map(user_sequence_dict)
        utils.WriteDictToFile(user_index_file, user_map)

    # For item index, load from txt file if already exists, otherwise generate from user sequence and save.
    if os.path.exists(item_index_file):
        item_info = utils.ReadLineFromFile(item_index_file)
        item_map = get_dict_from_lines(item_info)
    else:
        if item_indexing == 'metapath':
            item_map = generate_metapath_id(data_path, dataset, metapath_cluster_method, metapath_cluster_num)
        else:
            item_map = dict()
            user_list = user_sequence_dict.keys()
            for user in user_list:
                items = user_sequence_dict[user][:-2]
                for item in items:
                    if item not in item_map:
                        item_map[item] = str(len(item_map) + 1001)
            for user in user_list:
                items = user_sequence_dict[user][-2:]
                for item in items:
                    if item not in item_map:
                        item_map[item] = str(len(item_map) + 1001)
        utils.WriteDictToFile(item_index_file, item_map)

    # add skip-user for zero-shot item
    skip_user = []
    for user in user_sequence_dict.keys():
        test_item = user_sequence_dict[user][-1]
        if test_item not in item_map.keys():
            skip_user.append(user)
    print('Skip user: ', skip_user)

    reindex_user_sequence_dict = reindex(user_sequence_dict, user_map, item_map, skip_user=skip_user)
    utils.WriteDictToFile(reindex_sequence_file, reindex_user_sequence_dict)

    return reindex_user_sequence_dict, item_map, user_map


def generate_metapath_id(data_path, dataset, metapath_cluster_method, metapath_cluster_num, is_item=True):
    # 加载向量数据
    vectors_dict = pd.read_csv(f'{data_path}/{dataset}/metapath2vec_user_item.csv')  # {user/item_id: emb}

    item_ids = [i for i in vectors_dict['item'].values if i != 'is_user']
    user_ids = [i for i in vectors_dict['user'].values if i != 'is_item']
    item_vectors = np.array([list(map(float, line.split())) for line, is_item in zip(vectors_dict['emb'].values, vectors_dict['item'].values) if is_item != 'is_user'])  # array(N, dim)
    user_vectors = np.array([list(map(float, line.split())) for line, is_item in zip(vectors_dict['emb'].values, vectors_dict['item'].values) if is_item == 'is_user'])

    if is_item:
        ids, vectors = item_ids, item_vectors
    else:
        ids, vectors = user_ids, user_vectors

    K = metapath_cluster_num
    if metapath_cluster_method == 'km':
        from utils.kmeans import Kmeans
        model = Kmeans(ids, vectors, k=K)
        in_center_info, centroids = model.fit()

    elif metapath_cluster_method == 'kmcos':
        from utils.kmeans import KmeansCosine
        model = KmeansCosine(ids, vectors, k=K)
        in_center_info, centroids = model.fit()

    elif metapath_cluster_method == 'lsh':
        raise NotImplementedError

    # 利用metapath的[center_id, in_center_id]计算映射表
    center_start = 0
    in_center_start = 1001
    asin2id = {}
    centroids_dict = {}
    for d in in_center_info:
        asin = d[0]
        center_label = center_start + d[1]
        in_center_id = in_center_start + d[2]
        asin2id[asin] = f"<CT{center_label}><{in_center_id}>"
        # Save {<CT{center_label}>: emb}
        if center_label not in centroids_dict.keys():
            centroids_dict[f"<CT{center_label}>"] = centroids[d[1]]

    utils.save_pickle(centroids_dict, f'{data_path}/{dataset}/centroid_metapath_indexing_{metapath_cluster_method}_{metapath_cluster_num}.pkl')

    return asin2id

    
def add_token_to_indexing(item_map, grouping, index_now, token_size):
    for group in grouping:
        index_now = index_now % token_size
        for (item, idx) in grouping[group]:
            if item not in item_map:
                item_map[item] = ''
            item_map[item] += f'<CI{index_now}>'
        index_now += 1
    return item_map, index_now

def add_last_token_to_indexing_random(item_map, item_list, token_size):
    last_tokens = random.sample([i for i in range(token_size)], len(item_list))
    for i in range(len(item_list)):
        item = item_list[i]
        if item not in item_map:
            item_map[item] = ''
        item_map[item] += f'<CI{last_tokens[i]}>'
    return item_map

def add_last_token_to_indexing_sequential(item_map, item_list, token_size):
    for i in range(len(item_list)):
        item = item_list[i]
        if item not in item_map:
            item_map[item] = ''
        item_map[item] += f'<CI{i}>'
    return item_map
    
    
def get_dict_from_lines(lines):
    """
    Used to get user or item map from lines loaded from txt file.
    """
    index_map = dict()
    for line in lines:
        info = line.split(" ")
        index_map[info[0]] = info[1]
    return index_map
        
        
        
        
def generate_user_map(user_sequence_dict):
    """
    generate user map based on user sequence dict.
    """
    user_map = dict()
    for user in user_sequence_dict.keys():
        user_map[user] = str(len(user_map) + 1)
    return user_map


def reindex(user_sequence_dict, user_map, item_map):
    """
    reindex the given user sequence dict by given user map and item map
    """
    reindex_user_sequence_dict = dict()
    for user in user_sequence_dict:
        uid = user_map[user]
        items = user_sequence_dict[user]
        reindex_user_sequence_dict[uid] = [item_map[i] for i in items]
        
    return reindex_user_sequence_dict
    
    
def construct_user_sequence_dict(user_sequence):
    """
    Convert a list of string to a user sequence dict. user as key, item list as value.
    """

    user_seq_dict = dict()
    for line in user_sequence:
        user_seq = line.split(" ")
        user_seq_dict[user_seq[0]] = user_seq[1:]
    return user_seq_dict