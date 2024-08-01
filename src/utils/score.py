import random
import pickle
from tqdm import tqdm
from scipy.special import softmax
import numpy as np

def save_pickle(save_path, data):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

def kl_divergence(P, Q):
    """
    Calculate the KL-divergence between two probability distributions P and Q.
    """
    epsilon = 1e-10  # small value to avoid log(0)
    P = np.where(P == 0, epsilon, P)  # Replace any 0 in P with epsilon
    Q = np.where(Q == 0, epsilon, Q)  # Replace any 0 in Q with epsilon
    return np.sum(P * np.log(P / Q), axis=1)

def Diversity_Score(emb_matrix, sample_num=10000):
    """
    emb_matrix: array(N, dim)
    """
    prob_matrix = softmax(emb_matrix, axis=1)

    num_embeddings = prob_matrix.shape[0]
    kl_divergence_values = []
    for _ in tqdm(range(sample_num)):
        i, j = random.sample(range(num_embeddings), 2)
        kl_div_i_j = kl_divergence(prob_matrix[i:i+1], prob_matrix[j:j+1])
        kl_div_j_i = kl_divergence(prob_matrix[j:j+1], prob_matrix[i:i+1])
        kl_divergence_values.append((kl_div_i_j + kl_div_j_i) / 2)

    mean_kl_div = np.mean(kl_divergence_values)
    return mean_kl_div



def calculate_similarity_components(data, item_asins, user_averages):
    # Build a dictionary to hold pre-computed components for each item
    item_components = {item: {'sum': 0, 'squared_sum': 0} for item in item_asins}

    for user, items in data.items():
        user_avg = user_averages[user]
        for item, rating in items.items():
            if item in item_components:
                deviation = rating - user_avg
                item_components[item]['sum'] += deviation
                item_components[item]['squared_sum'] += deviation ** 2

    return item_components

def calculate_user_similarity_components(data, user_asins, user_averages):
    # Build a dictionary to hold pre-computed components for each user
    user_components = {user: {'sum': 0, 'squared_sum': 0} for user in user_asins}


    for user, items in data.items():
        user_avg = user_averages[user]
        for item, rating in items.items():
            deviation = rating - user_avg
            user_components[user]['sum'] += deviation
            user_components[user]['squared_sum'] += deviation ** 2
    
    return user_components


def adjusted_cosine_similarity_precomputed(components, item1, item2):
    num = components[item1]['sum'] * components[item2]['sum']
    denom = np.sqrt(components[item1]['squared_sum']) * np.sqrt(components[item2]['squared_sum'])
    
    return num / denom if denom != 0 else 0

def euclidean_distance(ground_truth, experimental):
    # Convert lists to numpy arrays if not already
    ground_truth = np.array(ground_truth)
    experimental = np.array(experimental)
    # ground_truth = (ground_truth - np.mean(ground_truth)) / np.std(ground_truth)
    # experimental = (experimental - np.mean(experimental)) / np.std(experimental)
    # Calculate the Euclidean distance
    distance = np.linalg.norm(ground_truth - experimental) / len(ground_truth)
    return distance

def pearson_correlation_coefficient(ground_truth, experimental):
    ground_truth = np.array(ground_truth)
    experimental = np.array(experimental)

    mean_ground_truth = np.mean(ground_truth)
    mean_experimental = np.mean(experimental)
    
    cov = np.mean((ground_truth - mean_ground_truth) * (experimental - mean_experimental))
    std_ground_truth = np.std(ground_truth)
    std_experimental = np.std(experimental)
    
    correlation_coefficient = cov / (std_ground_truth * std_experimental)
    return correlation_coefficient

def euclidean_distance_matrix(embeddings):
    # Compute the Euclidean distance matrix
    from scipy.spatial import distance
    dist_matrix = distance.squareform(distance.pdist(embeddings, 'euclidean'))
    similarity_matrix = np.exp(-dist_matrix)
    return similarity_matrix

def cosine_matrix(embeddings_matrix):

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings_matrix)
    pca = PCA(n_components=0.95)
    embeddings_matrix = pca.fit_transform(embeddings_normalized)
    # Calculate the cosine similarity matrix
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    normalized_embeddings = embeddings_matrix / norms
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    return similarity_matrix

def pearson_correlation_matrix(data):
    from scipy.stats import pearsonr
    n = data.shape[0]
    corr_matrix = np.empty((n, n))
    for i in range(n):
        for j in range(i+1, n):
            corr, _ = pearsonr(data[i], data[j])
            corr_matrix[i, j] = corr_matrix[j, i] = corr
    np.fill_diagonal(corr_matrix, 1)
    return corr_matrix

def Memorization_Score(ratings_dict, embeddings_dict, ground_truth_similarity=[], prefix='', exp_type='cosine', ms_item=True):
    item_asins = list(embeddings_dict.keys())
    num_items = len(item_asins)
    user_averages = {user: np.mean(list(items.values())) for user, items in ratings_dict.items()}

    # Cal exper sim
    embeddings_matrix = np.array([embeddings_dict[item] for item in item_asins])
    if exp_type == 'cosine':
        similarity_matrix = cosine_matrix(embeddings_matrix)
    elif exp_type == 'euclidean':
        similarity_matrix = euclidean_distance_matrix(embeddings_matrix)
    # elif exp_type == 'pearson':
    #     similarity_matrix = pearson_correlation_matrix(embeddings_matrix)

    upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
    experimental_similarity = similarity_matrix[upper_tri_indices]
    
    if len(ground_truth_similarity) == 0:
        # Precompute the similarity components for each item
        if ms_item:
            item_components = calculate_similarity_components(ratings_dict, item_asins, user_averages)
        else:
            item_components = calculate_user_similarity_components(ratings_dict, item_asins, user_averages)
            
        # Initialize the ground truth similarity array
        ground_truth_similarity = np.zeros((num_items * (num_items - 1)) // 2)
        k = 0
        # Iterate over each unique pair of items to compute their similarity
        for i in tqdm(range(num_items - 1)):
            for j in range(i + 1, num_items):
                item1, item2 = item_asins[i], item_asins[j]
                ground_truth_similarity[k] = adjusted_cosine_similarity_precomputed(item_components, item1, item2)
                k += 1

        save_pickle(f'./data/ratings/ground_truth_similarity_{prefix}.pkl', ground_truth_similarity)

    def normalize_list(lst):
        arr = np.array(lst)
        min_val = arr.min()
        max_val = arr.max()
        range_val = max_val - min_val
        return (arr - min_val) / range_val
    # ms = pearson_correlation_coefficient(ground_truth_similarity, experimental_similarity)
    from sklearn.metrics import mean_squared_error
    # ms = euclidean_distance(ground_truth_similarity, experimental_similarity)
    ms = mean_squared_error(normalize_list(ground_truth_similarity), experimental_similarity)
    print('Length of pair:', len(ground_truth_similarity))

    return ground_truth_similarity, experimental_similarity, ms