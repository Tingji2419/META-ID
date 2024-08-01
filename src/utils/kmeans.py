from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist


class KmeansCosine():
    def __init__(self, user_ids, vectors, k='auto', parallel=False):
        self.user_ids = user_ids
        self.vectors = vectors
        self.k = k
        self.model = KMeans(n_clusters=self.k, random_state=0)

        self.saves = []
        self.centroids = []
        self.parallel = parallel

    def fit(self):
        length = np.sqrt((self.vectors**2).sum(axis=1))[:, None]
        self.vectors = self.vectors / length

        self.model.fit(self.vectors)

        len_ = np.sqrt(np.square(self.model.cluster_centers_).sum(axis=1)[:, None])
        centroids = self.model.cluster_centers_ / len_

        distances = cdist(self.vectors, centroids, metric='cosine')  # 计算cosine距离
        # distances = np.linalg.norm(self.vectors[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=-1)
        labels = np.argmin(distances, axis=-1)  # 计算类中心
        self.centroids = centroids

        in_center_ids = self.sort_in_centroid(labels, centroids)
        for index, user in enumerate(self.user_ids):
            # [user/item_id: 类别id, 类内id(通过与类中心聚类排序)]
            one_user_emb = [user, labels[index], in_center_ids[user]]
            self.saves.append(one_user_emb)

        return self.saves, self.centroids

    def sort_in_centroid(self, labels, centroids):
        # 计算每个item的距离类中心的距离，然后根据距离排序，自小到大编号
        in_center_ids = {}

        for i in range(self.k):
            indices = np.where(labels == i)[0]
            distances = cdist(self.vectors[indices], [centroids[i]], metric='cosine')[:, 0]
            # np.linalg.norm(self.vectors[indices] - centroids[i], axis=-1)
            sorted_indices = indices[np.argsort(distances)]
            in_center_ids.update({self.user_ids[item_index]: j for j, item_index in enumerate(sorted_indices)})

        return in_center_ids

    def get_result(self):
        return self.saves


class Kmeans():
    def __init__(self, user_ids, vectors, k='auto', parallel=False):
        self.user_ids = user_ids
        self.vectors = vectors
        if k == 'auto':
            self.k = auto_kmeans(vectors, max_k=int(len(user_ids) / 1000) + 1)
        else:
            self.k = k
        self.model = KMeans(n_clusters=self.k, random_state=0)

        self.saves = []
        self.centroids = {}
        self.parallel = parallel

    def fit(self):
        if self.parallel:
            raise NotImplementedError
        else:
            self.model.fit(self.vectors)
            centroids = self.model.cluster_centers_

        distances = np.linalg.norm(self.vectors[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=-1)
        labels = np.argmin(distances, axis=-1)  # 计算类中心
        self.centroids = centroids

        in_center_ids = self.sort_in_centroid(labels, centroids)
        for index, user in enumerate(self.user_ids):
            # [user/item_id: 类别id, 类内id(通过与类中心聚类排序)]
            one_user_emb = [user, labels[index], in_center_ids[user]]
            self.saves.append(one_user_emb)

        return self.saves, self.centroids

    def sort_in_centroid(self, labels, centroids):
        # 计算每个item的距离类中心的距离，然后根据距离排序，自小到大编号
        in_center_ids = {}

        for i in range(self.k):
            indices = np.where(labels == i)[0]
            distances = np.linalg.norm(self.vectors[indices] - centroids[i], axis=-1)
            sorted_indices = indices[np.argsort(distances)]
            in_center_ids.update({self.user_ids[item_index]: j for j, item_index in enumerate(sorted_indices)})

        return in_center_ids

    def get_result(self):
        return self.saves


def auto_kmeans(data, max_k=10):
    """
    根据肘部法则自动确定K-Means聚类的k值。

    Args:
        data (ndarray): 数据集，形状为(n_samples, n_features)。
        max_k (int): 最大的k值。默认为10。

    Returns:
        int: 自动确定的k值。
    """
    # 计算不同k值下的SSE
    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    # import matplotlib.pyplot as plt
    # 绘制SSE-k曲线
    # plt.plot(range(1, max_k + 1), sse, 'o-')
    # plt.xlabel('Number of clusters (k)')
    # plt.ylabel('Sum of squared errors (SSE)')
    # plt.title('SSE-k plot')
    # plt.savefig('SSE-k.png')

    # 根据肘部法则自动确定k值
    diff = [sse[i] - sse[i - 1] for i in range(1, len(sse))]
    elbow = diff.index(max(diff)) + 2
    print('Auto select K for KMeans: ', elbow)
    return elbow
