from sklearn.cluster import SpectralClustering
import numpy as np

class SpectralClusteringWrapper():
    def __init__(self, user_ids, vectors, n_clusters='auto', parallel=False):
        self.user_ids = user_ids
        self.vectors = vectors
        if n_clusters == 'auto':
            # Estimating the number of clusters ('auto' strategy can be defined based on your dataset)
            self.n_clusters = min(10, len(user_ids) // 2)  # Example heuristic
        else:
            self.n_clusters = n_clusters
        self.model = SpectralClustering(n_clusters=self.n_clusters, affinity='nearest_neighbors', random_state=0)
        self.parallel = parallel

        self.saves = []
        self.labels = []

    def fit(self):
        if self.parallel:
            raise NotImplementedError
        else:
            self.labels = self.model.fit_predict(self.vectors)

        unique_labels = set(self.labels)
        self.centroids = {label: np.mean(self.vectors[self.labels == label], axis=0) for label in unique_labels}

        in_center_ids = self.sort_in_centroid(self.labels, self.centroids)
        for index, user in enumerate(self.user_ids):
            label = self.labels[index]
            one_user_emb = [user, label, in_center_ids.get(user, -1)]
            self.saves.append(one_user_emb)

        return self.saves, self.centroids

    def sort_in_centroid(self, labels, centroids):
        in_center_ids = {}

        for label, centroid in centroids.items():
            indices = np.where(labels == label)[0]
            distances = np.linalg.norm(self.vectors[indices] - centroid, axis=-1)
            sorted_indices = indices[np.argsort(distances)]
            in_center_ids.update({self.user_ids[item_index]: j for j, item_index in enumerate(sorted_indices)})

        return in_center_ids

    def get_result(self):
        return self.saves
