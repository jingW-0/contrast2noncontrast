import torch

class KMeans:
    def __init__(self, n_clusters, max_iters=100, tolerance=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.centroids = None

    def fit(self, data):
        # Randomly initialize centroids
        indices = torch.randperm(data.size(0))[:self.n_clusters]
        self.centroids = data[indices].clone()

        for _ in range(self.max_iters):
            # Assign clusters
            distances = torch.cdist(data, self.centroids)
            labels = distances.argmin(dim=1)

            # Update centroids
            new_centroids = torch.stack([data[labels == k].mean(dim=0) for k in range(self.n_clusters)])

            # Check for convergence
            centroid_shift = torch.norm(self.centroids - new_centroids, dim=1).sum()
            if centroid_shift <= self.tolerance:
                break

            self.centroids = new_centroids

    def predict(self, data):
        distances = torch.cdist(data, self.centroids)
        labels = distances.argmin(dim=1)
        return labels

class KMeans_cos:
    def __init__(self, n_clusters, max_iters=100, tolerance=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.centroids = None

    def fit(self, data):
        # Normalize data to have unit norm
        # print(f"fit data {data.shape}, min {data.min()}, max {data.max()}")
        data_norm = data / data.norm(dim=1, keepdim=True)
        # print(f"min {data_norm.min()}, max {data_norm.max()}")


        # Randomly initialize centroids
        indices = torch.randperm(data.size(0))[:self.n_clusters]
        self.centroids = data_norm[indices].clone()

        for _ in range(self.max_iters):
            # Compute cosine similarity (as distance)
            similarities = torch.mm(data_norm, self.centroids.t())
            labels = similarities.argmax(dim=1)

            # Update centroids to maximize cosine similarity
            new_centroids = torch.stack([
                data_norm[labels == k].mean(dim=0) for k in range(self.n_clusters)
            ])
            new_centroids = new_centroids / new_centroids.norm(dim=1, keepdim=True)  # Normalize centroids

            # Check for convergence (using similarity changes)
            centroid_shift = torch.norm(self.centroids - new_centroids, dim=1).sum()
            if centroid_shift <= self.tolerance:
                break

            self.centroids = new_centroids


    def predict(self, data):
        data_norm = data / data.norm(dim=1, keepdim=True)
        similarities = torch.mm(data_norm, self.centroids.t())
        labels = similarities.argmax(dim=1)
        return labels