import numpy as np

class KMeans:
    def __init__(self, num_clusters, max_iters=10000):
        self.num_clusters = num_clusters
        self.max_iters = max_iters
    
    def fit(self, X):
        num_samples, num_features = X.shape
        centroid_indices = np.random.choice(num_samples, self.num_clusters, replace=False)
        self.centroids = X[centroid_indices]

        for i in range(self.max_iters):
            preds = np.argmin(((X[:, None]- self.centroids)**2).sum(axis=2), axis=1)
            new_cent = np.array([X[preds==cluster].mean(0) for cluster in range(self.num_clusters)])
            if (new_cent==self.centroids).all():
                break
            self.centroids = new_cent
    
    def predict(self, X):
        return np.argmin(((X[:, None]- self.centroids)**2).sum(axis=2), axis=1)

if __name__ == "__main__":
    X = np.random.rand(10, 2)
    kmeans = KMeans(2)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    print(labels)