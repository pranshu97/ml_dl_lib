from ml_models import KMeans, KNN
import numpy as np

if __name__ == "__main__":
    # KMeans example
    X = np.random.rand(10, 2)
    kmeans = KMeans(2)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    print(labels)

    # KNN example
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    knn = KNN(n_neighbours=1)
    knn.fit(X, y)
    X_test = np.array([[7, 8]])
    y_pred = knn.predict(X_test)
    print(y_pred)