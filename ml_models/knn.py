import numpy as np
import pandas as pd

class KNN:
    def __init__(self, n_neighbours = 5):
        self.n_neighbours = n_neighbours

    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, X_test):
        y = []
        for x in X_test:
            distances = np.sqrt(np.sum((self.X - x)**2, axis=1))
            indices = np.argsort(distances)[:self.n_neighbours]
            labels = self.y[indices]
            value_counts = pd.Series(labels).value_counts()
            y.append(value_counts.idxmax())
        return np.array(y)
    
if __name__=='__main__':
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    knn = KNN(n_neighbours=1)
    knn.fit(X, y)
    X_test = np.array([[7, 8]])
    y_pred = knn.predict(X_test)
    print(y_pred)