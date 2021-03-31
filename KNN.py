import numpy as np
from collections import Counter

class KNN():
    
    def __init__(self):##no init required
        pass

    def train(self, X, y):##KNN is just a lazy learner
        self.X_train = X
        self.y_train = y

    def predict(self, X, k):
        dists = self.compute_distances(X)
        # print("computed distances")

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            k_closest_y = []
            labels = self.y_train[np.argsort(dists[i,:])].flatten()
            # find k nearest lables
            k_closest_y = labels[:k]##slices the first K labels only

            
            c = Counter(k_closest_y)##majority vote
            y_pred[i] = c.most_common(1)[0][0] ## takes out the label of the majority vote

        return(y_pred)

    def compute_distances(self, X): ## just a cool trick to avoid loop to find euclidean distances
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        dot_pro = np.dot(X, self.X_train.T)
        sum_square_test = np.square(X).sum(axis = 1)
        sum_square_train = np.square(self.X_train).sum(axis = 1)
        dists = np.sqrt(-2 * dot_pro + sum_square_train + np.matrix(sum_square_test).T)

        return(dists)
