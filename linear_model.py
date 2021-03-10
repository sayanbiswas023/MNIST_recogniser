import numpy as np

class linearregression:

    def __init__(self, learning_rate, n_iters):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.slope = None
        self.intercept = None

    def fit(self, X, Y):
        n_samples,n_features = X.shape

        # init parameters
        self.slope = np.zeros(n_features)
        self.intercept = 0
        self.X = X
        self.Y = Y

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.slope) + self.intercept
            # compute gradients
            dm = (1 / n_samples) * np.dot(X.T, (y_predicted - self.Y))
            dc = (1 / n_samples) * np.sum(y_predicted - self.Y)

            # update parameters
            self.slope -= self.lr * dm
            self.intercept -= self.lr * dc


    def predict(self, X):
        y_approximated = np.dot(X, self.slope) + self.intercept
        return y_approximated
