import numpy as np

"""
The model OWNS the parameters
The user PROVIDES the data
"""


class LinearRegression:

    def __init__ ():
        pass

    def __init__ (self, epochs, learning_rate):

        # w should be vector for non-univariate Linear Regression
        self.w = 0
        self.b = 0

        # need to put check to the values of LR, epochs before assigning them
        self.alpha = learning_rate
        self.epochs = epochs
    
    def _cost_function(self, X, y):
        """
        Computes the cost function for linear regression.

        Args:
            X (ndarray(m,)): data, m examples
            y (ndarray(m,)): target values

        Returns
            total_cost (float): The cost of using w, b for linear regression 
            to fit the data points in X and y.

        """
        # length of input vector
        m = X.shape[0]
        y_hat = np.dot(X, self.w) + self.b
        error = y_hat - y
        sum_of_square_error = np.sum(error ** 2)
        total_cost = sum_of_square_error/(2*m)
        return total_cost
    
    def _compute_gradient(self, X, y):
        """
        Computes the gradient for linear regression.

        Args:
            X (ndarray(m,)): data, m examples
            y (ndarray(m,)): target values

        Returns
            dw (float): The gradient wrt to w. 
            db: The gradient wrt to b.
        """
        # size of data
        m = X.shape[0]

        y_hat = np.dot(X, self.w) + self.b
        error =  y_hat - y
        dw = (1/m) * np.sum(X.T * error)
        db = (1/m) * np.sum(error)

        return dw, db

    def fit(self, X, y):
        """
        Fitting data X wrt to given y label outputs.

        Args:
            X (ndarray(m,)): data, m examples
            y (ndarray(m,)): target values

        Returns
            None
        """

        # reshaping
        if (X.ndim == 1):
            X = X.reshape(-1, 1)

        # initialization of weights
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for i in range(self.epochs):
            cost = self._cost_function(X, y)
            print(f"Cost: {cost} and epochs: {i + 1}")
            gradient_dw, gradient_db = self._compute_gradient(X, y)
            if(i == 0):
                print(gradient_dw, " ", gradient_db)
            self.w = self.w - self.alpha*(gradient_dw)
            self.b = self.b - self.alpha*(gradient_db)

        print(f"W: {self.w}, b: {self.b}")


    def score(self, X, y):
        y_hat = self.predict(X)
        mean_y = np.mean(y)
        SS_res = np.sum((y - y_hat) ** 2)
        SS_tot = np.sum((y - mean_y) ** 2)
        return (1 - (SS_res/SS_tot))

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.dot(X, self.w) + self.b

        