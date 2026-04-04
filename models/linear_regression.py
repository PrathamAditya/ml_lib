import numpy as np

"""
The model OWNS the parameters
The user PROVIDES the data
"""


class LinearRegression:

    def __init__ (self, learning_rate, epochs):

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
        m = X.shape[0]

        cost_sum = 0

        # later I need to implement vectorization here.
        for i in range(m):
            f_wb = self.w * X[i] + self.b
            cost = (f_wb - y[i]) ** 2
            cost_sum += cost

        total_cost = 1/(2*m)* cost_sum
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

        error = (self.w*X + self.b) - y
        dw = (1/m)*np.sum(error*X)
        db = 1/m*np.sum(error)

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

        for i in range(self.epochs):
            cost = self._cost_function(X, y)
            print(f"Cost: {cost} and epochs: {i + 1}")
            gradient_dw, gradient_db = self._compute_gradient(X, y)
            self.w = self.w - self.alpha*(gradient_dw)
            self.b = self.b - self.alpha*(gradient_db)
            
    def predict(self, X):
        print("Calling the predict method")

        