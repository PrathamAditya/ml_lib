import numpy as np

class LogisticRegression:
    def __init__(self):
        pass

    def __init__(self, epochs, learning_rate):
        # w should be vector for non-univariate Linear Regression
        self.w = 0
        self.b = 0

        # need to put check to the values of LR, epochs before assigning them
        self.alpha = learning_rate
        self.epochs = epochs

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
    
    def _sigmoid(z):
        return 1/(1 + np.exp(-z))
        
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
        # number of traning data
        m = X.shape[0]
        z = np.dot(X, self.w) + self.b
        y_hat = self._sigmoid(z)
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        loss = -y*np.log(y_hat) - (1 - y)*np.log(1 - y_hat)
        sum_loss = np.sum(loss)
        total_cost = sum_loss / m
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

        z = np.dot(X, self.w) + self.b
        y_hat = self._sigmoid(z)
        error =  y_hat - y
        dw = (1/m) * np.sum(X.T, error)
        db = (1/m) * np.sum(error)

        return dw, db
    
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy
    
    def predict(self, X):
        is_single_input = 0
        if X.ndim == 1:
            is_single_input = 1
            X = X.reshape(1, -1)
        # else:
        #     X = X.reshape(1, -1)
        
        z = np.dot(X, self.w) + self.b
        y_hat = self._sigmoid(z)

        predictions = y_hat >= 0.5

        if is_single_input:
            return predictions[0]
        else: 
            return predictions

    

