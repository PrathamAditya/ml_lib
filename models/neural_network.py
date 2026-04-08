import numpy as np

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, learning_rate, epochs):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # else all will become identical
        self.W2 = np.random.randn(hidden_size, 1) * 0.01

        # bias
        self.b1 = np.zeros(hidden_size)
        self.b2 = 0


    def _forward():
        