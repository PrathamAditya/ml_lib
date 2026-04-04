from models.linear_regression import LinearRegression
import numpy as np

def Test():
    
    X = np.array([1,2,3,4,5,6,7,8,9,10])
    m = X.shape[0]
    mean = np.sum(X) / m
    std = np.sqrt( np.sum((X - mean)**2) / m )
    X = (X - mean) / std
    # True relation: y = 3x + 5 + noise
    y = np.array([8.2, 11.1, 14.0, 17.2, 20.1, 23.0, 26.2, 29.1, 32.0, 35.2])
    mean_y = np.sum(y) / m
    std_y = np.sqrt( np.sum((y - mean)**2) / m )
    y = (y - mean_y)/std_y
    x = LinearRegression(10000, 0.001)
    x.fit(X, y)

    scaled_test_input = (10-mean)/std
    #print(mean, std , scaled_test_input)
    print(f"Prediction for x: {x.predict(scaled_test_input)*std_y + mean_y}")

Test()


