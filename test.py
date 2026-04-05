from models.linear_regression import LinearRegression
import numpy as np


def UnivariateTest():
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
    print(f"Score: {x.score(X, y)}")


def test_multivariate_1():
    X = np.array([
    [1, 7],
    [2, 3],
    [3, 8],
    [4, 2],
    [5, 9],
    [6, 1],
    [7, 5],
    [8, 4],
    [9, 6],
    [10, 0]
])
    y = 2 * X[:, 0] + 3 * X[:, 1] + 5
    m = X.shape[0]
    mean_X = np.mean(X, axis=0)
    std_X = np.sqrt(np.mean((X - mean_X)**2, axis=0))
    X = (X - mean_X) / std_X
    mean_y = np.mean(y)
    std_y = np.sqrt(np.mean((y - mean_y)**2))
    y = (y - mean_y) / std_y

    # Train
    model = LinearRegression(epochs=10000, learning_rate=0.001)
    model.fit(X, y)

    # Test input
    test_input = np.array([10, 10])

    # Scale input (IMPORTANT: use same mean/std)
    test_scaled = (test_input - mean_X) / std_X

    # Predict (then inverse scale)
    pred_scaled = model.predict(test_scaled)
    pred = pred_scaled * std_y + mean_y

    print(f"Prediction for [10,10]: {pred}")

def test_multivariate_2():
    X = np.array([
    [1, 7],
    [2, 3],
    [3, 8],
    [4, 2],
    [5, 9],
    [6, 1],
    [7, 5],
    [8, 4],
    [9, 6],
    [10, 0]
])
    y = 2 * X[:, 0] + 3 * X[:, 1] + 5

    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    X_scaled = (X - mean_X) / std_X
    mean_y = np.mean(y)
    std_y = np.std(y)
    y_scaled = (y - mean_y) / std_y

    model = LinearRegression(epochs=10000, learning_rate=0.001)
    model.fit(X_scaled, y_scaled)


    test_input = np.array([10, 10])

    test_scaled = (test_input - mean_X) / std_X

    pred_scaled = model.predict(test_scaled)

    pred = pred_scaled * std_y + mean_y

    print("Prediction for [10,10]:", pred)

    test_batch = np.array([
    [10, 10],
    [5, 5]
])

    test_batch_scaled = (test_batch - mean_X) / std_X
    preds_scaled = model.predict(test_batch_scaled)
    preds = preds_scaled * std_y + mean_y

    print("Batch Predictions:", preds)
    print("Learned weights:", model.w)
    print("Bias:", model.b)

UnivariateTest()



