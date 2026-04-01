import numpy as np

def gradient_descent(X,y,lr=0.01,epochs=1000):
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(epochs):
        y_pred = X @ w
        grad = (2/n) * X.T @ (y_pred - y)
        w -= lr * grad

    return w