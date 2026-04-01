import numpy as np

def sgd(X,y,lr=0.01,epochs=1000):
    bs,dim = X.shape
    w = np.zeros(dim)

    for epoch in range(epochs):
        indices = np.random.permutation(bs)
        for i in range(indices):
            x_i = X[i]
            y_i = y[i]
            y_pred = x_i @ w
            grad = 2 * x_i * (y_pred - y_i)
            w -= lr * grad

    return w