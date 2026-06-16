import numpy as np

def dropout(x,drop_prob,training):
    if not training:
        return x
    if drop_prob == 1:
        return np.zeros_like(x)
    mask = (np.random.rand(*x.shape) > drop_prob).astype(np.float32)
    return mask * x /(1.0 - drop_prob) #关键:存活的元素放大 1/keep_prob 倍 —— 这叫 inverted dropout