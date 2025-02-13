import numpy as np

f = np.array([1, 1, 0])  # -> _t
x = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]
)


print(list(filter(lambda y: y.sum() == (f * y).sum(), x)))
