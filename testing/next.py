import numpy as np

train_ds = iter(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

for _ in range(10):
    print(next(train_ds))