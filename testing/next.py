import numpy as np
import random

# Demonstrating the use of the next() function
data = iter(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

for _ in range(10):
    print(next(data))

# Demonstrating the use of the StopIteration exception
class RandomIterable:
    def __iter__(self):
        return self
    def __next__(self):
        if random.choice(["go", "go", "stop"]) == "stop":
            raise StopIteration  # signals "the end"
        return 1

for i in range(5):
    print(list(RandomIterable()))