"""Input pipeline for DAE dataset."""

import jax
import jax.numpy as jnp
from jax import random
from typing import Union

import generate_data

def create_data_generator(kwargs):
    generate_dataset = generate_data.create_generate_basic_data(**kwargs)

    def data_generator(
        key: jnp.ndarray, 
        n: int, 
        batch_size: Union[int, None] = None,
        ):
        
        # Generate the full dataset in one go
        rng, key1, key2 = random.split(key, 3)
        data = generate_dataset(key1, n)
        
        if batch_size is None:
            batch_size = n  # Default batch size is all data
            num_batches = 1
        else:
            assert n % batch_size == 0, "Total number of samples must be divisible by batch size."
            num_batches = n // batch_size

        # This function shuffles and reshapes the data
        @jax.jit
        def shuffle_data(key, data):
            indices = random.permutation(key, n)  # JAX-based shuffling
            output = tuple(data_point[indices].reshape((num_batches, batch_size, *data_point.shape[1:])) for data_point in data)
            return output

        # Define a looped batch iterator that reshuffles after each pass
        def batch_iterator(rng):
            while True:
                # Shuffle the entire dataset using JAX's random module
                rng, key = random.split(rng)
                data_batched = shuffle_data(key, data)
                
                # Yield each batch one by one in sequence
                for i in range(num_batches):
                    yield tuple(data_point[i] for data_point in data_batched)

        return batch_iterator(key2)

    return data_generator

# Example usage
# data_generator = create_data_generator({
#     "params": {
#         "amplitude": (0.1, 1.0),
#         "decay_count": 2,
#         "decay_rate": (0.1, 0.5),
#         "frequency": (0.1, 0.5),
#         "phase": (0.1, 0.5),
#     },
#     "t_max": 10,
#     "t_len": 1120,
#     "SNR": 10,
#     "wavelet": "coif6",
#     "mode": "zero",
#     "max_dwt_level": 5,
#     "dtype": jnp.float32,
# })
#
# key = random.PRNGKey(0)
# key, subkey = random.split(key)
# batch_iterator = data_generator(subkey, 100, 10)
# for batch in batch_iterator:
#     print(batch[0].shape, batch[1].shape, batch[2].shape)
#
# # Output:
# # (10, 1120) (10, 68) (10, 1120)
