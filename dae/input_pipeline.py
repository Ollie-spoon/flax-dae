"""Input pipeline for DAE dataset."""

import jax
import jax.numpy as jnp
from jax import random
from typing import Union

import generate_data

def create_data_generator(kwargs):
    _generate_data = generate_data.create_generate_basic_data(**kwargs)

    def data_generator(key: jnp.ndarray, n: int, batch_size: Union[int, None] = None):
        # Generate the full dataset in one go
        rng, key = random.split(key)
        data = _generate_data(key, n)
        
        clean_signal = data[0]  # Noiseless version in the time domain
        noisy_approx = data[1]  # Noisy version of the approximation coefficient
        noisy_signal = data[2]  # Noisy version in the time domain
        
        if batch_size is None:
            batch_size = n  # Default batch size is all data
            num_batches = 1
        else:
            assert n % batch_size == 0, "Total number of samples must be divisible by batch size."
            num_batches = n // batch_size

        # This function shuffles and reshapes the data
        @jax.jit
        def shuffle_data(key, clean_signal, noisy_approx):
            indices = random.permutation(key, n)  # JAX-based shuffling
            shuffled_clean_signal = clean_signal[indices]
            shuffled_noisy_approx = noisy_approx[indices]
            shuffled_noisy_signal = noisy_signal[indices]

            # Reshape into batches
            clean_signal_batched = shuffled_clean_signal.reshape((num_batches, batch_size, *clean_signal.shape[1:]))
            noisy_approx_batched = shuffled_noisy_approx.reshape((num_batches, batch_size, *noisy_approx.shape[1:]))
            noisy_signal_batched = shuffled_noisy_signal.reshape((num_batches, batch_size, *noisy_signal.shape[1:]))
            return clean_signal_batched, noisy_approx_batched, noisy_signal_batched

        # Define a looped batch iterator that reshuffles after each pass
        def batch_iterator(rng):
            while True:
                # Shuffle the entire dataset using JAX's random module
                rng, key = random.split(rng)
                clean_signal_batched, noisy_approx_batched, noisy_signal_batched = shuffle_data(key, clean_signal, noisy_approx)
                
                # Yield each batch one by one in sequence
                for i in range(num_batches):
                    yield clean_signal_batched[i], noisy_approx_batched[i], noisy_signal_batched[i]

        return batch_iterator(rng)

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
