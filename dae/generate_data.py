import jax.numpy as jnp
import jax
from cr.wavelets import wavedec
from typing import Union

def create_multi_exponential_decay(t):
    def multi_exponential_decay(params):
        decay = jnp.sum(params[::2] * jnp.exp(-t[:, None] / params[1::2]), axis=1)
        return decay
    return jax.jit(multi_exponential_decay)

def create_wavelet_decomposition(wavelet):
    def wavelet_decomposition(data):
        approximation_coeffs = wavedec(data=data, wavelet=wavelet, mode='symmetric')[0]
        return approximation_coeffs
    return jax.jit(wavelet_decomposition)

# JIT compile and parallelize single data generation
def create_generate_single_data(t, noise_power, wavelet):
    
    multi_exponential_decay = create_multi_exponential_decay(t)
    wavelet_decomposition = create_wavelet_decomposition(wavelet)
    
    # Efficient noise generation inside the JIT function
    @jax.jit
    def generate_single_data(key, params):
        decay_truth = multi_exponential_decay(params)
        noisy_signal = decay_truth + noise_power * jax.random.normal(key, shape=t.shape)

        # Perform wavelet decomposition
        noisy_coeffs = wavelet_decomposition(noisy_signal)
        clean_coeffs = wavelet_decomposition(decay_truth)

        return noisy_coeffs, clean_coeffs
    
    return generate_single_data

def __generate_random_uniform_array(key, iterations, min, max):
    return jax.random.uniform(key, shape=(iterations,), minval=min, maxval=max)

__generate_random_uniform_array = jax.jit(__generate_random_uniform_array, static_argnums=1)

def generate_params_array(key, iterations, min, max):
    if min == max:
        return min * jnp.ones(iterations)
    else:
        return __generate_random_uniform_array(key, iterations, min, max)

def create_generate_basic_data(
        params: dict, 
        t_max: int, 
        t_len: int, 
        SNR: Union[int, float], 
        wavelet: str, 
        dtype: type = jnp.float64
    ):
    
    # Use linspace directly in JAX with dtype
    t = jnp.linspace(0, t_max, t_len, dtype=dtype)
    noise_power = (params["a1"] + params["a2"]) / SNR
    generate_single_data = create_generate_single_data(t, noise_power, wavelet)
    batched_generate = jax.vmap(generate_single_data, in_axes=(0, 1))
    
    def generate_basic_data(key, iterations):
        # Generate random tau values for all iterations (inside JIT context)
        key, key1, key2 = jax.random.split(key, 3)
        
        param_array = jnp.array([
            generate_params_array(0.0, iterations, params["a1"], params["a1"]),
            generate_params_array(key1, iterations, params["tau1_min"], params["tau1_max"]),
            generate_params_array(0.0, iterations, params["a2"], params["a2"]),
            generate_params_array(key2, iterations, params["tau2_min"], params["tau2_max"]),
        ])

        # Use vmap to parallelize over iterations, ensuring efficient batching
        rng_keys = jax.random.split(key, iterations)
        
        # Return the dataset as a JAX array (already the default dtype for jnp arrays)
        dataset = batched_generate(rng_keys, param_array)
        
        return dataset
    
    return generate_basic_data

# Okay, I've made lots of changes, and now the comments on this file are wrong. Can you remove all the comments, and then add comments explaining the purpose of each function and their inputs and outputs, along with explanations for particularly tricky lines of code:
