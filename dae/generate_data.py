import jax.numpy as jnp
import jax
from typing import Union
import data_processing

# JIT compile and parallelize single data generation
def create_generate_single_data(t, wavelet, mode, max_dwt_level):
    
    multi_exponential_decay = data_processing.create_multi_exponential_decay(t)
    # wavelet_decomposition = create_wavelet_decomposition(wavelet, mode)
    # wavelet_approx = data_processing.create_wavelet_approx(wavelet, mode, max_dwt_level)
    
    # Efficient noise generation inside the JIT function
    @jax.jit
    def generate_single_data(key, params, noise_power):
        clean_signal = multi_exponential_decay(params)
        noisy_signal = clean_signal + noise_power * jax.random.normal(key, shape=t.shape)
        # noisy_signal = clean_signal + noise_power * jax.random.laplace(key, shape=t.shape)
        # noisy_signal = clean_signal + noise_power * jax.random.cauchy(key, shape=t.shape)

        # Perform wavelet decomposition
        # noisy_coeffs = wavelet_approx(noisy_signal)
        noisy_coeffs = jnp.array([jnp.nan])

        return clean_signal, noisy_coeffs, noisy_signal, params, noise_power
    
    return generate_single_data

# def __generate_random_uniform_array(key, iterations, min, max):
#     return jax.random.uniform(key, shape=(iterations,), minval=min, maxval=max)

# __generate_random_uniform_array = jax.jit(__generate_random_uniform_array, static_argnums=1)

# def generate_params_array(key, iterations, min, max):
#     if min == max:
#         return min * jnp.ones(iterations)
#     else:
#         return __generate_random_uniform_array(key, iterations, min, max)

def create_generate_basic_data(
        params: dict, 
        t_max: int, 
        t_len: int, 
        SNR: Union[int, float], 
        wavelet: str, 
        mode: str,
        max_dwt_level: int,
        dtype: type,
    ):
    
    # Use linspace directly in JAX with dtype
    t = jnp.linspace(0, t_max, t_len, dtype=dtype)
    generate_single_data = create_generate_single_data(t, wavelet, mode, max_dwt_level)
    batched_generate = jax.vmap(generate_single_data, in_axes=(0, 0, 0))
    
    def generate_basic_data(key, iterations):
        # Generate random tau values for all iterations (inside JIT context)
        key, key_amp, key_ampsum, key_tau, key_noise = jax.random.split(key, 5)
        
        amplitudes = jax.random.dirichlet(
            key=key_amp,
            alpha=jnp.ones(params["decay_count"]),
            shape=(iterations,),
            dtype=dtype,
        )
        
        # amplitudes = jnp.array([
        #     jnp.ones(iterations)*0.25, 
        #     jnp.ones(iterations)*0.75, 
        # ], dtype=dtype).T
        
        # # Generate random amplitude sums for each iteration
        amp_sum = jax.random.gamma(
            key=key_ampsum, 
            a=2.0,
            shape=(iterations,), 
            dtype=dtype,
        ) / 20.0 + 0.95
        
        # Combine the amplitudes and amplitude sums
        amplitudes = amplitudes * amp_sum[:, None]
        
        decay_constants = jax.random.uniform(
            key=key_tau,
            shape=(iterations, params["decay_count"]),
            minval=params["tau_min"],
            maxval=params["tau_max"],
            dtype=dtype,
        )
        
        # sort the decay constants in ascending order
        decay_constants = jnp.sort(decay_constants, axis=1)
        
        param_array = data_processing.format_params(amplitudes, decay_constants)
        
        # jax.debug.print("param_array.shape: {}", param_array.shape)
        
        SNR_array = (jax.random.normal(
            key=key_noise, 
            shape=(iterations,), 
            dtype=dtype,
        ) / 5.0 + 1) * SNR
        
        noise_power_array = amp_sum / SNR_array
        # noise_power_array = 1.0 / SNR_array
        # noise_power_array = jnp.ones(iterations) / SNR
        
        keys = jax.random.split(key, iterations)
        
        # Return the dataset as a JAX array (already the default dtype for jnp arrays)
        dataset = batched_generate(keys, param_array, noise_power_array)
        
        return dataset
    
    return generate_basic_data

# Okay, I've made lots of changes, and now the comments on this file are wrong. Can you remove all the comments, and then add comments explaining the purpose of each function and their inputs and outputs, along with explanations for particularly tricky lines of code:
