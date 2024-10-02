import jax.numpy as jnp
import jax
from cr.wavelets import wavedec, downcoef
from typing import Union

def create_multi_exponential_decay(t):
    def multi_exponential_decay(params):
        decay = jnp.sum(params[::2] * jnp.exp(-t[:, None] / params[1::2]), axis=1)
        return decay
    return jax.jit(multi_exponential_decay)

def create_wavelet_decomposition(wavelet, mode):
    def wavelet_decomposition(data):
        coeffs = wavedec(data=data, wavelet=wavelet, mode=mode)
        return coeffs
    return jax.jit(wavelet_decomposition)

def create_wavelet_approx(wavelet, mode, max_dwt_level):
    def wavelet_approx(data):
        coeffs = downcoef(part='a', data=data, wavelet=wavelet, mode=mode, level=max_dwt_level)
        return coeffs
    return jax.jit(wavelet_approx)

# JIT compile and parallelize single data generation
def create_generate_single_data(t, wavelet, mode, max_dwt_level):
    
    multi_exponential_decay = create_multi_exponential_decay(t)
    # wavelet_decomposition = create_wavelet_decomposition(wavelet, mode)
    wavelet_approx = create_wavelet_approx(wavelet, mode, max_dwt_level)
    
    # Efficient noise generation inside the JIT function
    @jax.jit
    def generate_single_data(key, params, noise_power):
        clean_signal = multi_exponential_decay(params)
        noisy_signal = clean_signal + noise_power * jax.random.normal(key, shape=t.shape)
        # noisy_signal = clean_signal + noise_power * jax.random.laplace(key, shape=t.shape)
        # noisy_signal = clean_signal + noise_power * jax.random.cauchy(key, shape=t.shape)

        # Perform wavelet decomposition
        noisy_coeffs = wavelet_approx(noisy_signal)

        return clean_signal, noisy_coeffs
    
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
        
        # amplitudes = jax.random.dirichlet(
        #     key=key_amp,
        #     alpha=jnp.ones(params["decay_count"]),
        #     shape=(iterations,),
        #     dtype=dtype,
        # )
        
        amplitudes = jnp.array([
            jnp.ones(iterations)*0.25, 
            jnp.ones(iterations)*0.75, 
        ], dtype=dtype).T
        
        # # Generate random amplitude sums for each iteration
        # amp_sum = jax.random.gamma(
        #     key=key_ampsum, 
        #     a=2.0,
        #     shape=(iterations,), 
        #     dtype=dtype,
        # ) / 20.0 + 0.95
        
        # # Combine the amplitudes and amplitude sums
        # amplitudes = amplitudes * amp_sum[:, None]
        
        decay_constants = jax.random.uniform(
            key=key_tau,
            shape=(iterations, params["decay_count"]),
            minval=params["tau_min"],
            maxval=params["tau_max"],
            dtype=dtype,
        )
        
        param_array = jnp.empty((iterations, 2*params["decay_count"]), dtype=dtype)
        param_array = param_array.at[:, ::2].set(amplitudes)
        param_array = param_array.at[:, 1::2].set(decay_constants)
        
        # jax.debug.print("param_array.shape: {}", param_array.shape)
        
        SNR_array = (jax.random.normal(
            key=key_noise, 
            shape=(iterations,), 
            dtype=dtype,
        ) / 10.0 + 0.975) * SNR
        
        # noise_power_array = amp_sum / SNR_array
        noise_power_array = 1.0 / SNR_array
        
        keys = jax.random.split(key, iterations)
        
        # Return the dataset as a JAX array (already the default dtype for jnp arrays)
        dataset = batched_generate(keys, param_array, noise_power_array)
        
        return dataset
    
    return generate_basic_data

# Okay, I've made lots of changes, and now the comments on this file are wrong. Can you remove all the comments, and then add comments explaining the purpose of each function and their inputs and outputs, along with explanations for particularly tricky lines of code:
