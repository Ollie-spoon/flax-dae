import jax.numpy as jnp
import numpy as np
from pywt import wavedec
from typing import Union

def multi_exponential_decay(t, params):
    if len(params) % 2 != 0:
        raise ValueError("Params must have an even length")
    
    decay = np.sum([params[i] * np.exp(-t / params[i+1]) for i in range(0, len(params), 2)], axis=0)
    
    return decay

def generate_basic_data(
        params: dict, 
        t_max: int, 
        t_len: int, 
        SNR: Union[int, float], 
        wavelet: str, 
        iterations: int, 
        dtype: type=np.float64,
    ):
    a1 = params["a1"]
    a2 = params["a2"]
    t = np.linspace(0, t_max, t_len, dtype=dtype)
    noise_power = (a1+a2)/SNR

    tau1 = np.random.uniform(low=params["tau1_min"], high=params["tau1_max"], size=iterations)
    tau2 = np.random.uniform(low=params["tau2_min"], high=params["tau2_max"], size=iterations)
    
    # print(f"tau1.shape: {tau1.shape}")
    
    params = np.column_stack([a1*np.ones(iterations), tau1, a2*np.ones(iterations), tau2])
    
    # print(f"params.shape: {params.shape}")
    
    # Run the first iteration outside the loop to initialize the dataset
    decay_truth = multi_exponential_decay(t, params[0])
    noisy = decay_truth + np.random.normal(scale=noise_power, size=len(t))
    
    clean_coeffs = wavedec(decay_truth, wavelet, mode='symmetric')[0]
    noisy_coeffs = wavedec(noisy, wavelet, mode='symmetric')[0]

    dataset = np.empty((iterations, 2, len(clean_coeffs)))
    
    dataset[0, 0] = noisy_coeffs
    dataset[0, 1] = clean_coeffs
    
    del clean_coeffs, noisy_coeffs, noisy
    
    for i in range(1, iterations):
        decay_truth = multi_exponential_decay(t, params[i])

        dataset[i, 0] = wavedec(
            decay_truth+np.random.normal(scale=noise_power, size=len(t)),
            wavelet, 
            mode='symmetric',
        )[0]
        dataset[i, 1] = wavedec(
            decay_truth, 
            wavelet, 
            mode='symmetric',
        )[0]
    
    jax_dtype = jnp.float64 if dtype == np.float64 else jnp.float32
    
    return jnp.asarray(dataset, dtype=jax_dtype)