import jax
from jax import jit, random
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.random import uniform, normal, split, key
from cr.wavelets import wavedec, waverec, downcoef
import pickle
from flax import linen as nn
import matplotlib.pyplot as plt

import sys
import os

# # Add the /utils directory to the system path
sys.path.append(os.path.abspath('C:/Users/omnic/OneDrive/Documents/MIT/Programming/dae/flax/dae'))

from models import model

def denoise_bi_exponential():
    # Generate bi-exponential decay
    
    rng = key(2027)
    rng, key1, key2, key3, key4 = split(rng, 5)
    
    t = jnp.linspace(0, 100, 1000)
    a1, a2 = 0.6, 0.4
    tau1 = uniform(key1, minval=5, maxval=30, shape=())
    tau2 = uniform(key2, minval=20, maxval=45, shape=())
    decay = a1 * jnp.exp(-t/tau1) + a2 * jnp.exp(-t/tau2)

    # Add Gaussian noise
    SNR = 100
    noise_scale = 1/SNR
    noise = noise_scale * normal(key3, shape=t.shape)
    noisy_decay = decay + noise

    # Wavelet decomposition
    wavelet = 'coif6'
    mode = 'symmetric'
    coeffs = wavedec(noisy_decay, wavelet, mode=mode)
    coeffs_clean = wavedec(decay, wavelet, mode=mode)
    clean_approx = coeffs_clean[0]

    # Load neural network model
    with open(r"C:/Users/omnic/OneDrive/Documents/MIT/Programming/dae/flax/tmp/checkpoints/checkpoint_200.pkl", 'rb') as f:
        checkpoint = pickle.load(f)

    # Pass approximation coefficients through neural network
    noisy_approx = coeffs[0]
    denoised_approx_coeffs, _, _ = eval_f(
        params=checkpoint['params'],
        noisy_data=noisy_approx,
        model_args=checkpoint['model_args'],
        z_rng=key4
    )
    
    print(f"model_args: {checkpoint['model_args']}")
    
    # print(f"type(denoised_approx_coeffs): {type(denoised_approx_coeffs)}")
    # print(f"denoised_approx_coeffs: {denoised_approx_coeffs}")
    

    # Inverse wavelet decomposition with denoised approximation coefficients
    plt.title("Comparison of noisy and denoised approximation coefficients.")
    # plt.plot(coeffs[0], label='Noisy')
    plt.plot(denoised_approx_coeffs, label='Denoised')
    plt.plot(clean_approx, label='Clean')
    plt.xlabel("index")
    plt.ylabel("coefficient amplitude")
    plt.legend()
    plt.show()
    
    coeffs[0] = denoised_approx_coeffs
    denoised_decay = waverec(coeffs, wavelet, mode=mode)
    coeffs_clean[0] = noisy_approx
    injected_original = waverec(coeffs_clean, wavelet, mode=mode)
    coeffs_clean[0] = coeffs[0]
    injected_denoised = waverec(coeffs_clean, wavelet, mode=mode)
    
    print(f"The original SNR of the signal was {1/jnp.std(noisy_decay-decay)}")
    print(f"The denoised signal SNR was {1/jnp.std(denoised_decay-decay)}")
    
    print(f"mse of denoised approx coefficients: {get_mse_loss(denoised_approx_coeffs, clean_approx)}")
    print(f"mse of noisy approx coefficients: {get_mse_loss(noisy_approx, clean_approx)}")
    
    print(f"The mse loss for the noisy signal was {get_mse_loss(injected_original, decay)}")
    print(f"The mse loss for the denoised signal was {get_mse_loss(injected_denoised, decay)}")

    # Plot comparison
    plt.title("Comparison of noisy and denoised signals")
    plt.plot(t, noisy_decay - decay, label='Noisy')
    plt.plot(t, denoised_decay - decay, label='Denoised')
    plt.xlabel("time (ms)")
    plt.ylabel("signal amplitude")
    plt.legend()
    plt.show()
    
    plt.title("Comparison of noisy and denoised approximation coefficient injections")
    plt.plot(t, noisy_decay - decay, label='Noisy')
    plt.plot(t, injected_original - decay, label='Noisy Injected')
    plt.plot(t, injected_denoised - decay, label='Denoised Injected')
    plt.xlabel("time (ms)")
    plt.ylabel("signal noise amplitude")
    plt.legend()
    plt.show()
    
    
    
    noisy_approx_ish = downcoef(part='a', data=noisy_decay, wavelet=wavelet, mode=mode, level=4)
    
    print(noisy_approx_ish.shape)
    print(noisy_approx.shape)
    
    # For this section we are going to test out some interesting things
    
    # First, we're going to see how much the edges effect the signal
    # Is this data lost?
    
    # coeffs_clean[0] = clean_approx
    # # coeffs_clean[0][:10] = noisy_approx[:10]
    # for i in range(1,21):
    #     coeffs_clean[0] = coeffs_clean[0].at[-i:].set(noisy_approx[-i:])
    #     reconstructed_clean = waverec(coeffs_clean, wavelet, mode=mode)
        
    #     # plt.plot(t, decay, label="original")
    #     # plt.plot(t, reconstructed_clean, label="Reconstructed")
    #     # plt.legend()
    #     # plt.show()
        
    #     plt.title(f"for changing the last {i} values")
    #     plt.plot(t, decay-reconstructed_clean)
    #     plt.show()

def get_mse_loss(recon_x, noiseless_x):
    return jnp.mean(jnp.square(recon_x - noiseless_x))

# Define the evaluation function
def eval_f(params, noisy_data, model_args, z_rng):
    
    def eval_model(vae):
        return vae(noisy_data, z_rng, deterministic=True)

    return nn.apply(eval_model, model(**model_args))({'params': params})


denoise_bi_exponential()