import jax
from jax import jit, random
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.random import split, key
from cr.wavelets import wavedec, waverec, downcoef
import pickle
from flax import linen as nn
import matplotlib.pyplot as plt
from time import time

import sys
import os

# # Add the /utils directory to the system path
sys.path.append(os.path.abspath('C:/Users/omnic/OneDrive/Documents/MIT/Programming/dae/flax/dae'))

from models import model

def denoise_bi_exponential():
    # Generate bi-exponential decay
    
    rng = 2024*int(time())
    # rng = 3496215516992
    print(f"rng: {rng}")
    rng = key(2024*int(time()))
    rng, key1, key2, key3, key4 = split(rng, 5)
    
    # Define the test data parameters
    data_args = {
        "params": {
            "a1": 0.6, 
            "a2": 0.4, 
            "tau1_min": 20, 
            "tau1_max": 120, 
            "tau2_min": 80, 
            "tau2_max": 180,
        },
        "t_max": 400, 
        "t_len": 1120, 
        "SNR": 100,
        "wavelet": "coif6", 
        "mode": "zero",
        "dtype": jnp.float32,
    }
    
    t = jnp.linspace(0, data_args["t_max"], data_args["t_len"])
    a1, a2 = 0.6, 0.4
    tau1 = random.uniform(key1, 
                   minval=data_args["params"]["tau1_min"], 
                   maxval=data_args["params"]["tau1_max"], 
                   shape=())
    tau2 = random.uniform(key2, 
                   minval=data_args["params"]["tau2_min"], 
                   maxval=data_args["params"]["tau2_max"], 
                   shape=())
    tau1, tau2 = 40, 120
    decay = a1 * jnp.exp(-t/tau1) + a2 * jnp.exp(-t/tau2)
    
    print(f"Amplitudes: {a1}, {a2}")
    print(f"Decay constants: {tau1}, {tau2}")

    # Add Gaussian noise
    SNR = 100
    noise_scale = 1/SNR
    noise = noise_scale * random.normal(key3, shape=t.shape)
    noisy_decay = decay + noise

    # Wavelet decomposition
    wavelet = 'coif6'
    mode = 'zero'
    coeffs = wavedec(noisy_decay, wavelet, mode=mode)
    coeffs_clean = wavedec(decay, wavelet, mode=mode)
    clean_approx = coeffs_clean[0]

    # Load neural network model
    with open(r"C:\Users\omnic\OneDrive\Documents\MIT\Programming\dae\flax\permanent_saves\thurs_night_trio_joat.pkl", 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Current favourites:
    # permanent_saves\thurs_lunch_current_best.pkl: Has excellent phase reduction, but not great magnitude reduction
    # permanent_saves\68_95_20_0_6365_100snr_var.pkl: best magnitude reduction so far, I think, pretty good phase reduction.
    # permanent_saves\thurs_19_latent_30.pkl: Pretty good, but I'm not sure if it's the best.
    # permanent_saves\fft_m_max.pkl: 

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
    
    print(f"mse(wt, noisy): {get_mse_loss(noisy_approx, clean_approx)}")
    print(f"mse(wt, denoised): {get_mse_loss(denoised_approx_coeffs, clean_approx)}\n")
    print(f"mse ratio: {get_mse_loss(denoised_approx_coeffs, clean_approx)/get_mse_loss(noisy_approx, clean_approx)}")
    
    print(f"mse(t, noisy): {get_mse_loss(injected_original, decay)}")
    print(f"mse(t, denoised): {get_mse_loss(injected_denoised, decay)}\n")
    print(f"mse ratio: {get_mse_loss(injected_denoised, decay)/get_mse_loss(injected_original, decay)}")

    # # Plot comparison
    # plt.title("Comparison of noisy and denoised signals")
    # plt.plot(t, noisy_decay - decay, label='Noisy')
    # plt.plot(t, denoised_decay - decay, label='Denoised')
    # plt.xlabel("time (ms)")
    # plt.ylabel("signal amplitude")
    # plt.legend()
    # plt.show()
    
    plt.title("Comparison of noise before and after denoising\nin the time domain")
    plt.plot(t, noisy_decay - decay, label='Noise (all freq)')
    plt.plot(t, jnp.zeros_like(t), label='zero', linewidth=0.5, color='black')
    plt.plot(t, injected_original - decay, label='Before denoising (low freq)')
    plt.plot(t, injected_denoised - decay, label='After denoising (low freq)')
    plt.xlabel("time (ms)")
    plt.ylabel("signal noise amplitude")
    plt.legend()
    plt.show()
    
    
    
    noisy_approx_ish = downcoef(part='a', data=noisy_decay, wavelet=wavelet, mode=mode, level=4)
    
    print(noisy_approx_ish.shape)
    print(noisy_approx.shape)
    
    # Now I want to test out what the fourier transform of the noiseless and injected signals looks like
    decay_fft = jnp.fft.fftshift(jnp.fft.fft(decay))
    injected_original_fft = jnp.fft.fftshift(jnp.fft.fft(injected_original))
    injected_denoised_fft = jnp.fft.fftshift(jnp.fft.fft(injected_denoised))
    
    
    # plt.plot(jnp.abs(injected_original_fft-decay_fft)[520:600], label='Noise before denoising')
    # plt.plot(jnp.abs(injected_denoised_fft-decay_fft)[520:600], label='Noise after denoising')
    # plt.title("Comparison of the noise before and after denoising\nin the frequency domain")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Amplitude")
    # # plt.plot(jnp.abs(decay_fft), label='Clean')
    # plt.legend()
    # plt.show()
    
    # I've realized that the above does not tell the whole picture. 
    # I need to use the magnitude and the phase separately to get a better idea of what's going on
    
    decay_fft_mag = jnp.abs(decay_fft)
    decay_fft_phase = jnp.angle(decay_fft)
    
    injected_original_fft_mag = jnp.abs(injected_original_fft)
    injected_original_fft_phase = jnp.angle(injected_original_fft)
    
    injected_denoised_fft_mag = jnp.abs(injected_denoised_fft)
    injected_denoised_fft_phase = jnp.angle(injected_denoised_fft)
    
    plt.plot((injected_original_fft_mag - decay_fft_mag)[520:600], label='Noise before denoising')
    plt.plot((injected_denoised_fft_mag - decay_fft_mag)[520:600], label='Noise after denoising')
    plt.title("Comparison of the noise before and after denoising\nin the magnitude of the fourier transform")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.show()
    
    plt.plot((injected_original_fft_phase - decay_fft_phase)[520:600], label='Noise before denoising')
    plt.plot((injected_denoised_fft_phase - decay_fft_phase)[520:600], label='Noise after denoising')
    plt.title("Comparison of the noise before and after denoising\nin the phase of the fourier transform")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase")
    plt.show()
    
    print(f"mse(fft_m, noisy): {get_mse_loss(injected_original_fft_mag, decay_fft_mag)}")
    print(f"mse(fft_m, denoised): {get_mse_loss(injected_denoised_fft_mag, decay_fft_mag)}\n")
    
    print(f"mse(fft_p, noisy): {get_mse_loss(injected_original_fft_phase, decay_fft_phase)}")
    print(f"mse(fft_p, denoised): {get_mse_loss(injected_denoised_fft_phase, decay_fft_phase)}\n")
    
    
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