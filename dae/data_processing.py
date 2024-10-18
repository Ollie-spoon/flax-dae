# import numpy as np
# import matplotlib.pyplot as plt
# import pywt
import jax
import jax.numpy as jnp
from cr.wavelets import wavedec, downcoef


# # lets generate a test exponential decay

# t = np.linspace(0, 100, 1000)

# a1 = 0.6
# a2 = 0.4
# tau1 = 15
# tau2 = 25

# decay = a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)

# # lets add some noise to the decay
# SNR = 100
# noise_power = (a1 + a2) / SNR
# noisy_decay = decay + np.random.normal(scale=noise_power, size=len(t))

# # lets plot the decay and the noisy decay
# plt.plot(t, noisy_decay, label="Noisy Decay")
# plt.plot(t, decay, label="Clean Decay")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.title("Exponential Decay")
# plt.show()

# # Now we're going to take the wavelet transform of the decay and the noisy decay

# wavelet = "coif6"
# clean_coeffs = pywt.wavedec(decay, wavelet, mode="symmetric")
# noisy_coeffs = pywt.wavedec(noisy_decay, wavelet, mode="symmetric")

# # lets plot the wavelet coefficients
# plt.plot(np.log10(noisy_coeffs[0]), label="Noisy Decay")
# plt.plot(np.log10(clean_coeffs[0]), label="Clean Decay")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.title("Wavelet Coefficients")
# plt.show()

# # Lets now plot the ratio of the difference between the noisy and clean coefficients to the clean coefficients
# diff_ratio = np.abs((clean_coeffs[0] - noisy_coeffs[0]) / clean_coeffs[0])
# plt.plot(diff_ratio)
# plt.xlabel("Time")
# plt.ylabel("Difference Ratio")
# plt.title("Difference Ratio of Wavelet Coefficients")
# plt.show()


# We need a function to add the outputs of the neural network to the original noisy data

@jax.vmap
def add_difference(noisy_data, difference):
    return noisy_data + difference

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

# We now want to analyse the example batch and extract from it the standard 
# deviation of the noise from each point in the signal across the batch
def get_noise_std(batch, wavelet, mode, max_dwt_level):
    clean_signal, noisy_approx, _ = batch
    
    # Calculate the clean approximation
    wavelt_approx = create_wavelet_approx(wavelet, mode, max_dwt_level)
    clean_approx = jax.vmap(wavelt_approx)(clean_signal)
    
    # Calculate the difference between the noisy data and the noisy approximation
    difference = noisy_approx - clean_approx
    # Calculate the standard deviation of the difference across the batch
    noise_std = jnp.std(difference, axis=0)
    return noise_std

# We want to reformat the prediction parameters so that they are roughly normalized
@jax.jit
def normalize_exp_params(taus, amps, noise_power):
    # Normalize the taus
    taus_norm = jnp.log10(taus)-1
    # Normalize the amplitudes
    amps_norm = amps
    # Normalize the noise power
    noise_power_norm = -jnp.log10(noise_power)-1
    
    return taus_norm, amps_norm, noise_power_norm#

# We want to reformat the prediction parameters so that they are roughly normalized
@jax.jit
def unnormalize_exp_params(taus_norm, amps_norm, noise_power_norm):
    # Normalize the taus
    taus = jnp.power(10, taus_norm + 1)
    # Normalize the amplitudes
    amps = amps_norm
    # Normalize the noise power
    noise_power = jnp.power(10, -(noise_power_norm + 1))
    
    return taus, amps, noise_power