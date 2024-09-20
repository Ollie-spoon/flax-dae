# import numpy as np
# import matplotlib.pyplot as plt
# import pywt
import jax

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