# import numpy as np
# import matplotlib.pyplot as plt
# import pywt
import jax
from jax import jit, vmap
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

@vmap
def add_difference(noisy_data, difference):
    return noisy_data + difference

def create_multi_exponential_decay(t):
    def multi_exponential_decay(params):
        decay = jnp.sum(params[::2] * jnp.exp(-t[:, None] / params[1::2]), axis=1)
        return decay
    return jit(multi_exponential_decay)

def create_wavelet_decomposition(wavelet, mode):
    def wavelet_decomposition(data):
        coeffs = wavedec(data=data, wavelet=wavelet, mode=mode)
        return coeffs
    return jit(wavelet_decomposition)

def create_wavelet_approx(wavelet, mode, max_dwt_level):
    def wavelet_approx(data):
        coeffs = downcoef(part='a', data=data, wavelet=wavelet, mode=mode, level=max_dwt_level)
        return coeffs
    return jit(wavelet_approx)

# We now want to analyse the example batch and extract from it the standard 
# deviation of the noise from each point in the signal across the batch
def get_noise_std(batch, wavelet, mode, max_dwt_level):
    clean_signal, noisy_approx, _ = batch
    
    # Calculate the clean approximation
    wavelt_approx = create_wavelet_approx(wavelet, mode, max_dwt_level)
    clean_approx = vmap(wavelt_approx)(clean_signal)
    
    # Calculate the difference between the noisy data and the noisy approximation
    difference = noisy_approx - clean_approx
    # Calculate the standard deviation of the difference across the batch
    noise_std = jnp.std(difference, axis=0)
    return noise_std

# We want to reformat the prediction parameters so that they are roughly normalized
@jit
def normalize_exp_params(taus, amps):
    
    # Normalize the taus
    taus_norm = jnp.log10(taus)-1
    # Normalize the amplitudes
    amps_norm = amps
    
    return taus_norm, amps_norm

# We want to reformat the prediction parameters so that they are roughly normalized
@jit
def unnormalize_exp_params(taus_norm, amps_norm):
    
    # Normalize the taus
    taus = jnp.power(10, taus_norm + 1)
    # Normalize the amplitudes
    amps = amps_norm
    
    return taus, amps

@jit
def reformat_prediction(prediction):
    amps_norm, taus_norm = extract_params(prediction)
    taus, amps = unnormalize_exp_params(taus_norm, amps_norm)
    prediction = format_params(amps, taus)
    return prediction

@jit
def extract_params(params):
    """
    Extracts amplitude and tau parameters from a parameter array.
    """
    
    amps = params[:, ::2]
    taus = params[:, 1::2]
    
    return amps, taus 

@jit 
def format_params(amps, taus):
    """
    Formats amplitude and tau parameters into a single array.
    """
    output = jnp.empty((amps.shape[0], amps.shape[1] + taus.shape[1]))
    output = output.at[:, ::2].set(amps)
    output = output.at[:, 1::2].set(taus)
    return output

# Normalize the signal
@vmap
@jit
def normalize_signal(signal):
    """
    Normalizes a signal by dividing by the first value.
    """
    return signal / signal[0], signal[0]

# Un-normalize the signal
@vmap
@jit
def unnormalize_signal(signal, x0):
    """
    Un-normalizes a signal by multiplying by the first value.
    """
    return signal * x0

# Convert x to dx
@vmap
@jit
def dx_from_x(x):
    """
    Converts a signal x to a signal of the same length dx, where dx[i] = x[i+1] - x[i].
    When combined with x[0], this can be used to reconstruct x.
    
    Inputs:
    x: jnp.array, the signal to convert to dx.
    
    Outputs:
    dx: jnp.array, the signal of differences.
    """
    dx = jnp.diff(x)
    dx = jnp.append(dx, dx[-1])
    return dx

@vmap
@jit
def x_from_dx(dx):
    """
    Converts a signal of differences dx to a signal x, where dx[i] = x[i-1] + x[i].
    This function is specifically for normalized signals where x[0] = 1.
    To exchange for a different x[0], simply add it to the output in place of the "1".
    
    Inputs:
    dx: jnp.array, the signal of differences.
    
    Outputs:
    x: jnp.array, the reconstructed signal.
    """
    
    x = jnp.append(0, jnp.cumsum(dx[1:])) + 1 + dx[0]
    return x

class TransformerTokenizer:
    
    def __init__(self, token_dim, max_tokens, overlap_per_token):
        self.token_dim = token_dim
        self.max_tokens = max_tokens
        self.overlap_per_token = overlap_per_token
        
        self.token_dim_no_overlap = token_dim - overlap_per_token
        
        # len values correspond to the length in signal represenation
        self.min_len = token_dim
        self.max_len = self.get_len_from_tokens(max_tokens)
    
    ## TOKENIZAITON FUNCTIONS ##
    @vmap
    @jit
    def tokenize(self, data):
        
        # first we need to clip and pad the data to fit the tokens
        padded = self.pad_signal(data)
        
        
        # Tokenize the data:
        # 1. Token = padded[i:i+token_dim] or segments of token_dim length
        # 2. for i starting at 0, increasing by token_dim-overlap_per_token
        # 3. until i+token_dim >= len(padded)
        tokenized = jnp.zeros((self.max_tokens, self.token_dim))
        
        return self.populate_tokens(self, padded, tokenized)

    @vmap
    @jit
    def pad_signal(self, data):
        complete_tokens = jnp.min(self.get_tokens_from_len(data.shape[-1]), self.max_tokens)
        tokenized_len = self.get_len_from_tokens(complete_tokens)
        padded = jnp.zeros(self.max_len)
        padded = padded.at[:tokenized_len].set(data[:tokenized_len])
        return padded
    
    def populate_tokens(self, padded, tokenized, i=0):
        
        if i < self.max_tokens:
            start = i*self.token_dim_no_overlap
            end = start + self.token_dim
            
            tokenized = tokenized.at[i].set(padded[:, start:end])
            return self.fill_in_padding(padded, tokenized, i+1)
        else:
            return tokenized
    
    ## DETOKENIZAITON FUNCTIONS ##
    @vmap
    @jit
    def detokenize(self, tokens):
        
        return self.combine_tokens(
            tokens=tokens, 
            output=jnp.zeros(self.max_len), 
            tokens_left=tokens.shape[0],
        )
    
    def combine_tokens(self, tokens, output, tokens_left):
        # Pseudo code:
        # take the initial token array with input shape (tokens_left, self.token_dim)
        # if tokens_left > 1:
        #     handleoverlap(output, tokens[0])
        #     take the first token and add it to the output array
        #     return combine_tokens(tokens[1:], tokens_left-1)
        # else:
        #     return output
        
        if tokens_left > 1:
            token = tokens[0]
            
            i = self.max_tokens - tokens_left
            overlap_start = self.token_dim_no_overlap*i
            overlap_end = overlap_start + self.overlap_per_token
            
            token = token.at[:self.overlap_per_token].set(self.overlap_combine_fn(
                output[overlap_start:overlap_end], 
                token[:self.overlap_per_token],
            ))
            
            output = output.at[overlap_start:overlap_start+self.token_dim].set(token)
            
            return self.combine_tokens(tokens[1:], output, tokens_left-1)
        else:
            return output
    
    def overlap_combine_fn(self, overlap_1, overlap_2):
        return (overlap_1 + overlap_2) / 2
    
    def get_len_from_tokens(self, tokens) -> int:
        return tokens*self.token_dim_no_overlap + self.overlap_per_token
    
    def get_tokens_from_len(self, length) -> int:
        return (length - self.overlap_per_token) // self.token_dim_no_overlap + 1
    