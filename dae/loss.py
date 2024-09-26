from jax import jit, vmap, tree_util
from jax.debug import print
import jax.numpy as jnp
from jax.scipy.special import erfc
from cr.wavelets import wavedec, waverec

# Define the loss functions

# Mean Squared Error Loss
@jit
def get_mse_loss(recon_x, noiseless_x, scale=159419):
    return jnp.mean(jnp.square(recon_x - noiseless_x)) * scale

# Mean Absolute Error Loss
@vmap
@jit
def get_mae_loss(recon_x, noiseless_x):
    return jnp.mean(jnp.abs(recon_x - noiseless_x))

# Huber Loss
@vmap
@jit
def huber_loss(recon_x, noiseless_x, delta=1.0):
    diff = recon_x - noiseless_x
    abs_diff = jnp.abs(diff)
    quadratic = jnp.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    return 0.5 * quadratic**2 + delta * linear

@vmap
@jit
def get_huber_loss(recon_x, noiseless_x, delta=1.0):
    return jnp.mean(huber_loss(recon_x, noiseless_x, delta))

# Logarithmic Loss
@vmap
@jit
def abs_complex_log10(numbers):
    # Compute the complex logarithm
    absolutes = jnp.abs(numbers)  # Adding 0j to ensure complex type
    negative = (1-numbers/absolutes)
    # Return the absolute value of the complex logarithm
    return jnp.log10(absolutes)+negative

@vmap
@jit
def get_log_mse_loss(recon_x, noiseless_x, eps=1e-16):
    return jnp.square(abs_complex_log10(recon_x+eps) - abs_complex_log10(noiseless_x+eps)).mean()

# Maximum Loss
@vmap
@jit
def get_max_loss(recon_x, noiseless_x, scale=72.51828996):
    return jnp.max(jnp.abs(recon_x - noiseless_x))*scale

# L2 Regularization Loss
@jit
def get_l2_loss(params, alpha=0.0000001):
    l2_loss = tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), params)
    return alpha * sum(tree_util.tree_leaves(l2_loss))

# KL Divergence Loss

# KL Divergence Loss for Log-Normal Distribution
@jit
def get_kl_divergence_lognorm(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

# KL Divergence Loss for Truncated Normal Distribution
@jit
def get_kl_divergence_truncated_normal(mean, logvar):
    # a and b are the lower and upper bounds of the truncated normal distribution
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar)) + \
           jnp.log(erfc(-mean/jnp.sqrt(jnp.exp(logvar))))


# utility function to create noise injection function
def create_noise_injection(wavelet, mode):
    
    # Noise injection function
    @vmap
    @jit
    def noise_injection(recon_x, clean_signal):
        """ Inject noise into the clean signal via the approximation coefficients from the wavelet decomposition """
        
        # forward wavelet transform
        clean_coeffs = wavedec(clean_signal, wavelet, mode)
        
        # Noise injection
        clean_coeffs[0] = recon_x
        
        # inverse wavelet transform
        injected_denoised = waverec(clean_coeffs, wavelet, mode)
        
        return injected_denoised
    return noise_injection

# FFT MSE Loss
@jit
def fft_mse_loss(clean_signal, noisy_signal, magnitude_scale, phase_scale):
    clean_fft = jnp.fft.fft(clean_signal)
    noisy_fft = jnp.fft.fft(noisy_signal)
    
    clean_mag = jnp.abs(clean_fft)
    clean_phase = jnp.angle(clean_fft)
    
    noisy_mag = jnp.abs(noisy_fft)
    noisy_phase = jnp.angle(noisy_fft)
    
    return jnp.mean(jnp.square(clean_mag - noisy_mag))*magnitude_scale, jnp.mean(jnp.square(clean_phase - noisy_phase))*phase_scale

# Combine the loss functions into a single value
def create_compute_metrics(wavelet, mode):

    noise_injection = create_noise_injection(wavelet, mode)

    @jit
    def compute_metrics(recon_approx, noisy_approx, mean, logvar, clean_signal, model_params):

        # Noise injection/preprocessing
        injected_denoised = noise_injection(recon_approx, clean_signal)
        
        # print(f"recon_approx: {recon_approx.shape}")
        # print(f"noisy_approx: {noisy_approx.shape}")
        # print(f"mean: {mean.shape}")
        # print(f"logvar: {logvar.shape}")
        # print(f"clean_signal: {clean_signal.shape}")
        # print(f"model_params: {model_params}")
        # print(f"injected_denoised: {injected_denoised.shape}")
        
        # calculating losses    
        metrics = {}
        
        metrics["mse_wt"] = get_mse_loss(recon_approx, noisy_approx, scale=8919).mean()
        metrics["mse_t"] = get_mse_loss(injected_denoised, clean_signal, scale=159419).mean()
        mag, phs = fft_mse_loss(clean_signal, injected_denoised, magnitude_scale=312.285, phase_scale=22596.8)
        metrics["mse_fft_m"], metrics["mse_fft_p"] = mag.mean(), phs.mean()
        
        # metrics["kl"] = get_kl_divergence_lognorm(mean, logvar).mean()
        # metrics["kl"] = get_kl_divergence_truncated_normal(mean, logvar).mean()
        
        # metrics["max"] = get_max_loss(injected_denoised, clean_signal).mean()
        
        metrics["l2"] = get_l2_loss(model_params)
        
        # for key, value in metrics.items():
        #     print(f"key: {key}")
        #     print("value: {}", value)
        
        metrics["loss"] = jnp.sum(jnp.array([value for _, value in metrics.items()]))
        
        return metrics
    return compute_metrics
