from jax import jit, vmap, tree_util
from jax.debug import print as jprint
import jax.numpy as jnp
from jax.scipy.special import erfc
from cr.wavelets import wavedec, waverec
from time import time

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
def get_l2_loss(params, alpha=0.00001):
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
def fft_mse_loss(clean_signal, noisy_signal, mag_scale, phase_scale, mag_max_scale, phase_max_scale):
    clean_fft = jnp.fft.fft(clean_signal)
    noisy_fft = jnp.fft.fft(noisy_signal)
    
    clean_mag = jnp.abs(clean_fft)[0:34]
    clean_phase = jnp.angle(clean_fft)[0:34]
    
    noisy_mag = jnp.abs(noisy_fft)[0:34]
    noisy_phase = jnp.angle(noisy_fft)[0:34]
    
    mag_mean = jnp.sqrt(
        jnp.mean(jnp.square(clean_mag - noisy_mag)) +
        jnp.mean(jnp.abs(clean_mag - noisy_mag))
        )*mag_scale
    phase_mean = jnp.mean(jnp.square(clean_phase - noisy_phase))*phase_scale
    
    mag_max = jnp.max(jnp.abs(clean_mag[0:5] - noisy_mag[0:5]))*mag_max_scale
    phase_max = jnp.max(jnp.abs(clean_phase[0:5] - noisy_phase[0:5]))*phase_max_scale
    
    return mag_mean, phase_mean, mag_max, phase_max

# Combine the loss functions into a single value
def create_compute_metrics(wavelet, mode):

    noise_injection = create_noise_injection(wavelet, mode)

    @jit
    def compute_metrics(recon_approx, noisy_approx, mean, logvar, clean_signal, model_params):

        # Noise injection/preprocessing
        injected_denoised = noise_injection(recon_approx, clean_signal)
        
        # jprint(f"recon_approx: {recon_approx.shape}")
        # jprint(f"noisy_approx: {noisy_approx.shape}")
        # jprint(f"mean: {mean.shape}")
        # jprint(f"logvar: {logvar.shape}")
        # jprint(f"clean_signal: {clean_signal.shape}")
        # jprint(f"model_params: {model_params}")
        # jprint(f"injected_denoised: {injected_denoised.shape}")
        
        # calculating losses    
        metrics = {}
        
        normal_weights = {
            "wt": 3500,
            "t": 300000,
            "fft_m": 6,
            "fft_p": 4000000,
            "fft_m_max": 10.0,
            "fft_p_max": 500.0,
            "l2": 0.00001,
        }
        
        metrics["mse_wt"] = get_mse_loss(recon_approx, noisy_approx, scale=normal_weights["wt"]/10).mean()
        # metrics["mse_t"] = get_mse_loss(injected_denoised, clean_signal, scale=normal_weights["t"]/10).mean()
        mag, phase, mag_max, phase_max = fft_mse_loss(
            clean_signal, 
            injected_denoised, 
            mag_scale=normal_weights["fft_m"],
            phase_scale=normal_weights["fft_p"],
            mag_max_scale=normal_weights["fft_m_max"]/10,
            phase_max_scale=normal_weights["fft_p_max"]/10,
        )
        metrics["mse_fft_m"] = mag.mean()
        metrics["mse_fft_p"] = phase.mean()
        metrics["mse_fft_m_max"] = mag_max.mean()
        metrics["mse_fft_p_max"] = phase_max.mean()
        metrics["var_fft_m_max"] = mag_max.var()*10
        # metrics["kl"] = get_kl_divergence_lognorm(mean, logvar).mean()
        # metrics["kl"] = get_kl_divergence_truncated_normal(mean, logvar).mean()
        
        # metrics["max"] = get_max_loss(injected_denoised, clean_signal).mean()
        
        metrics["l2"] = get_l2_loss(model_params)
        
        # for key, value in metrics.items():
        #     jprint(f"key: {key}")
        #     jprint("value: {}", value)
        
        metrics["loss"] = jnp.sum(jnp.array([value for _, value in metrics.items()]))
        
        return metrics
    return compute_metrics

def print_metrics(epoch, metrics, start_time, new_best=False):
    print(
        f"New best loss at epoch {epoch + 1}, " if new_best else f"epoch: {epoch + 1}, ",
        f"time {time()-start_time:.2f}s, "
        f"loss: {metrics['loss']:.4f}, "
        f"mse_wt: {metrics['mse_wt']:.4f}, "
        # f"mse_t: {metrics['mse_t']:.4f}, "
        f"mse_fft_m: {metrics['mse_fft_m']:.4f}, "
        f"mse_fft_p: {metrics['mse_fft_p']:.4f}, "
        f"mse_fft_m_max: {metrics['mse_fft_m_max']:.4f}, "
        f"mse_fft_p_max: {metrics['mse_fft_p_max']:.4f}, "
        f"var_fft_m_max: {metrics['var_fft_m_max']:.4f}, "
        # f"kl: {metrics['kl']:.8f}, "
        # f"mae: {metrics['mae']:.8f}, "
        # f"max: {metrics['max']:.5f}, "
        # f"huber: {metrics['huber']:.8f}, "
        # f"log_mse: {metrics['log_mse']:.8f}, "
        f"l2: {metrics['l2']:.8f}"
            )
    