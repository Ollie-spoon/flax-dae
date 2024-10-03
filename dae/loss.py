from jax import jit, vmap, tree_util
from jax.nn import relu as ReLU
from jax.debug import print as jprint
import jax.numpy as jnp
from jax.scipy.special import erfc
from cr.wavelets import wavedec, waverec
from time import time
from typing import Dict, List

#### ~~~~ Define the loss functions ~~~~ ####

# Mean Squared Error Loss
@vmap
@jit
def get_mse_loss(noiseless_x, recon_x):
    return jnp.mean(jnp.square(recon_x - noiseless_x))

# Mean Absolute Error Loss
@vmap
@jit
def get_mae_loss(noiseless_x, recon_x):
    return jnp.mean(jnp.abs(recon_x - noiseless_x))

# Huber Loss
@vmap
@jit
def huber_loss(noiseless_x, recon_x, delta=1.0):
    diff = recon_x - noiseless_x
    abs_diff = jnp.abs(diff)
    quadratic = jnp.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    return 0.5 * quadratic**2 + delta * linear

@vmap
@jit
def get_huber_loss(noiseless_x, recon_x, delta=1.0):
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
def get_log_mse_loss(noiseless_x, recon_x, eps=1e-16):
    return jnp.square(abs_complex_log10(recon_x+eps) - abs_complex_log10(noiseless_x+eps)).mean()

# Maximum Loss
@vmap
@jit
def get_max_loss(noiseless_x, recon_x):
    return jnp.max(jnp.abs(recon_x - noiseless_x))

# L2 Regularization Loss
@jit
def get_l2_loss(params):
    l2_loss = tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), params)
    return sum(tree_util.tree_leaves(l2_loss))

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
    def noise_injection(clean_signal, noisy_approx, denoised_approx):
        """ Inject noise into the clean signal via the approximation coefficients from the wavelet decomposition """

        # forward wavelet transform
        clean_coeffs = wavedec(clean_signal, wavelet, mode)
        
        # jprint("clean_coeffs: ", jnp.array([c.shape for c in clean_coeffs]))
        
        # Noise injection
        clean_approx = clean_coeffs[0]
        clean_coeffs[0] = denoised_approx
        
        
        # inverse wavelet transform
        injected_denoised = waverec(clean_coeffs, wavelet, mode)
        
        clean_coeffs[0] = noisy_approx
        
        injected_noisy = waverec(clean_coeffs, wavelet, mode)
        
        return clean_approx, injected_noisy, injected_denoised
    return noise_injection

# FFT MSE Loss
@vmap
@jit
def fft_losses(clean_signal, noisy_signal, prediction_signal):
    clean_fft = jnp.fft.fft(clean_signal)
    pred_fft = jnp.fft.fft(prediction_signal)
    noisy_fft = jnp.fft.fft(noisy_signal)
    
    clean_mag = jnp.abs(clean_fft)[0:68]
    clean_phase = jnp.angle(clean_fft)[0:68]
    
    pred_mag = jnp.abs(pred_fft)[0:68]
    pred_phase = jnp.angle(pred_fft)[0:68]
    
    noisy_mag = jnp.abs(noisy_fft)[0:68]
    noisy_phase = jnp.angle(noisy_fft)[0:68]
    
    mag_mean = jnp.sqrt(
        jnp.mean(jnp.square(clean_mag - pred_mag)) +
        jnp.mean(jnp.abs(clean_mag - pred_mag))
        )
    phase_mean = jnp.mean(jnp.square(clean_phase - pred_phase))
    
    mag_max = jnp.sum(ReLU(jnp.abs(pred_mag) - jnp.abs(noisy_mag)))
    phase_max = jnp.sum(ReLU(jnp.abs(pred_phase) - jnp.abs(noisy_phase)))
    
    # mag_max = jnp.max(jnp.abs(clean_mag - pred_mag))*mag_max_scale
    # phase_max = jnp.max(jnp.abs(clean_phase - pred_phase))*phase_max_scale
    
    return mag_mean, phase_mean, mag_max, phase_max

#### ~~~~ Define the loss functions dict ~~~~ ####

SCALE_AGNOSTIC_LOSSES = ["l2", "kl"]

def create_compute_metrics(loss_scaling: Dict[str, float], example_batch, wavelet, mode):
    """
    Creates a compute metrics function with scaled loss weights.
    
    Args:
    - loss_scaling (Dict[str, float]): Dictionary of loss scaling factors.
    - loss_types (Dict[str, str]): Dictionary of loss types (e.g., "mse", "l2", etc.).
    - example_batch: Example batch of data.
    - wavelet: Wavelet object.
    - mode: Mode string.
    
    Returns:
    - compute_metrics (Callable): Compute metrics function with scaled loss weights.
    """
    
    # Define compute metrics function with scaled loss weights
    def compute_metrics(clean_signal, noisy_approx, recon_approx, mean, logvar, model_params):
        """
        Computes metrics with scaled loss weights.
        
        Args:
        - recon_approx: Reconstructed approximation.
        - noisy_approx: Noisy approximation.
        - mean: Mean value.
        - logvar: Log variance value.
        - clean_signal: Clean signal.
        - model_params: Model parameters.
        
        Returns:
        - metrics (Dict[str, float]): Dictionary of metrics.
        """
        
        # Perform noise injection
        
        clean_approx, injected_noisy, injected_denoised  = noise_injection(clean_signal, noisy_approx, recon_approx)
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Compute losses
        if "wt" in loss_scaling:
            metrics["wt"] = get_mse_loss(clean_approx, recon_approx).mean()
        if "t" in loss_scaling:
            metrics["t"] = get_mse_loss(clean_signal, injected_denoised).mean()
        if "fft_m" in loss_scaling or "fft_p" in loss_scaling or "fft_m_max" in loss_scaling or "fft_p_max" in loss_scaling:
            fft_losses_values = fft_losses(clean_signal, injected_noisy, injected_denoised)
            if "fft_m" in loss_scaling:
                metrics["fft_m"] = fft_losses_values[0].mean()
            if "fft_p" in loss_scaling:
                metrics["fft_p"] = fft_losses_values[1].mean()
            if "fft_m_max" in loss_scaling:
                metrics["fft_m_max"] = fft_losses_values[2].mean()
            if "fft_p_max" in loss_scaling:
                metrics["fft_p_max"] = fft_losses_values[3].mean()
        if "kl" in loss_scaling:
            metrics["kl"] = get_kl_divergence_lognorm(mean, logvar).mean()
        if "l2" in loss_scaling:
            metrics["l2"] = get_l2_loss(model_params)
        
        for key in loss_scaling.keys():
            metrics[key] *= scaled_weights[key]
        
        # Compute total loss
        metrics["loss"] = jnp.sum(jnp.array([value for _, value in metrics.items()]))
        
        return metrics
    
    # Define baseline weights to get losses on the same order of magnitude
    scaled_weights = {
        "wt": 17300,
        "t": 300000,
        "fft_m": 10,
        "fft_p": 150000,
        "fft_m_max": 0.00003,
        "fft_p_max": 0.02,
        "l2": 0.003,
        "kl": 0.000005,
        # "l2": 0.00002,
        # "kl": 0.000006,
    }
    
    # Extract data from example batch
    clean_signal, noisy_approx = example_batch
    
    # Compute example metrics using baseline weights
    noise_injection = create_noise_injection(wavelet, mode)
    
    
    with loss_scaling.unlocked():
        for key, value in loss_scaling.items():
            if value == 0:
                del scaled_weights[key]
                del loss_scaling[key]
        scale_agnostic_scaling = {key: value for key, value in loss_scaling.items() if key in SCALE_AGNOSTIC_LOSSES}
        for key in scale_agnostic_scaling:
            del loss_scaling[key]
        # Print scaled weights
        print_metrics(loss_scaling, pre_text="Loss values for completely random data: (beating these values is the bare minimum goal)\n")
        
        # Compute example metrics for each loss type
        example_metrics = compute_metrics(clean_signal, noisy_approx, noisy_approx, None, None, None)
        
        loss_scaling.update(scale_agnostic_scaling)
       
    
    # Update weights to be scaled
    print(f"loss_scaling: {loss_scaling}")
    for key in loss_scaling.keys():
        if key == "loss":
            continue
        
        if key in scaled_weights:
            scaled_weights[key] *= loss_scaling[key]
            if key not in SCALE_AGNOSTIC_LOSSES:
                scaled_weights[key] /= example_metrics[key]
        else:
            raise ValueError(f"Loss type '{key}' not found in scaled weights dictionary.")
    
    # JIT compile compute metrics function
    return jit(compute_metrics)

def print_metrics(metrics, pre_text=""):
    
    print_out = pre_text
    if 'loss' in metrics:
        print_out += f"loss: {metrics['loss']:.4f}, "
    for key in metrics.keys():
        if key != "loss":
            print_out += f"{key}: {metrics[key]:.4f}, "
    
    print(print_out)
    return
    