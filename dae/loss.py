from jax import jit, vmap, tree_util
import jax.numpy as jnp
from jax.scipy.special import erfc
from cr.wavelets import wavedec, waverec

@jit
def get_mse_loss(recon_x, noiseless_x, scale=159419):
    return jnp.mean(jnp.square(recon_x - noiseless_x)) * scale

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
def get_mae_loss(recon_x, noiseless_x):
    return jnp.mean(jnp.abs(recon_x - noiseless_x))

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

@vmap
@jit
def get_log_mse_loss(recon_x, noiseless_x, eps=1e-16):
    return jnp.square(abs_complex_log10(recon_x+eps) - abs_complex_log10(noiseless_x+eps)).mean()

@vmap
@jit
def get_max_loss(recon_x, noiseless_x, scale=72.51828996):
    return jnp.max(jnp.abs(recon_x - noiseless_x))*scale

@jit
def get_l2_loss(params, alpha=0.0000001):
    l2_loss = tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), params)
    return alpha * sum(tree_util.tree_leaves(l2_loss))

@vmap
@jit
def get_custom_loss(recon_diff, original_diff, alpha=0.002):
    return alpha * jnp.max([0, 1 - recon_diff/original_diff]).mean()

@vmap
@jit
def get_kl_divergence_lognorm(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@vmap
@jit
def get_kl_divergence_truncated_normal(mean, logvar):
    # a and b are the lower and upper bounds of the truncated normal distribution
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar)) + \
           jnp.log(erfc(-mean/jnp.sqrt(jnp.exp(logvar))))


# Combine the loss functions into a single value
def compute_metrics(recon_diff, noiseless_x, noisy_x, model_params):
        
    metrics = {}
    
    recon_x = noisy_x+recon_diff 
    
    metrics["mse"] = get_mse_loss(recon_x, noiseless_x).mean()
    metrics["max"] = get_max_loss(recon_x, noiseless_x).mean()
    
    metrics["l2"] = get_l2_loss(model_params)
    
    metrics["loss"] = jnp.sum(value for _, value in metrics)
    
    return metrics

def create_noise_injection(wavelet, mode):
    @vmap
    @jit
    def noise_injection(recon_x, clean_signal):
        
        # forward wavelet transform
        clean_coeffs = wavedec(clean_signal, wavelet, mode)
        
        # Noise injection
        clean_coeffs[0] = recon_x
        
        # inverse wavelet transform
        injected_denoised = waverec(clean_coeffs, wavelet, mode)
        
        return injected_denoised
    return noise_injection


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
        
        metrics["mse_t"] = get_mse_loss(injected_denoised, clean_signal, scale=159419).mean()
        metrics["mse_wt"] = get_mse_loss(recon_approx, noisy_approx, scale=8919).mean()
        
        # metrics["kl"] = get_kl_divergence_lognorm(mean, logvar).mean()
        # metrics["kl"] = get_kl_divergence_truncated_normal(mean, logvar).mean()
        
        # metrics["max"] = get_max_loss(injected_denoised, clean_signal).mean()
        
        # metrics["l2"] = get_l2_loss(model_params)
        
        # for key, value in metrics.items():
        #     # print(f"{key}: {value}")
        #     print(f"type of {key}: {type(value)}")
        
        metrics["loss"] = jnp.sum(jnp.array([value for _, value in metrics.items()]))
        
        return metrics
    return compute_metrics
