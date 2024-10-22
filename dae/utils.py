# import math

# import numpy as np
# from PIL import Image
# import flax.serialization
# import orbax.checkpoint as ocp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
# from jax import devices
from cr.wavelets import dwt_max_level
from pywt import Wavelet
import pickle
import data_processing
# import GPUtil
# import psutil

def get_approx_length(signal_length, wavelet):
    if type(wavelet) == str:
        wavelet = Wavelet(wavelet)
    max_level = dwt_max_level(signal_length, wavelet.dec_len)
    return dwt_coeff_len(signal_length, wavelet, level=max_level), max_level

def dwt_coeff_len(signal_length, wavelet, level=1):
    if level == 0:
        return int(signal_length)
    signal_length = jnp.floor((signal_length + wavelet.dec_len - 1) / 2)
    return dwt_coeff_len(signal_length, wavelet, level-1)

def plot_comparison(comparison, epoch_number, save_location):
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    
    noiseless_list = comparison[0]
    recon_list = comparison[1]
    
    for i in range(3):
        noiseless = noiseless_list[i]
        recon = recon_list[i]
        
        # print(f"noiseless: {noiseless.shape}")
        # print(f"recon: {recon.shape}")
        
        # Plot reconstructed data overlaid over noiseless data
        axes[i, 0].plot(noiseless, label='Noiseless Data')
        axes[i, 0].plot(recon, label='Reconstructed Data')
        axes[i, 0].set_title(f'Noiseless vs Reconstructed Data (Pair {i+1})')
        axes[i, 0].legend()
        
        # Plot difference and absolute log difference
        difference = noiseless - recon
        abs_log_diff = jnp.log10(jnp.abs(difference) + 1e-10)  # Adding a small value to avoid log(0)
        
        axes[i, 1].plot(difference, label='Difference')
        axes[i, 1].plot(abs_log_diff, label='Absolute Log Difference')
        axes[i, 1].set_title(f'Difference and Absolute Log Difference (Pair {i+1})')
        axes[i, 1].legend()
    
    plt.suptitle(f'Comparison Plots for Epoch {epoch_number}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_location)
    plt.close()
    return

def plot_inverse_loss(metric_list, save_path):
    """
    Plot the inverse of each loss component over time and save the plot to a file.

    Args:
        loss_components (dict): A dictionary containing the loss components as keys and their values over time as lists.
        save_path (str): The path to save the plot file.
    """
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    keys = metric_list[0].keys()

    # Plot the inverse of each loss component
    for key in keys:
        if key != "l2" and key != "kl" and key != "mse_wt":
            loss_values = jnp.array([metric[key] for metric in metric_list])
            ax.plot(jnp.log(1/loss_values), label=key)

    # Set the x-axis label
    ax.set_xlabel('Time')
    # ax.set_xscale('log')

    # Set the y-axis label
    ax.set_ylabel('Accuracy (Inverse of the Loss)')
    # ax.set_yscale('log')

    # Add a legend
    ax.legend()

    # Save the plot to a file
    plt.savefig(save_path)
    plt.close()
    
    return
    
    


def save_model(state, step: int, ckpt_dir: str, model_args: dict, logging=True):
    """Save model parameters, optimizer state, and model arguments using pickle."""
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Create the checkpoint data, including model arguments
    checkpoint = {
        'params': state.params,
        'opt_state': state.opt_state,
        'model_args': model_args,
        'step': state.step,
    }

    save_path = os.path.join(ckpt_dir, f'checkpoint_{step}.pkl')
    
    # Convert the path if needed
    save_path = __convert_to_path(save_path)
    
    # Save the checkpoint as a pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    if logging:
        print(f"Checkpoint saved at step {step} in {save_path}")
    
def load_model(ckpt_path: str):
    """Load model parameters, optimizer state, and model arguments from a pickle checkpoint."""
    load_path = os.path.join(ckpt_path)
    
    # Convert the path if needed
    load_path = __convert_to_path(load_path)
    
    # Load the checkpoint data
    with open(load_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Extract model parameters, optimizer state, and model arguments
    params = checkpoint['params']
    opt_state = checkpoint['opt_state']
    model_args = checkpoint['model_args']
    step = checkpoint['step']
    
    print(f"Checkpoint loaded from {load_path}")
    return params, opt_state, model_args, step

def __convert_to_path(string):
    if isinstance(string, str):
        return string.replace('\\', '/')
    raise ValueError("Input must be a string.")

def print_comparison(comparison):
    true_params, noise_powers, prediction = comparison
    
    amps_true = true_params[:, 0::2]
    taus_true = true_params[:, 1::2]
    
    noise_power_predicted = prediction[:, -1]
    prediction = prediction[:, :-1]
    taus_predicted = prediction[:, 0::2]
    amps_predicted = prediction[:, 1::2]
    
    taus_predicted, amps_predicted, noise_power_predicted = data_processing.unnormalize_exp_params(taus_predicted, amps_predicted, noise_power_predicted)
    
    for i in range(len(true_params)):
        print(f"Parameter comparison {i+1} " + "\{true\}:\{predicted\}:")
        print(f"\tA1 {amps_true[i, 0]:.4f} {amps_predicted[i, 0]:.4f}")
        print(f"\tT1 {taus_true[i, 0]:.4f} {taus_predicted[i, 0]:.4f}")
        print(f"\tA2 {amps_true[i, 1]:.4f} {amps_predicted[i, 1]:.4f}")
        print(f"\tT2 {taus_true[i, 1]:.4f} {taus_predicted[i, 1]:.4f}")
        print(f"\tNoise Power {noise_powers[i]:.4f} {noise_power_predicted[i]:.4f}")
    
    return
        


# def __get_memory_size():
#     if devices()[0].device_kind == 'gpu':
#         # Get GPU memory size
#         gpus = GPUtil.getGPUs()
#         if gpus:
#             gpu = gpus[0]
#             return gpu.memoryTotal * 1024 ** 2 # This is in Bytes
#         else:
#             return ValueError("Jax identified a GPU, but none was found.")
#     else:
#         # Get CPU memory size
#         memory_info = psutil.virtual_memory()
#         return memory_info.total # This is in Bytes
    
# def __calculate_memory_usage(length, dtype):
#     if dtype == jnp.float32:
#         bytes_per_element = 4
#     elif dtype == jnp.float64:
#         bytes_per_element = 8
#     else:
#         raise ValueError("Unsupported data type")
    
#     total_bytes = length * bytes_per_element
#     return total_bytes

# def calculate_batch_size(io_length, dtype, data_points_per_epoch):
#     memory_usage_per_data_point = __calculate_memory_usage(io_length, dtype)
#     print(f"\tmemory_usage_per_data_point: {memory_usage_per_data_point}")
#     allocated_memory = __get_memory_size()
#     print(f"\tallocated_memory: {allocated_memory}")
#     max_batch_size = allocated_memory // memory_usage_per_data_point
#     print(f"\tmax_batch_size: {max_batch_size}")
    
#     if data_points_per_epoch < max_batch_size:
#         return 1, data_points_per_epoch
#     for batches in range(1, max_batch_size+1):
#         if data_points_per_epoch/batches < max_batch_size:
#             break
#     if data_points_per_epoch % batches == 0:
#         return batches, data_points_per_epoch
#     for actual_points_per_epoch in range(1, data_points_per_epoch)[::-1]:
#         if actual_points_per_epoch % batches == 0:
#             return batches, actual_points_per_epoch
    
#     return ValueError("Batch size could not be calculated.")




# def load_model(state, step: int, ckpt_dir: str):
#     """Load model parameters and optimizer state."""
#     checkpoint = orbax.checkpoint.restore_checkpoint(ckpt_dir, step)
#     state = state.replace(
#         params=flax.serialization.from_bytes(state.params, checkpoint['params']),
#         opt_state=flax.serialization.from_bytes(state.opt_state, checkpoint['opt_state']),
#     )
#     print(f"Checkpoint restored from step {step}")
#     return state