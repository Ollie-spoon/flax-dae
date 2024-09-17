# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is created with reference to torchvision/utils.py.

Modify: torch.tensor -> jax.numpy.DeviceArray
If you want to know about this file in detail, please visit the original code:
    https://github.com/pytorch/vision/blob/master/torchvision/utils.py
"""
# import math

# import jax.numpy as jnp
# import numpy as np
# from PIL import Image
import matplotlib.pyplot as plt
import flax.serialization
import orbax.checkpoint as ocp
import os



def plot_inverse_loss(loss_components, save_path):
    """
    Plot the inverse of each loss component over time and save the plot to a file.

    Args:
        loss_components (dict): A dictionary containing the loss components as keys and their values over time as lists.
        save_path (str): The path to save the plot file.
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the inverse of each loss component
    for loss_name, loss_values in loss_components.items():
        ax.plot(1/loss_values, label=loss_name)

    # Set the x-axis label
    ax.set_xlabel('Time')

    # Set the y-axis label
    ax.set_ylabel('Accuracy (Inverse of the Loss)')

    # Add a legend
    ax.legend()

    # Save the plot to a file
    plt.savefig(save_path)
    
    

import pickle

def save_model(state, step: int, ckpt_dir: str = 'checkpoints/'):
    """Save model parameters and optimizer state using pickle."""
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    checkpoint = {
        'params': state.params,
        'opt_state': state.opt_state,
    }

    save_path = os.path.join(ckpt_dir, f'checkpoint_{step}.pkl')
    
    with open(save_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved at step {step} in {save_path}")

def convert_to_path(string):
    if isinstance(string, str):
        return string.replace('/', '\\')
    raise ValueError("Input must be a string.")

# def load_model(state, step: int, ckpt_dir: str):
#     """Load model parameters and optimizer state."""
#     checkpoint = orbax.checkpoint.restore_checkpoint(ckpt_dir, step)
#     state = state.replace(
#         params=flax.serialization.from_bytes(state.params, checkpoint['params']),
#         opt_state=flax.serialization.from_bytes(state.opt_state, checkpoint['opt_state']),
#     )
#     print(f"Checkpoint restored from step {step}")
#     return state
