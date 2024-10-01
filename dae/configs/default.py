"""Default Hyperparameter configuration."""

import ml_collections
from jax.numpy import array

scale = 1

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # config.learning_rate = 0.001
    config.learning_rate_schedule=array([
        [500, 0.001], 
        [1000, 0.0005*scale],
        [2000, 0.0001*scale], 
        [6000, 0.00008*scale], 
        [10000, 0.00004*scale], 
        [20000, 0.00002*scale],
    ])
    config.loss_scaling = {
        "wt": 0.01,
        "t": 1.0,
        "fft_m": 1.0,
        "fft_p": 0.1,
        "fft_m_max": 0.0,
        "fft_p_max": 0.0,
        "l2": 1.0,
        "kl": 0.0,
    }
    config.latents = 35
    config.hidden = 150
    config.dropout_rate = 0.2
    config.io_dim = 68
    config.batch_size = 2500
    config.epoch_size = 20000
    config.num_epochs = 20000
    config.data_path = 'C:/Users/omnic/OneDrive/Documents/MIT/Programming/approximation_coefficients_dataset.npy'
    config.checkpoint_restore_path = ""
    return config
