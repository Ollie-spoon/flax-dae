"""Default Hyperparameter configuration."""

import ml_collections
from jax.numpy import array

scale = 4

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # config.learning_rate = 0.001
    config.learning_rate_schedule=array([
        [500, 0.01*scale], 
        [2000, 0.001*scale], 
        [6000, 0.0001*scale], 
        [20000, 0.00005*scale],
    ])
    config.latents = 40
    config.hidden = 150
    config.dropout_rate = 0.2
    config.io_dim = 68
    config.batch_size = 1000
    config.epoch_size = 10000
    config.num_epochs = 15000
    config.data_path = 'C:/Users/omnic/OneDrive/Documents/MIT/Programming/approximation_coefficients_dataset.npy'
    config.checkpoint_restore_path = ""
    return config
