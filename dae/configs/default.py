"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.learning_rate = 0.00025
    config.latents = 35
    config.hidden = 120
    config.dropout_rate = 0.2
    config.io_dim = 95
    config.batch_size = 500
    config.epoch_size = 10000
    config.num_epochs = 80
    config.data_path = 'C:/Users/omnic/OneDrive/Documents/MIT/Programming/approximation_coefficients_dataset.npy'
    config.checkpoint_restore_path = ""
    return config
