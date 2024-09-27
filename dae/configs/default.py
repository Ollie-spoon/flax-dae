"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.learning_rate = 0.0005
    config.latents = 30
    config.hidden = 125
    config.dropout_rate = 0.2
    config.io_dim = 68
    config.batch_size = 1000
    config.epoch_size = 10000
    config.num_epochs = 12000
    config.data_path = 'C:/Users/omnic/OneDrive/Documents/MIT/Programming/approximation_coefficients_dataset.npy'
    config.checkpoint_restore_path = ""
    return config
