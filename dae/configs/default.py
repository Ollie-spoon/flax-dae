"""Default Hyperparameter configuration."""

import ml_collections
from jax.numpy import array
from jax.numpy import float32

scale = 1

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # config.learning_rate = 0.001
    config.learning_rate_schedule=array([
        
        # [1, 0.000001],
        # [5, 0.0000005],
        # [9, 0.00002],
        # [13, 0.00001],
        # [17, 0.000005],
        # [21, 0.000002],
        
        
        ## All of the below lr are for adam and are waaaay too high for sgd
        [1, 0.0001],
        # [5, 0.005],
        # [9, 0.002],
        # [13, 0.001],
        # [17, 0.0005],
        # [21, 0.0002],
        
        
        # [19, 0.02],
        # [33, 0.01],
        # [57, 0.005],
        # [91, 0.002],
        
        # loss{0:0}: 1.5400043532630622e+19
        # loss{0:1}: nan
        
        # [1, 0.0004],
        # [10, 0.0003], 
        # [30, 0.0002], 
        # [50, 0.0001], 
        # [60, 0.00009], 
        # [70, 0.00008], 
        # [80, 0.00007], 
        # [90, 0.00006], 
        # [100, 0.00005],
        # [150, 0.000035],
        # [200, 0.00002], 
        # [300, 0.00001], 
        # [400, 0.000005], 
        # [600, 0.000001], 
        
        # [5000, 0.0001],
        # [20000, 0.00004],
        # [1000, 0.0005*scale],
        # [2000, 0.0001*scale], 
        # [6000, 0.00008*scale], 
        # [10000, 0.00004*scale], 
        # [20000, 0.00002*scale],
    ])
    config.loss_scaling = {
        "wt": 0.0,
        "t": 1.0,
        "fft_m": 1.0,
        "fft_p": 1.0,
        "fft_m_max": 0.0,
        "fft_p_max": 0.0,
        "fft_m_struct": 1.0,
        "l2": 0.005,
        "kl": 0.0,
        "output_std": 0.0,
    }
    config.data_args = {
        "params": {
            "a_min": 0,
            "a_max": 1,
            "tau_min": 10,
            "tau_max": 200,
            "decay_count": 2,
        },
        "t_max": 400, 
        "t_len": 676, 
        "SNR": 100,
        "wavelet": "db12", 
        "mode": "constant",
        "dtype": float32,
    }
    config.latents = 4
    config.hidden = 10
    config.dropout_rate = 0.2
    config.io_dim = config.data_args["t_len"]
    config.batch_size = 1250
    config.epoch_size = 12500
    config.num_epochs = 5000
    config.cycles_per_epoch = 5
    config.data_path = 'C:/Users/omnic/OneDrive/Documents/MIT/Programming/approximation_coefficients_dataset.npy'
    config.checkpoint_restore_path = ""
    config.working_dir = ""
    return config
