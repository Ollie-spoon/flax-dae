"""Training and evaluation logic."""

from absl import logging
import jax
jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import ml_collections
from time import time
from numpy import float64 as npfloat64

import optax
import models
import utils
import data_processing
import input_pipeline
import loss

# Define the training step
def create_train_step(model_args, data_args):
    
    get_metrics = loss.create_compute_metrics(data_args["wavelet"], data_args["mode"])

    @jax.jit
    def train_step(state, batch, z_rng, dropout_rng):
        noisy_approx, clean_signal = batch
        
        def loss_fn(params):
            prediction, mean, logvar = models.model(
                hidden=model_args["hidden"],
                latents=model_args["latents"],
                dropout_rate=model_args["dropout_rate"],
                io_dim=model_args["io_dim"],
            ).apply(
                {'params': params},
                x=noisy_approx,
                z_rng=z_rng,
                deterministic=False,
                rngs={'dropout': dropout_rng},
            )

            # loss = compute_metrics(difference_prediction, noiseless_data, noisy_approx, state.params)['loss']
            loss = get_metrics(prediction, noisy_approx, mean, logvar, clean_signal, params)["loss"]
            return loss

        grads = jax.grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads)
    return train_step


# Define the evaluation function
def create_eval_f(model_args, data_args):
    
    get_metrics = loss.create_compute_metrics(data_args["wavelet"], data_args["mode"])
    
    @jax.jit
    def eval_f(params, batch, z_rng):
        noisy_approx, clean_signal = batch
        
        def eval_model(vae):
            z_rng, 
            prediction, mean, logvar = vae(noisy_approx, z_rng, deterministic=True)
            
            # This could be reintroduced at some point
            # comparison = jnp.array([noiseless_data[:3], noisy_data[:3] + difference_prediction[:3]])
            
            # metrics = compute_metrics(difference_prediction, noiseless_data, noisy_data, params)
            metrics = get_metrics(prediction, noisy_approx, mean, logvar, clean_signal, params)
            return metrics#, comparison

        return nn.apply(eval_model, models.model(
            hidden=model_args["hidden"],
            latents=model_args["latents"],
            dropout_rate=model_args["dropout_rate"],
            io_dim=model_args["io_dim"],
        ))({'params': params})
    return eval_f


def train_and_evaluate(config: ml_collections.ConfigDict, working_dir: str):
    """Train and evaulate pipeline."""
    # Set up the random number generators
    # rng is the random number generator and therefore never passed to the model
    time_keeping = time()
    rng = random.key(2024)
    rng, init_rng, z_rng, io_rng = random.split(rng, 4)
    
    # Define the test data parameters
    data_args = {
        "params": {
            "a1": 0.6, 
            "a2": 0.4, 
            "tau1_min": 20, 
            "tau1_max": 120, 
            "tau2_min": 80, 
            "tau2_max": 180,
        },
        "t_max": 400, 
        "t_len": 1120, 
        "SNR": 100,
        "wavelet": "coif6", 
        "mode": "zero",
        "dtype": jnp.float32,
    }
    
    # Generate extract the input/output dimensions and maximum number of 
    # dwt transforms allowed from this length signal
    io_dim, max_dwt_level = utils.get_approx_length(data_args["t_len"], data_args["wavelet"])
    logging.info(f"io_dim: {io_dim}")
    if io_dim != config.io_dim:
        logging.info(f"Warning: io_dim ({io_dim}) does not match the data dimension requested in config flags ({config.io_dim})")
    
    data_args["max_dwt_level"] = max_dwt_level
    
    # Initialize the model and the training state
    if config.checkpoint_restore_path != '':
        # Restore the model and the optimizer state
        params, opt_state, model_args = utils.load_model(working_dir + config.checkpoint_restore_path)

        # Restore the train state
        state = train_state.TrainState(
            step=100*10000,  # Restore the step count from opt_state
            apply_fn=models.model(**model_args).apply,
            params=params,
            tx=optax.adam(config.learning_rate),
            opt_state=opt_state,  # Set the optimizer state
        )
        
        original_state = state
    else:
        model_args = {
            "latents": config.latents,
            "hidden": config.hidden,
            "dropout_rate": config.dropout_rate, 
            "io_dim": io_dim
        }
        
        # Initialize the model with some dummy data
        logging.info('Initializing model.')
        init_data = jnp.ones((config.batch_size, io_dim), data_args["dtype"])
        params = models.model(**model_args).init(init_rng, init_data, z_rng)['params']

        # Initialize the training state including the optimizer
        state = train_state.TrainState.create(
            apply_fn=models.model(**model_args).apply,
            params=params,
            tx=optax.adam(config.learning_rate),
        )
        
    # Create the data generator
    train_step = create_train_step(model_args, data_args)
    eval_f = create_eval_f(model_args, data_args)
    data_generator = input_pipeline.create_data_generator(data_args)
    
    # loss_dict = {
    #     'mse': True, 
    #     'mae': False, 
    #     'huber': False, 
    #     'log_mse': False, 
    #     'max': True, 
    #     'custom': False, 
    #     'l2': True, 
    # }
    
    del init_rng, io_rng

    metric_list = []
    best_loss = jnp.inf
    
    logging.info(f"time taken to initialize: {time()-time_keeping:.3f}s")
    del time_keeping
    
    logging.info(
        f"This training instance has the following details:\n"
        f"configs: {config}\n"
        f"data_args: {data_args}\n"
    )
    
    # Train the model
    # SNR_shift = 10.0
    for epoch in range(config.num_epochs):
        # SNR_shift *= 1.1
        # if SNR_shift > 100:
        #     SNR_shift = 100
        # data_args["SNR"] = SNR_shift
        
        # data_generator = input_pipeline.create_data_generator(data_args)
        
        start_time = time()
        
        rng, train_rng, test_rng = random.split(rng, 3)
        # Create a training data set   
        train_ds = data_generator(
            key=train_rng, 
            n=config.epoch_size, 
            batch_size=config.batch_size
        )
        
        # Create a test data set 
        test_batch = next(data_generator(
            key=test_rng, 
            n=config.batch_size
        ))
        
        # print(f"time taken to generate data: {time()-start_time:.3f}s")
        # time_keeping = time()
        
        # Train the model for one epoch
        for batch in train_ds:
            rng, z_rng, dropout_rng = random.split(rng, 3)
            state = train_step(state, batch, z_rng, dropout_rng)
        
        # print(f"time taken to train: {time()-time_keeping:.3f}s")
        # time_keeping = time()

        # Evaluate the model
        rng, z_rng = random.split(rng)
        metrics = eval_f(state.params, test_batch, z_rng)
        # metrics, comparison = eval_f(state.params, test_batch, model_args)
        
        # print(f"time taken to evaluate: {time()-time_keeping:.3f}s")
        
        metric_list.append(metrics)

        # Print the evaluation metrics
        if (epoch + 1) % 10 == 0:
            loss.print_metrics(epoch, metrics, start_time)
        
        # Save the best model, assuming that it performs equally well on the validation set
        # if epoch > config.num_epochs/10 and metrics['loss'] < best_loss:
        if metrics['loss'] < best_loss:
            current_loss = metrics['loss']
            
            # Create a validation data set 
            rng, test_rng, z_rng = random.split(rng, 3)
            test_batch = next(data_generator(
                key=test_rng, 
                n=config.batch_size
            ))
            metrics = eval_f(state.params, test_batch, z_rng)
            
            current_loss = jnp.maximum(metrics['loss'], current_loss)
            
            if current_loss < best_loss:
                best_loss = current_loss
                loss.print_metrics(epoch, metrics, start_time, new_best=True)
                utils.save_model(state, 0, working_dir + 'tmp/checkpoints/best_this_run', model_args, logging=False)
        
        # Save the model
        if (epoch + 1) % 100 == 0:
            utils.save_model(state, epoch + 1, working_dir + 'tmp/checkpoints', model_args)
            state = original_state
            # utils.plot_comparison(comparison, epoch+1, working_dir + 'tmp/checkpoints/reconstruction_{}.png'.format(epoch+1))
            
    # Save the results
    utils.plot_inverse_loss(metric_list, working_dir + "dae/loss.png")


## Best so far:

# With the following configuration:
# config.learning_rate = 0.001
# config.latents = 25
# config.hidden = 180
# config.dropout_rate = 0.2
# config.io_dim = 95
# epoch_size = 9500
# batch_size = 500
# Loss = mse + l2 + max

# eval epoch: 116, time 3.58s, loss: 0.0260859, mse: 0.00009402, max: 0.02588823, l2: 0.00010363
# eval epoch: 117, time 3.42s, loss: 0.0262221, mse: 0.00009293, max: 0.02602558, l2: 0.00010357
# eval epoch: 118, time 3.41s, loss: 0.0257543, mse: 0.00008996, max: 0.02556078, l2: 0.00010354
# eval epoch: 119, time 3.38s, loss: 0.0260113, mse: 0.00009351, max: 0.02581425, l2: 0.00010355
# eval epoch: 120, time 3.41s, loss: 0.0260424, mse: 0.00009257, max: 0.02584633, l2: 0.00010350

# upping the number of latents to 30 we get 
# eval epoch: 161, time 3.74s, loss: 0.0257338, mse: 0.00009046, max: 0.02555016, l2: 0.00009321
# eval epoch: 162, time 3.40s, loss: 0.0293197, mse: 0.00014210, max: 0.02908473, l2: 0.00009292
# eval epoch: 163, time 3.44s, loss: 0.0268722, mse: 0.00010170, max: 0.02667855, l2: 0.00009196
# eval epoch: 164, time 3.40s, loss: 0.0259965, mse: 0.00009732, max: 0.02580743, l2: 0.00009175
# eval epoch: 165, time 3.41s, loss: 0.0262769, mse: 0.00009714, max: 0.02608802, l2: 0.00009174
# eval epoch: 166, time 3.44s, loss: 0.0260565, mse: 0.00009405, max: 0.02587068, l2: 0.00009179
# eval epoch: 167, time 3.51s, loss: 0.0258304, mse: 0.00009537, max: 0.02564319, l2: 0.00009183
# eval epoch: 168, time 3.43s, loss: 0.0261023, mse: 0.00009875, max: 0.02591164, l2: 0.00009188
# eval epoch: 169, time 3.41s, loss: 0.0258311, mse: 0.00009291, max: 0.02564630, l2: 0.00009188
# eval epoch: 170, time 3.41s, loss: 0.0255857, mse: 0.00009315, max: 0.02540063, l2: 0.00009194
# eval epoch: 171, time 3.40s, loss: 0.0254874, mse: 0.00009288, max: 0.02530248, l2: 0.00009199
# eval epoch: 172, time 3.41s, loss: 0.0255369, mse: 0.00008883, max: 0.02535603, l2: 0.00009204
# eval epoch: 173, time 3.43s, loss: 0.0255975, mse: 0.00008834, max: 0.02541708, l2: 0.00009207
# eval epoch: 174, time 3.38s, loss: 0.0254376, mse: 0.00009055, max: 0.02525494, l2: 0.00009213
# eval epoch: 175, time 3.43s, loss: 0.0268936, mse: 0.00010335, max: 0.02669931, l2: 0.00009097
# eval epoch: 176, time 3.43s, loss: 0.0267051, mse: 0.00010312, max: 0.02651152, l2: 0.00009051
# eval epoch: 177, time 3.41s, loss: 0.0264707, mse: 0.00010026, max: 0.02628004, l2: 0.00009044
# eval epoch: 178, time 3.40s, loss: 0.0268434, mse: 0.00010230, max: 0.02665064, l2: 0.00009047
# eval epoch: 179, time 3.39s, loss: 0.0266812, mse: 0.00010098, max: 0.02648971, l2: 0.00009050
# eval epoch: 180, time 3.44s, loss: 0.0261971, mse: 0.00009703, max: 0.02600959, l2: 0.00009053

# Decreasing the number of latents to 10 and hidden to 60 we get
# eval epoch: 161, time 3.25s, loss: 0.0264301, mse: 0.00009607, max: 0.02606259, l2: 0.00027142
# eval epoch: 162, time 3.37s, loss: 0.0268303, mse: 0.00010049, max: 0.02645847, l2: 0.00027136
# eval epoch: 163, time 3.26s, loss: 0.0267976, mse: 0.00010271, max: 0.02642385, l2: 0.00027101
# eval epoch: 164, time 3.15s, loss: 0.0268432, mse: 0.00010020, max: 0.02647219, l2: 0.00027084
# eval epoch: 165, time 3.15s, loss: 0.0265610, mse: 0.00009944, max: 0.02619058, l2: 0.00027099
# eval epoch: 166, time 3.18s, loss: 0.0264657, mse: 0.00009564, max: 0.02609894, l2: 0.00027107
# eval epoch: 167, time 3.25s, loss: 0.0264697, mse: 0.00009641, max: 0.02610208, l2: 0.00027120
# eval epoch: 168, time 3.50s, loss: 0.0263918, mse: 0.00009663, max: 0.02602379, l2: 0.00027133
# eval epoch: 169, time 3.52s, loss: 0.0264481, mse: 0.00009931, max: 0.02607737, l2: 0.00027145
# eval epoch: 170, time 3.56s, loss: 0.0269476, mse: 0.00010126, max: 0.02657473, l2: 0.00027157
# eval epoch: 171, time 3.46s, loss: 0.0267506, mse: 0.00009727, max: 0.02638186, l2: 0.00027142
# eval epoch: 172, time 3.52s, loss: 0.0263439, mse: 0.00009638, max: 0.02597621, l2: 0.00027136
# eval epoch: 173, time 3.40s, loss: 0.0268415, mse: 0.00009599, max: 0.02647406, l2: 0.00027144
# eval epoch: 174, time 3.19s, loss: 0.0262941, mse: 0.00009432, max: 0.02592816, l2: 0.00027157
# eval epoch: 175, time 3.30s, loss: 0.0265672, mse: 0.00009509, max: 0.02620063, l2: 0.00027152
# eval epoch: 176, time 3.16s, loss: 0.0261877, mse: 0.00009481, max: 0.02582143, l2: 0.00027148
# eval epoch: 177, time 3.25s, loss: 0.0260634, mse: 0.00009385, max: 0.02569806, l2: 0.00027154
# eval epoch: 178, time 3.18s, loss: 0.0263191, mse: 0.00009616, max: 0.02595128, l2: 0.00027163
# eval epoch: 179, time 3.28s, loss: 0.0271828, mse: 0.00010563, max: 0.02680554, l2: 0.00027159
# eval epoch: 180, time 3.22s, loss: 0.0265318, mse: 0.00009949, max: 0.02616112, l2: 0.00027121

# Initial 20 epochs with 0.002 learning rate with only 1 hidden layer 95:380:50
# eval epoch: 1, time 6.40s, loss: 0.0686198, mse: 0.0007613381, max: 0.06786, l2: 0.00000071
# eval epoch: 2, time 2.21s, loss: 0.0315773, mse: 0.0001410900, max: 0.03144, l2: 0.00000071
# eval epoch: 3, time 2.19s, loss: 0.0296443, mse: 0.0001283213, max: 0.02952, l2: 0.00000070
# eval epoch: 4, time 2.18s, loss: 0.0287801, mse: 0.0001200374, max: 0.02866, l2: 0.00000070
# eval epoch: 5, time 2.18s, loss: 0.0276918, mse: 0.0001172707, max: 0.02757, l2: 0.00000069
# eval epoch: 6, time 2.35s, loss: 0.0279019, mse: 0.0001155628, max: 0.02779, l2: 0.00000069
# eval epoch: 7, time 2.26s, loss: 0.0277906, mse: 0.0001135420, max: 0.02768, l2: 0.00000069
# eval epoch: 8, time 2.19s, loss: 0.0275679, mse: 0.0001127483, max: 0.02746, l2: 0.00000068
# eval epoch: 9, time 2.21s, loss: 0.0271378, mse: 0.0001120558, max: 0.02703, l2: 0.00000068
# eval epoch: 10, time 2.18s, loss: 0.0276028, mse: 0.0001147511, max: 0.02749, l2: 0.00000068
# eval epoch: 11, time 2.23s, loss: 0.0271092, mse: 0.0001085477, max: 0.02700, l2: 0.00000068
# eval epoch: 12, time 2.21s, loss: 0.0273182, mse: 0.0001106180, max: 0.02721, l2: 0.00000068
# eval epoch: 13, time 2.23s, loss: 0.0271339, mse: 0.0001108885, max: 0.02702, l2: 0.00000068
# eval epoch: 14, time 2.19s, loss: 0.0270930, mse: 0.0001076422, max: 0.02699, l2: 0.00000068
# eval epoch: 15, time 2.23s, loss: 0.0275015, mse: 0.0001079853, max: 0.02739, l2: 0.00000067
# eval epoch: 16, time 2.23s, loss: 0.0268886, mse: 0.0001065150, max: 0.02678, l2: 0.00000067
# eval epoch: 17, time 2.23s, loss: 0.0271248, mse: 0.0001074637, max: 0.02702, l2: 0.00000067
# eval epoch: 18, time 2.19s, loss: 0.0270350, mse: 0.0001092915, max: 0.02693, l2: 0.00000067
# eval epoch: 19, time 2.33s, loss: 0.0269516, mse: 0.0001038134, max: 0.02685, l2: 0.00000067
# eval epoch: 20, time 2.30s, loss: 0.0275775, mse: 0.0001118066, max: 0.02747, l2: 0.00000067

# Continued with a 10th of the learning rate

# eval epoch: 1, time 8.58s, loss: 0.0264469, mse: 0.0001010960, max: 0.02635, l2: 0.00000067
# eval epoch: 2, time 2.48s, loss: 0.0262269, mse: 0.0000987375, max: 0.02613, l2: 0.00000067
# eval epoch: 3, time 2.73s, loss: 0.0264717, mse: 0.0001022424, max: 0.02637, l2: 0.00000067
# eval epoch: 4, time 2.84s, loss: 0.0267135, mse: 0.0001000851, max: 0.02661, l2: 0.00000067
# eval epoch: 5, time 2.47s, loss: 0.0259943, mse: 0.0001003670, max: 0.02589, l2: 0.00000067
# eval epoch: 6, time 2.48s, loss: 0.0261110, mse: 0.0000983344, max: 0.02601, l2: 0.00000067
# eval epoch: 7, time 2.60s, loss: 0.0260025, mse: 0.0000977995, max: 0.02590, l2: 0.00000067
# eval epoch: 8, time 2.55s, loss: 0.0262142, mse: 0.0000978893, max: 0.02612, l2: 0.00000067
# eval epoch: 9, time 2.46s, loss: 0.0258707, mse: 0.0000981567, max: 0.02577, l2: 0.00000067
# eval epoch: 10, time 2.46s, loss: 0.0264305, mse: 0.0001006367, max: 0.02633, l2: 0.00000067
# eval epoch: 11, time 2.47s, loss: 0.0261005, mse: 0.0000964193, max: 0.02600, l2: 0.00000067
# eval epoch: 12, time 2.56s, loss: 0.0260889, mse: 0.0000985223, max: 0.02599, l2: 0.00000067
# eval epoch: 13, time 2.62s, loss: 0.0259972, mse: 0.0000968078, max: 0.02590, l2: 0.00000067
# eval epoch: 14, time 2.85s, loss: 0.0261861, mse: 0.0000972415, max: 0.02609, l2: 0.00000067
# eval epoch: 15, time 2.32s, loss: 0.0264188, mse: 0.0000976527, max: 0.02632, l2: 0.00000067
# eval epoch: 16, time 2.29s, loss: 0.0262123, mse: 0.0000977814, max: 0.02611, l2: 0.00000067
# eval epoch: 17, time 2.22s, loss: 0.0261200, mse: 0.0000970996, max: 0.02602, l2: 0.00000067
# eval epoch: 18, time 2.30s, loss: 0.0260109, mse: 0.0001013300, max: 0.02591, l2: 0.00000067
# eval epoch: 19, time 2.24s, loss: 0.0261549, mse: 0.0000968106, max: 0.02606, l2: 0.00000067
# eval epoch: 20, time 2.34s, loss: 0.0261016, mse: 0.0000974375, max: 0.02600, l2: 0.00000067

# at 0.00004 after 80 epochs we get to just dancing about the threshold from 0.00008 to 0.00007 so
# 0.00001: Essentially no change. individual shifts between epochs were large, but no meta-improvement
# 0.00002: 0.000080-0.000083
# 0.00004: 0.000079-0.000081
# 0.00008: 0.000074-0.000077
# 0.00016: 0.000068-0.000071
# 0.00032: 0.000068-0.000072
# 0.00064: immediately jumps out of the minimum but then starts learning really fast but keeps jumping out.
# 0.00128: jumps out, 