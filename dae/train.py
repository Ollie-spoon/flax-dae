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
"""Training and evaluation logic."""

from absl import logging
from flax import linen as nn
import input_pipeline
import models
import utils
import data_processing
from flax.training import train_state
import jax
jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import ml_collections
import optax
from time import time
from numpy import float64 as npfloat64


@jax.jit
def abs_complex_log10(numbers):
    # Compute the complex logarithm
    absolutes = jnp.abs(numbers)  # Adding 0j to ensure complex type
    negative = (1-numbers/absolutes)
    # Return the absolute value of the complex logarithm
    return jnp.log10(absolutes)+negative

@jax.vmap
def get_mse_loss(recon_x, noiseless_x):
    return jnp.mean(jnp.square(recon_x - noiseless_x))

@jax.vmap
def get_mae_loss(recon_x, noiseless_x):
    return jnp.mean(jnp.abs(recon_x - noiseless_x))

@jax.jit
def huber_loss(recon_x, noiseless_x, delta=1.0):
    diff = recon_x - noiseless_x
    abs_diff = jnp.abs(diff)
    quadratic = jnp.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    return 0.5 * quadratic**2 + delta * linear

@jax.vmap
def get_huber_loss(recon_x, noiseless_x, delta=1.0):
    return jnp.mean(huber_loss(recon_x, noiseless_x, delta))

@jax.vmap
def get_log_mse_loss(recon_x, noiseless_x, eps=1e-16):
    return jnp.square(abs_complex_log10(recon_x+eps) - abs_complex_log10(noiseless_x+eps)).mean()

@jax.vmap
def get_max_loss(recon_x, noiseless_x):
    return jnp.max(jnp.abs(recon_x - noiseless_x))

@jax.jit
def get_l2_loss(params, alpha=0.002):
    l2_loss = jax.tree_util.tree_map(lambda x: jnp.mean(jnp.square(x)), params)
    return alpha * sum(jax.tree_util.tree_leaves(l2_loss))

@jax.jit
def get_custom_loss(recon_diff, original_diff, alpha=0.002):
    return alpha * jnp.max([0, 1 - recon_diff/original_diff]).mean()

# Combine the loss functions into a single value
def compute_metrics(recon_diff, noiseless_x, noisy_x, model_params):
    original_diff = noiseless_x - noisy_x 
    mse_loss = get_mse_loss(recon_diff, original_diff).mean()
    # mae_loss = get_mae_loss(recon_x, noiseless_x).mean()
    # huber_loss = get_huber_loss(recon_x, noiseless_x).mean()
    # print(f"mse_loss: {mse_loss}")
    
    # handle nan values
    # log_mse_loss = get_log_mse_loss(recon_x, noiseless_x).mean()
    
    max_loss = get_max_loss(noisy_x+recon_diff, noiseless_x).mean()
    l2_loss = get_l2_loss(model_params)
    # custom_loss = get_custom_loss(recon_diff, original_diff)
    
    # jax.debug.print("log_mse_loss: {}", log_mse_loss)
    
    loss = mse_loss + l2_loss + max_loss 
    
    return {
        'mse': mse_loss, 
        # 'mae': mae_loss, 
        # 'huber': huber_loss, 
        # 'log_mse': log_mse_loss, 
        'max': max_loss, 
        # 'custom': custom_loss, 
        'l2': l2_loss, 
        'loss': loss
    }


# Define the training step
def train_step(state, batch, model_args, dropout_rng):
    noisy_data, noiseless_data = batch
    difference_data = noiseless_data - noisy_data
    
    def loss_fn(params):
        difference_prediction = models.model(**model_args).apply(
            {'params': params},
            x=noisy_data,
            deterministic=False,
            rngs={'dropout': dropout_rng},
        )

        loss = compute_metrics(difference_prediction, noiseless_data, noisy_data, state.params)['loss']
        return loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


# Define the evaluation function
def eval_f(params, batch, model_args):
    noisy_data, noiseless_data = batch
    difference_data = noiseless_data - noisy_data
    
    # print(f"jnp.shape(noisy_data): {jnp.shape(noisy_data)}")
    # print(f"jnp.shape(noiseless_data): {jnp.shape(noiseless_data)}")
    
    def eval_model(vae):
        difference_prediction = vae(noisy_data, deterministic=True)
        # recon_x = data_processing.add_difference(noisy_data, recon_x)
        
        # Why is this in the eval_model function?
        comparison = jnp.array([noiseless_data[:3], noisy_data[:3] + difference_prediction[:3]])
        
        metrics = compute_metrics(difference_prediction, noiseless_data, noisy_data, params)
        return metrics, comparison

    return nn.apply(eval_model, models.model(**model_args))({'params': params})


def train_and_evaluate(config: ml_collections.ConfigDict, working_dir: str):
    """Train and evaulate pipeline."""
    # Set up the random number generators
    # rng is the random number generator and therefore never passed to the model
    rng = random.key(0)
    rng, init_rng = random.split(rng)
    
    # Define the test data parameters
    data_args = {
        "params": {
            "a1": 0.6, 
            "a2": 0.4, 
            "tau1_min": 5, 
            "tau1_max": 30, 
            "tau2_min": 20, 
            "tau2_max": 45,
        },
        "t_max": 100, 
        "t_len": 1000, 
        "SNR": 100, 
        "wavelet": "coif6", 
        "dtype": npfloat64,
    }
    
    # Generate an example test data point to extract the input/output dimensions from
    data_point_example = next(input_pipeline.generate_original_data(
        iterations=1, 
        train=False,
        batch_size=1,
        kwargs=data_args,
    ))
    io_dim = len(data_point_example[0][0])
    print(f"data_point_example[0].shape: {io_dim}")
    
    # # Calculate the batch size based on the available memory and the maximum epoch size
    # batches, config.epoch_size = utils.calculate_batch_size(
    #     io_dim, 
    #     type(data_point_example[0][0]), 
    #     config.epoch_size,
    # )
    # del data_point_example
    # batch_size = config.epoch_size // batches
    
    
    batch_size = 500
    config.epoch_size = 9500
    
    print(f"~~batch_size: {batch_size}")
    print(f"~~config.epoch_size: {config.epoch_size}")
    
    model_args = {
        "latents": config.latents,
        "hidden": config.hidden,
        "dropout_rate": config.dropout_rate, 
        "io_dim": io_dim
    }
    
    # Initialize the model and the training state
    if config.checkpoint_restore_path is not None:
        params, opt_state, model_args = utils.load_model(working_dir + config.checkpoint_restore_path)

        # Restore the train state
        state = train_state.TrainState(
            step=opt_state['step'],  # Restore the step count from opt_state
            apply_fn=models.model(**model_args).apply,
            params=params,
            tx=optax.adam(config.learning_rate),
            opt_state=opt_state['opt_state'],  # Set the optimizer state
        )
    else:
        # Initialize the model with some dummy data
        logging.info('Initializing model.')
        init_data = jnp.ones((batch_size, io_dim), data_args["dtype"])
        params = models.model(**model_args).init(init_rng, init_data)['params']

        # Initialize the training state including the optimizer
        state = train_state.TrainState.create(
            apply_fn=models.model(**model_args).apply,
            params=params,
            tx=optax.adam(config.learning_rate),
        )

    metric_list = []
    
    # Train the model
    for epoch in range(config.num_epochs):
        
        start_time = time()
        
        # Create a training data set   
        train_ds = input_pipeline.generate_original_data(
            train=True,
            iterations=config.epoch_size,
            batch_size=batch_size,
            kwargs=data_args,)
        
        # print(f"train_ds.shape: {train_ds.shape}")
        
        # Create a test data set 
        test_batch = next(input_pipeline.generate_original_data(
            train=False,
            iterations=batch_size,
            batch_size=batch_size,
            kwargs=data_args,
            ))
        
        # print(f"test_batch.shape: {len(test_batch)}")
        # print(f"test_batch[0].shape: {test_batch[0].shape}")
        # print(f"test_batch[1].shape: {test_batch[1].shape}")
        
        # Train the model for one epoch
        for batch in train_ds:
            rng, dropout_rng = random.split(rng)
            state = train_step(state, batch, model_args, dropout_rng)

        # Evaluate the model
        metrics, comparison = eval_f(state.params, test_batch, model_args)
        # print(f"metrics: {metrics}")
        # print(f"comparison: {comparison}")
        metric_list.append(metrics)

        # Print the evaluation metrics
        print(
            f"eval epoch: {epoch + 1}, "
            f"time {time()-start_time:.2f}s, "
            f"loss: {metrics['loss']:.7f}, "
            f"mse: {metrics['mse']:.8f}, "
            # f"mae: {metrics['mae']:.8f}, "
            f"max: {metrics['max']:.8f}, "
            # f"huber: {metrics['huber']:.8f}, "
            # f"log_mse: {metrics['log_mse']:.8f}, "
            f"l2: {metrics['l2']:.8f}"
        )
        
        # Save the model
        if (epoch + 1) % 20 == 0:
            utils.save_model(state, epoch + 1, working_dir + 'tmp/checkpoints', model_args)
            utils.plot_comparison(comparison, epoch+1, working_dir + 'tmp/checkpoints/reconstruction_{}.png'.format(epoch+1))
            
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