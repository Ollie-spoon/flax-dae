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
from flax.training import train_state
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import optax
from time import time


@jax.vmap
def mse(recon_x, noiseless_x):
  return jnp.mean(jnp.square(recon_x - noiseless_x))

@jax.vmap
def max_diff(recon_x, noiseless_x):
  return jnp.max(recon_x - noiseless_x)

# Combine the loss functions into a single value
def compute_metrics(recon_x, noiseless_x):
  mse_loss = mse(recon_x, noiseless_x).mean()
  max_loss = max_diff(recon_x, noiseless_x).mean()
  loss = mse_loss + max_loss
  return {'mse': mse_loss, 'max': max_loss, 'loss': loss}


# Define the training step
def train_step(state, batch, model_args, dropout_rng):
    noisy_data, noiseless_data = batch
    
    def loss_fn(params):
        recon_x = models.model(*model_args).apply(
            {'params': params},
            x=noisy_data,
            deterministic=False,
            rngs={'dropout': dropout_rng},
        )

        mse_loss = mse(recon_x, noiseless_data).mean()
        loss = mse_loss
        return loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


# Define the evaluation function
def eval_f(params, batch, model_args):
    noisy_data, noiseless_data = batch
    
    def eval_model(vae):
        recon_x = vae(noisy_data, deterministic=True)
        
        # Would be good to have but implement later
        # comparison = jnp.concatenate([
        #     images[:8].reshape(-1, 28, 28, 1),
        #     recon_images[:8].reshape(-1, 28, 28, 1),
        # ])
        
        metrics = compute_metrics(recon_x, noiseless_data)
        return metrics# , comparison

    return nn.apply(eval_model, models.model(*model_args))({'params': params})


def train_and_evaluate(config: ml_collections.ConfigDict):
    """Train and evaulate pipeline."""
    # Set up the random number generators
    # rng is the random number generator and therefore never passed to the model
    rng = random.key(0)
    rng, init_rng = random.split(rng)
    
    model_args = (
        config.latents, 
        config.hidden, 
        config.dropout_rate, 
        config.io_dim
        )
    
    # Load the data
    train_ds, steps_per_epoch = input_pipeline.build_train_set(config.data_path, config.batch_size)
    test_ds, _ = input_pipeline.build_test_set(config.data_path, config.batch_size)
    
    # Initialize the model with some dummy data
    logging.info('Initializing model.')
    init_data = jnp.ones((config.batch_size, config.io_dim), jnp.float32)
    params = models.model(*model_args).init(init_rng, init_data)['params']

    # Initialize the training state including the optimizer
    state = train_state.TrainState.create(
        apply_fn=models.model(*model_args).apply,
        params=params,
        tx=optax.adam(config.learning_rate),
    )

    # Train the model
    test_batch = next(test_ds)
    for epoch in range(config.num_epochs):

        start_time = time()
        # Train the model for one epoch
        for _ in range(steps_per_epoch):
            batch = next(train_ds)
            rng, dropout_rng = random.split(rng)
            state = train_step(state, batch, model_args, dropout_rng)

        # Evaluate the model
        metrics = eval_f(state.params, test_batch, model_args)

        # Print the evaluation metrics
        print(
            'eval epoch: {}, time {:.2f}s, loss: {:.6f}, MSE: {:.8f}, max: {:.6f}'.format(
                epoch + 1, time()-start_time, metrics['loss'], metrics['mse'], metrics['max']
            )
        )
        
        # Save the model
        if (epoch + 1) % 1 == 0:
            utils.save_model(state, epoch + 1, 'flax/tmp/checkpoints')

            
    # Save the results
    utils.plot_inverse_loss(metrics, "loss.png")
