"""Training and evaluation logic."""

from absl import logging
from jax import random
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax.training.train_state import TrainState
import ml_collections
from time import time
from typing import Any

import optax
import models
import utils
import input_pipeline
import loss


class CustomTrainState(TrainState):
    batch_stats: Any

def create_learning_rate_scheduler(config):
    batches_per_epoch = config.epoch_size / config.batch_size
    schedule_array = config.learning_rate_schedule

    @jax.jit
    def learning_rate_schedule(step):
        # Compute the indices of the schedule array where the step is within the bounds
        idx = jnp.searchsorted(schedule_array[:, 0] * batches_per_epoch, step)

        # Handle the case where the step is beyond the last schedule point
        idx = jnp.clip(idx, 0, len(schedule_array) - 2)

        # Compute the learning rate using the formula
        lr = schedule_array[idx, 1] * (schedule_array[idx + 1, 1] / schedule_array[idx, 1]) ** (
            (step - schedule_array[idx, 0] * batches_per_epoch) / ((schedule_array[idx + 1, 0] - schedule_array[idx, 0]) * batches_per_epoch)
        )

        return lr

    return learning_rate_schedule

# Define the training step
def create_train_step(get_metrics, model_args):
    @jax.jit
    def train_step(state, batch, rng):
        clean_signal, noisy_approx = batch
        z_rng, dropout_rng = random.split(rng)

        def loss_fn(params, batch_stats):
            # Apply the model with mutable batch_stats
            (prediction, mean, logvar), new_state = models.model(
                hidden=model_args["hidden"],
                latents=model_args["latents"],
                dropout_rate=model_args["dropout_rate"],
                io_dim=model_args["io_dim"],
            ).apply(
                {'params': params, 'batch_stats': batch_stats},
                x=noisy_approx,
                z_rng=z_rng,
                deterministic=False,
                mutable=["batch_stats"],
                rngs={'dropout': dropout_rng},
            )

            loss = get_metrics(clean_signal, noisy_approx, prediction, mean, logvar, params)["loss"]
            return loss, new_state

        # Get the current batch_stats from the state
        params, batch_stats = state.params, state.batch_stats
        (_, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch_stats)
        
        # Update the state with the new parameters and batch_stats
        state = state.apply_gradients(grads=grads, batch_stats=new_state["batch_stats"])
        return state

    return train_step


def create_eval_f(get_metrics, model_args):
    @jax.jit
    def eval_f(state, batch, z_rng):
        
        # Unpack the inputs
        params, batch_stats = state.params, state.batch_stats
        clean_signal, noisy_approx = batch
        
        # Apply the model in evaluation mode
        prediction, mean, logvar = models.model(
            hidden=model_args["hidden"],
            latents=model_args["latents"],
            dropout_rate=model_args["dropout_rate"],
            io_dim=model_args["io_dim"],
        ).apply(
            {'params': params, 'batch_stats': batch_stats},
            x=noisy_approx,
            z_rng=z_rng,
            deterministic=True,
            mutable=False  # No need to update batch_stats during evaluation
        )
        
        metrics = get_metrics(clean_signal, noisy_approx, prediction, mean, logvar, params)
        return metrics

    return eval_f



def train_and_evaluate(config: ml_collections.ConfigDict, working_dir: str):
    """Train and evaulate pipeline."""
    # Set up the random number generators
    # rng is the random number generator and therefore never passed to the model
    time_keeping = time()
    rng = random.key(jnp.int64(2000))
    rng, init_rng, example_rng, z_rng, io_rng  = random.split(rng, 5)
    
    # Define the test data parameters
    data_args = {
        "params": {
            "a_min": 0,
            "a_max": 1,
            "tau_min": 5,
            "tau_max": 300,
            "decay_count": 2,
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
        state = CustomTrainState(
            step=1000*10000,  # Restore the step count from opt_state
            apply_fn=models.model(**model_args).apply,
            params=params,
            tx=optax.adam(create_learning_rate_scheduler(config)),
            opt_state=opt_state,  # Set the optimizer state
        )
        
        # original_state = state
    else:
        model_args = {
            "latents": config.latents,
            "hidden": config.hidden,
            "dropout_rate": config.dropout_rate, 
            "io_dim": io_dim,
            "dtype": data_args["dtype"],
        }
        
        # Initialize the model with some dummy data
        logging.info('Initializing model.')
        init_data = jnp.ones((config.batch_size, io_dim), dtype=data_args["dtype"])
        
        variables = models.model(**model_args).init(init_rng, init_data, z_rng, deterministic=True)
        params = variables['params']
        batch_stats = variables['batch_stats']
        
        # Initialize the training state including the optimizer
        state = CustomTrainState.create(
            apply_fn=models.model(**model_args).apply,
            params=params,
            tx=optax.adam(create_learning_rate_scheduler(config)),
            batch_stats=batch_stats,  # Include batch_stats in the state
        )
        
    # Create the data generator
    data_generator = input_pipeline.create_data_generator(data_args)
    
    # Use the data generateor to create an example batch so that losses can be 
    # meaningfully normalized to the noisy data
    example_batch = next(data_generator(key=example_rng, n=config.epoch_size))
    get_metrics = loss.create_compute_metrics(config.loss_scaling, example_batch, data_args["wavelet"], data_args["mode"])
    
    # # This section creates a second example batch to verify that the loss normalization is working correctly
    # example_batch = next(data_generator(key=example_rng, n=config.epoch_size))
    # metrics = get_metrics(example_batch[0], example_batch[1], example_batch[1], None, None, state.params)
    # loss.print_metrics(metrics, "Example batch metrics: ")
    
    # Create the training step and evaluation function
    train_step = create_train_step(get_metrics, model_args)
    eval_f = create_eval_f(get_metrics, model_args)

    metric_list = []
    best_loss = jnp.inf
    
    logging.info(f"time taken to initialize: {time()-time_keeping:.3f}s")
    logging.info(
        f"This training instance has the following details:\n"
        f"configs: {config}\n"
        f"data_args: {data_args}\n"
    )
    # Clear memory
    del example_batch, time_keeping, init_rng, example_rng, z_rng, io_rng
    
    # We want to verify that all of the data has the jnp.float64 type
    # print(f"Intended Data type: {data_args['dtype']}")
    
    # Train the model
    for epoch in range(config.num_epochs):
        
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
        
        # Train the model for one epoch
        for batch in train_ds:
            rng, train_rng = random.split(rng)
            state = train_step(state, batch, train_rng)
        
        # Evaluate the model
        rng, eval_rng = random.split(rng)
        metrics = eval_f(state, test_batch, eval_rng)
        
        metric_list.append(metrics)

        # Print the evaluation metrics
        if (epoch + 1) % 1 == 0:
            loss.print_metrics(metrics, f"epoch: {epoch + 1}, time {time()-start_time:.2f}s, ")
            if jnp.isnan(metrics['loss']):
                logging.info("NaN loss detected. Exiting training.")
                break
        
        # Save the best model, assuming that it performs equally well on the validation set
        if epoch > config.num_epochs/20 and best_loss > sum(value for key, value in metrics.items() if key not in {"loss", "l2", "kl"}):
            
            # Create a validation data set 
            rng, test_rng, z_rng = random.split(rng, 3)
            test_batch = next(data_generator(
                key=test_rng, 
                n=config.batch_size*5
            ))
            metrics = eval_f(state, test_batch, eval_rng)
            
            comparison_loss = sum(value for key, value in metrics.items() if key not in {"loss", "l2", "kl"})
            
            metrics["loss"] = comparison_loss
            
            if comparison_loss < best_loss:
                best_loss = comparison_loss
                loss.print_metrics(metrics, f"New best loss at epoch: {epoch + 1}, ")
                utils.save_model(state, 0, working_dir + 'tmp/checkpoints/best_this_run', model_args, logging=False)
        
        # Save the model
        if (epoch + 1) % 100 == 0:
            utils.save_model(state, epoch + 1, working_dir + 'tmp/checkpoints', model_args)
            # state = original_state
            # utils.plot_comparison(comparison, epoch+1, working_dir + 'tmp/checkpoints/reconstruction_{}.png'.format(epoch+1))
            
    # Save the results
    utils.plot_inverse_loss(metric_list, working_dir + "dae/loss.png")
