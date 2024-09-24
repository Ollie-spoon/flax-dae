"""Input pipeline for DAE dataset."""

import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
from generate_data import create_generate_basic_data
from typing import Union

import jax.numpy as jnp
import jax

def create_data_generator(kwargs):
    generate_data = create_generate_basic_data(**kwargs)
    
    def data_generator(key: jnp.ndarray, n: int, batch_size: Union[int, None]=None):
        
        # Generate the full dataset in one go
        data = generate_data(key, n)
        
        noisy_approx = data[0]  # Noisy version of the approximation coefficient
        clean_signal = data[1]  # Noiseless version in the time domain
        
        if batch_size is None:
            batch_size = n  # Default batch size is all data
            num_batches = 1
        else:
            # Ensure the data can be split into batches
            assert n % batch_size == 0, "Total number of samples must be divisible by batch size."
            
            num_batches = n // batch_size
        
        # Reshape the data into batches
        noisy_approx_batched = noisy_approx.reshape((num_batches, batch_size, *noisy_approx.shape[1:]))
        clean_signal_batched = clean_signal.reshape((num_batches, batch_size, *clean_signal.shape[1:]))
        
        def batch_iterator():
            for i in range(num_batches):
                yield noisy_approx_batched[i], clean_signal_batched[i]
        
        return batch_iterator()
    
    return data_generator



def load_data(data_path, test_size=0.1):
    # Load the .npy file
    data = jnp.load(data_path)
    
    # Split the data into noisy (input) and noiseless (target)
    noisy_data = data[:, 0, :]  # Noisy version
    noiseless_data = data[:, 1, :]  # Noiseless version
    
    # Split into training and testing sets
    split_point = int(len(noisy_data) * (1 - test_size))
    train_noisy, test_noisy = noisy_data[:split_point], noisy_data[split_point:]
    train_noiseless, test_noiseless = noiseless_data[:split_point], noiseless_data[split_point:]
    
    return (train_noisy, train_noiseless), (test_noisy, test_noiseless)

# # instead of loading the data from a file, we will generate it
# def generate_original_data(key: jnp.ndarray, iterations: int, batch_size: int, kwargs):
#     # print("Generating data.")
#     # print("diagnostics: \niterations: ", iterations, "\ntrain: ", train, "\nbatch_size: ", batch_size, "\nkwargs: ", kwargs)
    
#     kwargs["iterations"] = iterations
#     data = generate_basic_data(key=key, **kwargs)
    
#     noisy_data = data[:, 0, :]  # Noisy version
#     noiseless_data = data[:, 1, :]  # Noiseless version
    
#     ds = tf.data.Dataset.from_tensor_slices((noisy_data, noiseless_data))
    
#     # No need to cache, shuffle, or repeat the dataset
#     ds = ds.batch(batch_size)
    
#     return iter(tfds.as_numpy(ds))


# def build_train_set(data_path, batch_size):
#     """Builds a training dataset from custom data."""
#     # Load the data
#     (noisy_data, noiseless_data), _ = load_data(data_path)
    
#     print("Data loaded.")
#     # Calculate steps per epoch
#     num_samples = noisy_data.shape[0]
    
#     # Create a tf.data.Dataset object
#     train_ds = tf.data.Dataset.from_tensor_slices((noisy_data, noiseless_data))
    
#     # Apply the image loading and processing function
#     # train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
#     # Shuffle, repeat, and batch the dataset
#     train_ds = train_ds.cache()
#     train_ds = train_ds.shuffle(1800)
#     train_ds = train_ds.repeat()
#     train_ds = train_ds.batch(batch_size)
    
#     print("Data batched and shuffled.")
    
#     # Return the dataset as a Numpy iterable
#     return iter(tfds.as_numpy(train_ds)), num_samples

# def build_test_set(data_path, batch_size=200):
#     """Builds a testing dataset from custom data."""
#     # Load the data
#     _, (noisy_data, noiseless_data) = load_data(data_path)
    
#     # Calculate steps per epoch
#     num_samples = noisy_data.shape[0]
    
#     # Create a tf.data.Dataset object
#     test_ds = tf.data.Dataset.from_tensor_slices((noisy_data, noiseless_data))
    
#     # Apply the image loading and processing function
#     # test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
#     test_ds = test_ds.batch(num_samples)
    
#     # Return the dataset as a Numpy iterable
#     return iter(tfds.as_numpy(test_ds)), num_samples

# print("Initializing dataset.")
# data_path = 'C:/Users/omnic/OneDrive/Documents/MIT/Programming/approximation_coefficients_dataset.npy'
# batch_size = 100
# data = build_train_set(data_path, batch_size)

# print(data)


## THIS IS THE ORIGINAL IMAGE BASED IMPLEMENTATION OF THE INPUT PIPELINE

# def build_train_set(batch_size, ds_builder):
#   """Builds train dataset."""

#   # Specify that this is the training set
#   train_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
#   # Define the image preprocessing pipeline
#   train_ds = train_ds.map(prepare_image)
#   # Enable caching for faster training
#   train_ds = train_ds.cache()
#   # Repeat the dataset indefinitely
#   train_ds = train_ds.repeat()
#   # Shuffle the dataset
#   train_ds = train_ds.shuffle(10000)
#   # Batch the dataset according to the batch size
#   train_ds = train_ds.batch(batch_size)
#   # Convert the dataset to an iterable numpy array
#   train_ds = iter(tfds.as_numpy(train_ds))
#   return train_ds


# def build_test_set(ds_builder):
#   """Builds test dataset. THIS WAS A MISTAKE IN THE DOCS"""
#   # When compared to the train set, we don't need to cache, iter, or repeat as 
#   # we only iterate over the test set once. We also don't need to shuffle
  
#   # Specify that this is the test set
#   test_ds = ds_builder.as_dataset(split=tfds.Split.TEST)
#   test_ds = test_ds.map(prepare_image).batch(10000)
#   test_ds = jnp.array(list(test_ds)[0])
#   test_ds = jax.device_put(test_ds)
#   return test_ds


# # Convert the image to a float32 tensor
# def prepare_image(x):
#   x = tf.cast(x['image'], tf.float32)
#   x = tf.reshape(x, (-1,))
#   return x
