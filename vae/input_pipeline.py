# Copyright 2024 The Flax Authors.
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

"""Input pipeline for VAE dataset."""

import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

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

def build_train_set(data_path, batch_size):
    """Builds a training dataset from custom data."""
    # Load the data
    (noisy_data, noiseless_data), _ = load_data(data_path)
    
    print("Data loaded.")
    # Calculate steps per epoch
    num_samples = noisy_data.shape[0]
    
    # Create a tf.data.Dataset object
    train_ds = tf.data.Dataset.from_tensor_slices((noisy_data, noiseless_data))
    
    # Apply the image loading and processing function
    # train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Shuffle, repeat, and batch the dataset
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(1800)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(batch_size)
    
    print("Data batched and shuffled.")
    
    # Return the dataset as a Numpy iterable
    return iter(tfds.as_numpy(train_ds)), num_samples

def build_test_set(data_path, batch_size=200):
    """Builds a testing dataset from custom data."""
    # Load the data
    _, (noisy_data, noiseless_data) = load_data(data_path)
    
    # Calculate steps per epoch
    num_samples = noisy_data.shape[0]
    
    # Create a tf.data.Dataset object
    test_ds = tf.data.Dataset.from_tensor_slices((noisy_data, noiseless_data))
    
    # Apply the image loading and processing function
    # test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    test_ds = test_ds.batch(num_samples)
    
    # Return the dataset as a Numpy iterable
    return iter(tfds.as_numpy(test_ds)), num_samples

print("Initializing dataset.")
data_path = 'C:/Users/omnic/OneDrive/Documents/MIT/Programming/approximation_coefficients_dataset.npy'
batch_size = 100
data = build_train_set(data_path, batch_size)

print(data)


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
