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
    
    # Create a tf.data.Dataset object
    train_ds = tf.data.Dataset.from_tensor_slices((noisy_data, noiseless_data))
    
    # Apply the image loading and processing function
    # train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Shuffle, repeat, and batch the dataset
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(1800)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(batch_size)
    
    # Return the dataset as a Numpy iterable
    return iter(tfds.as_numpy(train_ds))

def build_test_set(data_path, batch_size):
    """Builds a testing dataset from custom data."""
    # Load the data
    (noisy_data, noiseless_data), _ = load_data(data_path)
    
    # Create a tf.data.Dataset object
    train_ds = tf.data.Dataset.from_tensor_slices((noisy_data, noiseless_data))
    
    # Apply the image loading and processing function
    # train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Return the dataset as a Numpy iterable
    return iter(tfds.as_numpy(train_ds))