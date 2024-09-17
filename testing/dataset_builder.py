import tensorflow_datasets as tfds
from jax.random import permutation


def create_ds_builder(name, description, data_dir):
    
    config = tfds.core.BuilderConfig(
        name=name,
        version=None,
        release_notes=None,
        supported_versions=[],
        description=description,
        tags=[]
        )
    
    ds_builder = tfds.core.DatasetBuilder(
        data_dir=data_dir,
        config=config,
        )
    
    return ds_builder


ds_builder = create_ds_builder(
    name='exp_dec_limited', 
    description='A very limited expoenential decay dataset with biexponentials that contain decays between 5 and 45. The dataset is limited to 2000 samples in total.', 
    data_dir='C:/Users/omnic/OneDrive/Documents/MIT/Programming/approximation_coefficients_dataset.npy',
    )

print(ds_builder)

# # Specify that this is the training set
# train_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
# # # Define the image preprocessing pipeline
# # train_ds = train_ds.map(prepare_image)
# # Enable caching for faster training
# train_ds = train_ds.cache()
# # Repeat the dataset indefinitely
# train_ds = train_ds.repeat()
# # # Shuffle the dataset
# # train_ds = train_ds.shuffle(10000, reshuffle_each_iteration=True)
# # Batch the dataset according to the batch size
# train_ds = train_ds.batch(180)
# # Convert the dataset to an iterable numpy array
# print(train_ds)
# train_ds = iter(permutation(10000, tfds.as_numpy(train_ds), axis=0))
# print(train_ds)

