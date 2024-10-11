"""DAE model definitions."""

import jax 
# jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, random


class ResidualBlock(nn.Module):
    """Residual block with a skip connection and pre-activation."""
    
    io_dim: int
    hidden: int
    dropout_rate: float
    
    @nn.compact
    def __call__(self, x, deterministic: bool):
        x_init = x
        
        # Pre-activation
        # d = nn.GroupNorm(use_running_average=deterministic)(x)
        x = nn.gelu(x)

        # Hidden layer 1  (io_dim -> hidden)
        x = nn.Dense(features=self.hidden)(x)
        d = nn.GroupNorm(use_running_average=deterministic)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        
        # Hidden layer 2  (hidden -> io_dim)
        x = nn.Dense(features=self.io_dim)(x)
        
        # Residual connection
        x = x_init + x
        
        return x

class Encoder(nn.Module):
    """DAE Encoder."""

    io_dim: int
    hidden: int
    latents: int
    dropout_rate: float
    dtype: type
        
    @nn.compact
    def __call__(self, x, deterministic: bool):
        
        # # Residual block 1 (io_dim -> io_dim)
        # x = ResidualBlock(self.io_dim, self.hidden, self.dropout_rate)(x, deterministic=deterministic)
        
        # Hidden layer 1  (io_dim -> hidden)
        x = nn.Dense(features=self.hidden)(x)
        # d = nn.BatchNorm(use_running_average=deterministic)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        # Latent layer  (hidden -> latent)
        mean_x = nn.Dense(features=self.latents)(x)
        logvar_x = nn.Dense(features=self.latents)(x)
        
        return mean_x, logvar_x


class Decoder(nn.Module):
    """DAE Decoder."""

    io_dim: int
    hidden: int
    dropout_rate: float
    dtype: type
        
    @nn.compact
    def __call__(self, z, deterministic: bool):
        
        # Hidden layer 1  (latent -> io_dim)
        z = nn.Dense(features=self.io_dim)(z)
        # d = nn.BatchNorm(use_running_average=deterministic)(z)
        z = nn.gelu(z)
        z = nn.Dropout(rate=self.dropout_rate)(z, deterministic=deterministic)
        
        # # Residual block 1 (io_dim -> io_dim)
        # z = ResidualBlock(self.io_dim, self.hidden, self.dropout_rate)(z, deterministic=deterministic)

        # Output layer  (io_dim -> io_dim)
        z = nn.Dense(features=self.io_dim)(z)
        return z


class DAE(nn.Module):
    """Full DAE model."""

    io_dim: int
    hidden: int
    latents: int
    dropout_rate: float
    dtype: type

    def setup(self):
        self.encoder = Encoder(
            self.io_dim,
            self.hidden,
            self.latents,
            self.dropout_rate,
            self.dtype,
            )
        self.decoder = Decoder(
            self.io_dim,
            self.hidden,
            self.dropout_rate,
            self.dtype,
        )

    def __call__(self, x, z_rng, deterministic: bool):
        
        # Encode the input
        mean, logvar = self.encoder(x, deterministic)
        
        # Reparameterize the latent space representation from a distribution to a point
        z = jnp.where(deterministic, mean, reparameterize_norm(z_rng, mean, logvar))
        
        # Decode the latent space representation
        recon_x = self.decoder(z, deterministic)
        
        return recon_x, mean, logvar


@jit
def reparameterize_norm(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std

# @jit
# def reparameterize_truncated_normal(self, rng, mean, logvar):
#     # a and b are the lower and upper bounds of the truncated normal distribution
#     std = jnp.exp(0.5 * logvar)
#     eps = random.truncated_normal(key=rng, lower=0, upper=jnp.inf, shape=logvar.shape, dtype=self.dtype)
#     return mean + eps * std


def model(latents, hidden, dropout_rate, io_dim, noise_std, dtype=jnp.float32):
    # return DAE(
    #     latents=latents,
    #     hidden=hidden,
    #     dropout_rate=dropout_rate,
    #     io_dim=io_dim,
    #     dtype=dtype,
    # )
    return CNN(
        kernel_size=5,
        io_dim=io_dim,
        features=hidden,
        dropout_rate=dropout_rate,
        noise_std=noise_std,
    )
    # return UNet(
    #     kernel_size=7,
    #     io_dim=io_dim,
    #     features=hidden,
    #     padding='SAME',
    # )


class CNN(nn.Module):
    """Convolutional model, currently an autoencoder."""
    
    kernel_size: int
    io_dim: int
    features: int
    dropout_rate: float
    noise_std: jnp.array
    
    @nn.compact
    def __call__(self, x, z_rng, deterministic: bool):
        assert self.kernel_size % 2 == 1, "Kernel size must be odd."
        
        x = jnp.reshape(x, (x.shape[0], x.shape[1], 1))
        
        # Starting size = 1120
        # Ending size = 68
        
        # (1120 - (4*2))/2 = 556
        # (556 - (4*2))/2 = 274
        # (274 - (4*2))/2 = 133
        # (133 - (4*2))/2 = 64
        # (64 - (4*2))/2 = 28
        
        # jax.debug.print("x0: {}", x.shape)
        
        # Initial convolutional block (1120 -> 1112)
        x = ConvolutionalBlock(self.features, self.kernel_size, 'VALID', deterministic)(x)
        x = ConvolutionalBlock(self.features, self.kernel_size, 'VALID', deterministic)(x)
        
        # jax.debug.print("x1: {}", x.shape)
        
        # Layer 1 (1112 -> 556 -> 548)
        # Pooling (1112 -> 556)
        x = nn.pooling.avg_pool(x, window_shape=(2,), strides=(2,), padding='VALID')
        
        # Convolutional block (556 -> 548)
        x = ConvolutionalBlock(self.features*2, self.kernel_size, 'VALID', deterministic)(x)
        x = ConvolutionalBlock(self.features*2, self.kernel_size, 'VALID', deterministic)(x)
        
        # jax.debug.print("x2: {}", x.shape)
        
        # Layer 2 (548 -> 274 -> 266)
        # Pooling (548 -> 274)
        x = nn.pooling.avg_pool(x, window_shape=(2,), strides=(2,), padding='VALID')
        
        # Convolutional block (274 -> 266)
        x = ConvolutionalBlock(self.features*4, self.kernel_size, 'VALID', deterministic)(x)
        x = ConvolutionalBlock(self.features*4, self.kernel_size, 'VALID', deterministic)(x)
        
        # jax.debug.print("x3: {}", x.shape)
        
        # Layer 3 (266 -> 133 -> 125)
        # Pooling (266 -> 133)
        x = nn.pooling.avg_pool(x, window_shape=(2,), strides=(2,), padding='VALID')
        
        # Convolutional block (133 -> 125)
        x = ConvolutionalBlock(self.features*8, self.kernel_size, 'VALID', deterministic)(x)
        x = ConvolutionalBlock(self.features*8, self.kernel_size, 'VALID', deterministic)(x)
        
        # jax.debug.print("x4: {}", x.shape)
        
        # Layer 4 (125 -> 62 -> 54)
        # Pooling (125 -> 62)
        x = nn.pooling.avg_pool(x, window_shape=(2,), strides=(2,), padding='VALID')
        
        # Convolutional block (62 -> 54)
        x = ConvolutionalBlock(self.features*16, self.kernel_size, 'VALID', deterministic)(x)
        x = ConvolutionalBlock(self.features*16, self.kernel_size, 'VALID', deterministic)(x)
        
        # jax.debug.print("x5: {}", x.shape)
        
        # Layer 5 (54 -> 27 -> 19)
        # Pooling (54 -> 27)
        x = nn.pooling.avg_pool(x, window_shape=(2,), strides=(2,), padding='VALID')
        
        # Convolutional block (27 -> 19)
        x = ConvolutionalBlock(self.features*32, self.kernel_size, 'VALID', deterministic)(x)
        x = ConvolutionalBlock(self.features*32, self.kernel_size, 'VALID', deterministic)(x)
        
        # jax.debug.print("x6: {}", x.shape)
        
        # Flatten (19 -> 19)
        x = nn.Conv(features=1, kernel_size=(1, 1), padding='VALID')(x)
        # x = nn.gelu(x)
        x = jnp.reshape(x, (x.shape[0], x.shape[1]))
        
        # jax.debug.print("x7: {}", x.shape)
        
        # Fully connected layer (19 -> 136)
        x = nn.Dense(features=self.io_dim*2)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        
        # jax.debug.print("x8: {}", x.shape)
        
        # Output layer (136 -> 68)
        x = nn.Dense(features=self.io_dim)(x)

        return x
    
    
    # @nn.compact
    # def __call__(self, x, z_rng, deterministic: bool):
        
    #     dx = nn.Conv(features=self.io_dim, kernel_size=(self.kernel_size))(x)
    #     dx = nn.BatchNorm(use_running_average=deterministic)(dx)
    #     dx = nn.gelu(dx)
    #     dx = nn.ConvTranspose(features=self.io_dim, kernel_size=(self.kernel_size))(dx)
        
    #     x = x + dx

@jit
def reparameterize_dx(rng, sign, power, deterministic):
    
    # if deterministic is true then we want to simply round the sign, but if not 
    # we want to sample from a Bernoulli distribution so that the model learns 
    # that the sign is a probability of being positive
    sign = jnp.where(deterministic, jnp.round(sign), random.bernoulli(rng, sign))
    sign = 2 * sign - 1
    
    # power extends from 10 to 10^-9 
    power = 10*(power - 1) + 1
    
    return sign * 10 ** power


class UNet(nn.Module):
    """U-Net model for signal or denoising."""
    
    kernel_size: int
    io_dim: int
    features: int
    padding: str = 'SAME'
    
    @nn.compact
    def __call__(self, x, z_rng, deterministic: bool):
        assert self.kernel_size % 2 == 1, "Kernel size must be odd."
        
        x0 = jnp.reshape(x, (x.shape[0], x.shape[1], 1))
        
        # jax.debug.print("input: {}", x.shape)
        
        # Contracting path (Encoder)
        # Initial convolutional block
        x0 = ConvolutionalBlock(self.features, self.kernel_size, self.padding, deterministic)(x0)
        x0 = ConvolutionalBlock(self.features, self.kernel_size, self.padding, deterministic)(x0)
        
        # jax.debug.print("x0: {}", x0.shape)
        
        # Layer 1 (io_dim -> io_dim/2)
        x1 = UNetDownLayer(self.features*2, self.kernel_size, self.padding, deterministic)(x0)
        
        # jax.debug.print("x1: {}", x1.shape)
        
        # Layer 2 (io_dim/2 -> io_dim/4)
        x2 = UNetDownLayer(self.features*4, self.kernel_size, self.padding, deterministic)(x1)
        
        # jax.debug.print("x2: {}", x2.shape)
        
        # Expanding path (Decoder)
        
        # Upsample 1 (io_dim/4 -> io_dim/2)
        x1 = UNetUpLayer(self.features*2, self.kernel_size, self.padding, deterministic)(x2, x1)
        
        # jax.debug.print("x1: {}", x1.shape)
        
        # Upsample 2 (io_dim/2 -> io_dim)
        x0 = UNetUpLayer(self.features, self.kernel_size, self.padding, deterministic)(x1, x0)
        
        # jax.debug.print("x0: {}", x0.shape)
        
        # Final output layer
        x0 = nn.Conv(features=1, kernel_size=(1, 1), padding='VALID')(x0)  # Reduce to single output channel
        
        # jax.debug.print("output: {}", x0.shape)
        
        x0 = jnp.reshape(x0, (x0.shape[0], x0.shape[1]))
        
        # x0 = nn.Dense(features=self.io_dim)(x0)
        # x0 = nn.gelu(x0)
        # # x_ = nn.Dropout(rate=0.2)(x_, deterministic=deterministic)
        

        x_sign = nn.Dense(features=self.io_dim)(x0)
        x_power = nn.Dense(features=self.io_dim)(x0)
        
        x = reparameterize_dx(z_rng, x_sign, x_power, deterministic)
        
        
        return x #+ x

class ConvolutionalBlock(nn.Module):
    
    features: int
    kernel_size: int
    padding: str
    deterministic: bool
    
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.features, kernel_size=(self.kernel_size), padding=self.padding)(x)
        x = nn.GroupNorm(group_size=4, num_groups=None)(x)
        x = nn.gelu(x)
        
        return x

class UNetDownLayer(nn.Module):
    """U-Net down layer."""
    
    features: int
    kernel_size: int
    padding: str
    deterministic: bool
    
    @nn.compact
    def __call__(self, x):
        x = nn.pooling.max_pool(x, window_shape=(2,), strides=(2,), padding='VALID')
        
        x = ConvolutionalBlock(self.features, self.kernel_size, self.padding, self.deterministic)(x)
        x = ConvolutionalBlock(self.features, self.kernel_size, self.padding, self.deterministic)(x)
        
        return x
    
class UNetUpLayer(nn.Module):
    """U-Net up layer."""
    
    features: int
    kernel_size: int
    padding: str
    deterministic: bool
    
    @nn.compact
    def __call__(self, x, x_skip):
        x = nn.ConvTranspose(features=self.features, kernel_size=(self.kernel_size), strides=(2), padding='SAME')(x)
        
        x = jnp.concatenate([x_skip, x], axis=-1)  # Skip connection
        
        x = ConvolutionalBlock(self.features, self.kernel_size, self.padding, self.deterministic)(x)
        x = ConvolutionalBlock(self.features, self.kernel_size, self.padding, self.deterministic)(x)
        
        return x