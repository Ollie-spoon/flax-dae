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
    # return CNN(
    #     kernel_size=7,
    #     io_dim=io_dim,
    #     features=hidden,
    #     dropout_rate=dropout_rate,
    #     noise_std=noise_std,
    # )
    return UNet(
        kernel_size=7,
        io_dim=io_dim,
        features=hidden,
        padding='SAME',
    )


class CNN(nn.Module):
    """Convolutional model."""
    
    kernel_size: int
    io_dim: int
    features: int
    dropout_rate: float
    noise_std: jnp.array
    
    @nn.compact
    def __call__(self, x, z_rng, deterministic: bool):
        assert self.kernel_size % 2 == 1, "Kernel size must be odd."
        
        dx = nn.Conv(features=self.features, kernel_size=(self.kernel_size))(x)
        dx = nn.gelu(dx)
        dx = nn.Conv(features=self.features, kernel_size=(self.kernel_size))(dx)
        dx = nn.gelu(dx)
        dx = nn.Conv(features=self.features, kernel_size=(self.kernel_size))(dx)
        dx = nn.gelu(dx)
        dx = nn.Conv(features=self.features, kernel_size=(self.kernel_size))(dx)
        dx = nn.gelu(dx)
        dx = nn.Conv(features=1, kernel_size=(self.kernel_size))(dx)
        dx = nn.gelu(dx)
        
        # Hidden layer 1  (io_dim -> hidden)
        dx = nn.Dense(features=self.io_dim*2)(dx)
        dx = nn.gelu(dx)
        dx = nn.Dropout(rate=self.dropout_rate)(dx, deterministic=deterministic)
        
        # Output layer in two parts  (hidden -> io_dim)
        # sign = nn.Dense(features=self.io_dim)(dx)
        # sign = nn.sigmoid(sign)
        
        # power = nn.Dense(features=self.io_dim)(dx)
        # power = nn.sigmoid(power)
        
        # dx = reparameterize_dx(z_rng, sign, power, deterministic)
        
        dx = nn.Dense(features=self.io_dim)(dx)
        
        # dx_sum = jnp.sum(jnp.abs(dx))
        # dx = jnp.where(dx_sum < self.io_dim/10, dx, dx/dx_sum)
        # jax.debug.print("dx: ", dx)
        
        return dx
    
    
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