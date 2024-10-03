"""DAE model definitions."""

import jax 
jax.config.update('jax_enable_x64', True)
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
        x = nn.BatchNorm(use_running_average=deterministic)(x)
        x = nn.gelu(x)

        # Hidden layer 1  (io_dim -> hidden)
        x = nn.Dense(features=self.hidden)(x)
        x = nn.BatchNorm(use_running_average=deterministic)(x)
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
        x = nn.BatchNorm(use_running_average=deterministic)(x)
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
        z = nn.BatchNorm(use_running_average=deterministic)(z)
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


def model(latents, hidden, dropout_rate, io_dim, dtype=jnp.float32):
    return DAE(
        latents=latents,
        hidden=hidden,
        dropout_rate=dropout_rate,
        io_dim=io_dim,
        dtype=dtype,
      )
