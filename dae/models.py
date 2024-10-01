"""DAE model definitions."""

import jax 
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, random


class Encoder(nn.Module):
    """DAE Encoder."""

    latents: int
    hidden: int
    dropout_rate: float
    dtype: type

    def setup(self):
        # Create the modules we need to build the network
        
        # Hidden layer 1
        self.hidden_layer_1 = nn.Dense(
            features=self.hidden, 
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(dtype=self.dtype),
        )
        self.hidden_dropout_1 = nn.Dropout(rate=self.dropout_rate)
        
        # Latent layer
        self.mean_layer = nn.Dense(
            features=self.latents, 
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(dtype=self.dtype),
        )
        self.logvar_layer = nn.Dense(
            features=self.latents, 
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(dtype=self.dtype),
        )
        
    
    def __call__(self, x, deterministic: bool):

        # Hidden layer 1  (io_dim -> hidden)
        x = self.hidden_layer_1(x)
        x = nn.gelu(x)
        x = self.hidden_dropout_1(x, deterministic=deterministic)

        # Latent layer  (hidden -> latent)
        mean_x = self.mean_layer(x)
        logvar_x = self.logvar_layer(x)
        
        return mean_x, logvar_x


class Decoder(nn.Module):
    """DAE Decoder."""

    hidden: int
    dropout_rate: float
    io_dim: int
    dtype: type

    def setup(self):
        # Create the modules we need to build the network
        
        # Hidden layer 1
        self.hidden_layer_1 = nn.Dense(
            features=self.hidden,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(dtype=self.dtype),
        )
        self.hidden_dropout_1 = nn.Dropout(rate=self.dropout_rate)
        
        # ouptut layer
        self.output_layer = nn.Dense(
            features=self.io_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(dtype=self.dtype),
        )
        
    
    def __call__(self, z, deterministic: bool):
        
        # Hidden layer 1  (latent -> hidden)
        z = self.hidden_layer_1(z)
        z = nn.gelu(z)
        z = self.hidden_dropout_1(z, deterministic=deterministic)

        # Output layer  (hidden -> io_dim)
        z = self.output_layer(z)
        return z


class DAE(nn.Module):
    """Full DAE model."""

    latents: int
    hidden: int
    dropout_rate: float
    io_dim: int
    dtype: type

    def setup(self):
        self.encoder = Encoder(
            self.latents,
            self.hidden,
            self.dropout_rate,
            self.dtype,
            )
        self.decoder = Decoder(
            self.hidden,
            self.dropout_rate,
            self.io_dim,
            self.dtype,
        )

    def __call__(self, x, z_rng, deterministic: bool = True):
        mean, logvar = self.encoder(x, deterministic)
        z = jnp.where(deterministic, mean, reparameterize_norm(z_rng, mean, logvar))
        # z = mean
        # z = jnp.where(deterministic, mean, reparameterize_truncated_normal(z_rng, mean, logvar))
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
