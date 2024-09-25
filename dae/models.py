"""DAE model definitions."""

from flax import linen as nn
from jax import jit, random, numpy as jnp


class Encoder(nn.Module):
  """DAE Encoder."""

  latents: int
  hidden: int
  dropout_rate: float

  @nn.compact
  def __call__(self, x, deterministic):
    
    # Hidden layer 1  (io_dim -> hidden)
    x = nn.Dense(self.hidden, name='fc1')(x)
    x = nn.gelu(x)
    x = nn.Dropout(self.dropout_rate, deterministic=deterministic)(x)
    
    # # Hidden layer 2  (hidden -> hidden)
    # x = nn.Dense(self.hidden, name='fc2')(x)
    # x = nn.gelu(x)
    # x = nn.Dropout(self.dropout_rate, deterministic=deterministic)(x)
    
    # Latent layer  (hidden -> latent)
    mean_x = nn.Dense(self.latents, name='fc3_mean')(x)
    logvar_x = nn.Dense(self.latents, name='fc3_logvar')(x)
    # x = nn.Dense(self.latents, name='fc3')(x)
    return mean_x, logvar_x


class Decoder(nn.Module):
    """DAE Decoder."""

    hidden: int
    dropout_rate: float
    io_dim: int

    @nn.compact
    def __call__(self, z, deterministic):
        
        # Hidden layer 1  (latent -> hidden)
        z = nn.Dense(self.hidden, name='fc1')(z)
        z = nn.gelu(z)
        z = nn.Dropout(self.dropout_rate, deterministic=deterministic)(z)

        # # Hidden layer 2  (hidden -> hidden)
        # z = nn.Dense(self.hidden, name='fc2')(z)
        # z = nn.gelu(z)
        # z = nn.Dropout(self.dropout_rate, deterministic=deterministic)(z)

        # Output layer  (hidden -> io_dim)
        z = nn.Dense(self.io_dim, name='fc3')(z)
        return z


class DAE(nn.Module):
    """Full DAE model."""

    latents: int
    hidden: int
    dropout_rate: float
    io_dim: int
    a: float=0
    b: float=jnp.inf

    def setup(self):
        self.encoder = Encoder(
            self.latents,
            self.hidden,
            self.dropout_rate,
            )
        self.decoder = Decoder(
            self.hidden,
            self.dropout_rate,
            self.io_dim,
        )

    def __call__(self, x, z_rng, deterministic: bool = True):
        mean, logvar = self.encoder(x, deterministic)
        # z = reparameterize_lognorm(z_rng, mean, logvar) # used to be random for vae: reparameterize(x, z_rng)
        z = reparameterize_truncated_normal(z_rng, mean, logvar, self.a, self.b)
        recon_x = self.decoder(z, deterministic)
        return recon_x, mean, logvar

# Currently does nothing but can be used to reparameterize the latents
@jit
def reparameterize_lognorm(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std

@jit
def reparameterize_truncated_normal(rng, mean, logvar, a, b):
    # a and b are the lower and upper bounds of the truncated normal distribution
    std = jnp.exp(0.5 * logvar)
    eps = random.truncated_normal(rng, mean.shape, a, b)
    return mean + eps * std


def model(latents, hidden, dropout_rate, io_dim):
    return DAE(
        latents=latents,
        hidden=hidden,
        dropout_rate=dropout_rate,
        io_dim=io_dim,
      )
