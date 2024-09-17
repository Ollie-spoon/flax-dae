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

"""VAE model definitions."""

from flax import linen as nn
from jax import random
import jax.numpy as jnp


class Encoder(nn.Module):
  """VAE Encoder."""

  latents: int
  hidden: int
  dropout_rate: float

  @nn.compact
  def __call__(self, x, deterministic):
    x = nn.Dense(self.hidden, name='fc1')(x)
    x = nn.Dropout(self.dropout_rate, deterministic=deterministic)(x)
    x = nn.gelu(x)
    x = nn.Dense(self.latents, name='fc2_mean')(x)
    return x


class Decoder(nn.Module):
  """VAE Decoder."""
  
  hidden: int
  dropout_rate: float
  io_dim: int

  @nn.compact
  def __call__(self, z, deterministic):
    z = nn.Dense(self.hidden, name='fc1')(z)
    z = nn.Dropout(self.dropout_rate, deterministic=deterministic)(z)
    z = nn.gelu(z)
    z = nn.Dense(self.io_dim, name='fc2')(z)
    return z


class DAE(nn.Module):
    """Full VAE model."""

    latents: int = 20
    hidden: int = 50
    dropout_rate: float = 0.05
    io_dim: int = 95

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

    def __call__(self, x, deterministic: bool = True):
        x = self.encoder(x, deterministic)
        z = reparameterize(x) # used to be random for vae: reparameterize(x, z_rng)
        recon_x = self.decoder(z, deterministic)
        return recon_x

    # Generate samples from the VAE using latents as input
    def generate(self, z):
        return nn.sigmoid(self.decoder(z))

# Currently does nothing but can be used to reparameterize the latents
def reparameterize(x):
    return x


def model(latents, hidden, dropout_rate, io_dim):
    return DAE(
        latents=latents,
        hidden=hidden,
        dropout_rate=dropout_rate,
        io_dim=io_dim,
      )
