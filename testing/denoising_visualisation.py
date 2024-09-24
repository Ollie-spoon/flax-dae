import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.random import uniform, normal, split, key
from cr.wavelets import wavedec, waverec
import pickle
from flax import linen as nn
import matplotlib.pyplot as plt


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
    
    # Latent layer  (hidden -> latent)
    x = nn.Dense(self.latents, name='fc2_mean')(x)
    return x


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

        # Output layer  (hidden -> io_dim)
        z = nn.Dense(self.io_dim, name='fc2')(z)
        return z


class DAE(nn.Module):
    """Full DAE model."""

    latents: int
    hidden: int
    dropout_rate: float
    io_dim: int

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


def denoise_bi_exponential():
    # Generate bi-exponential decay
    
    rng = key(2025)
    rng, key1, key2, key3 = split(rng, 4)
    
    t = jnp.linspace(0, 100, 1000)
    a1, a2 = 0.6, 0.4
    tau1 = uniform(key1, minval=5, maxval=30, shape=())
    tau2 = uniform(key2, minval=20, maxval=45, shape=())
    decay = a1 * jnp.exp(-t/tau1) + a2 * jnp.exp(-t/tau2)

    # Add Gaussian noise
    SNR = 200
    noise_scale = 1/SNR
    noise = noise_scale * normal(key3, shape=t.shape)
    noisy_decay = decay + noise

    # Wavelet decomposition
    wavelet = 'coif6'
    mode = 'symmetric'
    coeffs = wavedec(noisy_decay, wavelet, mode=mode)
    coeffs_clean = wavedec(decay, wavelet, mode=mode)
    clean_approx = coeffs_clean[0]

    # Load neural network model
    with open(r"C:/Users/omnic/OneDrive/Documents/MIT/Programming/dae/flax/tmp/checkpoints/checkpoint_200.pkl", 'rb') as f:
        checkpoint = pickle.load(f)
    
    # params = checkpoint['params']
    # opt_state = checkpoint['opt_state']
    # model_args = checkpoint['model_args']

    # Pass approximation coefficients through neural network
    approx_coeffs = coeffs[0]
    denoised_approx_coeffs = eval_f(
        params=checkpoint['params'],
        noisy_data=approx_coeffs,
        model_args=checkpoint['model_args']
    )

    # Inverse wavelet decomposition with denoised approximation coefficients
    plt.title("Comparison of noisy and denoised approximation coefficients.")
    # plt.plot(coeffs[0], label='Noisy')
    noisy_approx = coeffs[0]
    coeffs[0] += denoised_approx_coeffs
    # plt.plot(coeffs[0], label='Denoised')
    # plt.plot(jnp.log10(clean_approx), label='Clean')
    # plt.xlabel("index")
    # plt.ylabel("coefficient amplitude")
    # plt.legend()
    # plt.show()
    
    denoised_decay = waverec(coeffs, wavelet, mode=mode)
    coeffs_clean[0] = noisy_approx
    injected_original = waverec(coeffs_clean, wavelet, mode=mode)
    coeffs_clean[0] = coeffs[0]
    injected_denoised = waverec(coeffs_clean, wavelet, mode=mode)
    
    print(f"The original SNR of the signal was {1/jnp.std(noisy_decay-decay)}")
    print(f"The denoised signal SNR was {1/jnp.std(denoised_decay-decay)}")
    
    print(f"The mse loss for the noisy signal was {get_mse_loss(noisy_approx, clean_approx)}")
    print(f"The mse loss for the denoised signal was {get_mse_loss(coeffs[0], clean_approx)}")

    # # Plot comparison
    # plt.title("Comparison of noisy and denoised signals")
    # plt.plot(t, noisy_decay - decay, label='Noisy')
    # plt.plot(t, denoised_decay - decay, label='Denoised')
    # plt.xlabel("time (ms)")
    # plt.ylabel("signal amplitude")
    # plt.legend()
    # plt.show()
    
    # plt.title("Comparison of noisy and denoised approximation coefficient injections")
    # plt.plot(t, noisy_decay - decay, label='Noisy')
    # plt.plot(t, injected_original - decay, label='Noisy Injected')
    # plt.plot(t, injected_denoised - decay, label='Denoised Injected')
    # plt.xlabel("time (ms)")
    # plt.ylabel("signal noise amplitude")
    # plt.legend()
    # plt.show()
    
    # For this section we are going to test out some interesting things
    
    # First, we're going to see how much the edges effect the signal
    # Is this data lost?
    
    # coeffs_clean[0] = clean_approx
    # # coeffs_clean[0][:10] = noisy_approx[:10]
    # for i in range(1,21):
    #     coeffs_clean[0] = coeffs_clean[0].at[-i:].set(noisy_approx[-i:])
    #     reconstructed_clean = waverec(coeffs_clean, wavelet, mode=mode)
        
    #     # plt.plot(t, decay, label="original")
    #     # plt.plot(t, reconstructed_clean, label="Reconstructed")
    #     # plt.legend()
    #     # plt.show()
        
    #     plt.title(f"for changing the last {i} values")
    #     plt.plot(t, decay-reconstructed_clean)
    #     plt.show()

def get_mse_loss(recon_x, noiseless_x):
    return jnp.mean(jnp.square(recon_x - noiseless_x))

# Define the evaluation function
def eval_f(params, noisy_data, model_args):
    
    def eval_model(vae):
        return vae(noisy_data, deterministic=True)

    return nn.apply(eval_model, model(**model_args))({'params': params})


denoise_bi_exponential()