from jax import jit, vmap, random, numpy as jnp
from flax import linen as nn
import chex

from models import ConvolutionalBlock

## TODO: 
## 1. Modify embedding layer to work with vmap
## 2. Check if the embedding layer is working correctly
## 3. Check if the tokenization layer is working correctly
## 4. Implement the detokenization layer and unembedding layer
## 4. Fix the MLP layer
## 5. Implement an encoder block layer
## 6. Implement the full transformer model

## DONE:
## 1. Implement the embedding layer

class TransformerConfig:
    
    # Tokenizer parameters
    min_tokens: int = 7
    max_tokens: int = 18
    token_dim: int = 64
    overlap_per_token: int = 8
    
    # Embedding parameters
    embed_features: int = 16
    embed_kernel_size: int = 3
    
    # Dimensions
    embed_dim: int = 24
    qkv_dim: int = 16
    mlp_dim: int = 4 * embed_dim
    
    # Transformer parameters
    num_heads: int = 8
    num_layers: int = 8
    
    # Dropout rates
    dropout_rate: float = 0.1
    attn_dropout_rate: float = 0.05
    
    # Stabilizer for the softmax function
    eps: float = 1e-6
    
    # dtype of the computation (default: float32)
    dtype: type = jnp.float32
    
    # Kernel initializer
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
    
    # Bias initializer
    bias_init: nn.initializers.Initializer = nn.initializers.zeros
    
    # Deterministic mode
    deterministic: bool = False
    
class Tokenizer(nn.Module):
    
    config: TransformerConfig
    
    def setup(self, config):
        self.token_dim = config.token_dim
        self.overlap = config.overlap_per_token
        self.max_tokens = config.max_tokens
        self.min_tokens = config.min_tokens
        
        self.max_len = self.get_len_from_tokens(self.max_tokens)
        self.min_len = self.get_len_from_tokens(self.min_tokens)
        
        
        
        return super().setup()
    
    def __call__(self, inputs):
        """Applies Transformer Tokenizer module.
        
        Tokenize input sequence into overlapping tokens.
        """
        
        # Clip input sequence to fit tokens
        temp = jnp.zeros((inputs.shape[0], self.max_len))
        
        
        token_clip_len = (inputs.shape[1] - token_dim) % (token_dim - overlap_per_token)
        seq_len = inputs.shape[1] - token_clip_len
        inputs = inputs[:, :seq_len]
        
        # Calculate number of tokens
        num_tokens = (seq_len - token_dim)//(token_dim - overlap_per_token) + 1
        num_tokens = min(num_tokens, max_tokens)
        
        # Stack tokens - This should result in a shape of (batch_size, num_tokens, token_dim)
        tokens = jnp.stack([inputs[:, i:i+token_dim] for i in range(0, seq_len-overlap_per_token, token_dim-overlap_per_token)], axis=1)

        return tokens
    
    def get_len_from_tokens(self, tokens):
        """Calculates the length of the input sequence from tokens.
        
        Calculates the length of the input sequence from tokens.
        """
        return tokens*(self.token_dim - self.overlap) + self.overlap
    
    @vmap
    def padd_signal(self, input):
        """Pads input sequence to fit tokens.
        
        Pads input sequence to fit tokens.
        """
        
        chex.assert_rank(input, 1)
        assert input >= self.token_dim, f"Input sequence length must be at least {self.token_dim}."
        
        
        clip_len = jnp.min(input.shape[-1], self.max_len)
        padded = jnp.zeros(self.max_len)
        padded = padded.at[:clip_len].set(input)
        
        unpadded_len = jnp.floor((input.shape[1] - token_dim) / (token_dim - overlap_per_token) + 1, dtype=jnp.int32)
        mask = jnp.zeros(max_tokens)
        mask = mask.at[:unpadded_len].set(1)
        
        return inputs
    
class Detokenizer(nn.Module):
    
    config: TransformerConfig
    
    def __call__(self, tokens):
        """Applies Transformer Detokenizer module.
        
        Detokenize token sequence into a single sequence.
        """
        config = self.config
        token_dim = config.token_dim
        overlap_per_token = config.overlap_per_token
        
        # Calculate sequence length
        seq_len = tokens.shape[1]*(token_dim - overlap_per_token) + overlap_per_token
        
        # Stack tokens - This should result in a shape of (batch_size, seq_len)
        seq = jnp.concatenate([tokens[:, i, :] for i in range(tokens.shape[1])], axis=0)

        return seq
    
class Embedding(nn.Module):
    
    config: TransformerConfig
    
    def __call__(self, inputs):
        """Applies Transformer Embedding module.
        
        Token embedding + positional encoding.
        """
        # chex.assert_rank(inputs, 2)
        # chex.assert_equal(inputs.shape[1], config.token_dim)
        
        config = self.config
        embed_dim = config.embed_dim
        features = config.embed_features
        max_tokens = config.max_tokens
        
        x = jnp.reshape(inputs, (*inputs.shape , 1))
        
        # Embedding layer (64 -> 60)
        x = nn.Conv(features=features, kernel_size=5, padding='VALID')(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        
        # Pooling layer (60 -> 30)
        x = nn.Conv(features=features, kernel_size=2, strides=2, padding='VALID')(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        
        # Embedding layer (30 -> 24)
        x = nn.Conv(features=features, kernel_size=7, padding='VALID')(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        
        # Flatten (16 features -> 1 feature)
        x = nn.Conv(features=1, kernel_size=1, padding='VALID')(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        x = jnp.reshape(x, x.shape[:-1])
        
        # At this point we have taken the input sequence and embedded it
        # from 64 dimensions to 24 dimensions.
                
        # Learned positional encoding
        pos_enc = self.param(
            name='pos_enc', 
            init_fn=nn.initializers.normal(stddev=0.02), 
            shape=(1, max_tokens, embed_dim),
        )
        
        return x + pos_enc[:, :x.shape[1], :]

class Unembedding(nn.Module):
    
    config: TransformerConfig
    
    def __call__(self, inputs):
        """Applies Transformer Unembedding module.
        
        Unembedding the input sequence into a token sequence.
        """
        # chex.assert_rank(inputs, 2)
        # chex.assert_equal(inputs.shape[1], config.embed_dim)
        
        config = self.config
        embed_dim = config.embed_dim
        features = config.embed_features
        max_tokens = config.max_tokens
        
        x = jnp.reshape(inputs, (*inputs.shape, 1))
        
        # Unembedding layer (24 -> 30)
        x = nn.ConvTranspose(features=features, kernel_size=7, padding='VALID')(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        
        # Unpooling layer (30 -> 60)
        x = nn.ConvTranspose(features=features, kernel_size=2, strides=2, padding='VALID')(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        
        # Unembedding layer (60 -> 64)
        x = nn.ConvTranspose(features=features, kernel_size=5, padding='VALID')(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        
        # Flatten (16 features -> 1 feature)
        x = nn.ConvTranspose(features=1, kernel_size=1, padding='VALID')(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        x = jnp.reshape(x, (x.shape[0], x.shape[1]))
        
        return x

class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.

    Args:
        config: TransformerConfig dataclass containing hyperparameters.
        out_dim: optionally specify out dimension.
    """

    config: TransformerConfig
    # out_dim: Optional[int] = None

    @nn.compact
    def __call__(self, inputs):
        """Applies Transformer MlpBlock module.
        
        Dense + gelu + Dropout -> Dense + Dropout
        Shape: embeding dim -> config.mlp_dim -> embeding dim
        
        As there is only one activation function here, this is essentially only a single nonlinear transformation.
        """
        config = self.config
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
                config.mlp_dim,
                dtype=config.dtype,
                kernel_init=nn.with_logical_partitioning(
                        config.kernel_init, ('embed', 'mlp')
                ),
                bias_init=nn.with_logical_partitioning(config.bias_init, ('mlp',)),
        )(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=config.dropout_rate)(
                x, deterministic=config.deterministic
        )
        output = nn.Dense(
                actual_out_dim,
                dtype=config.dtype,
                kernel_init=nn.with_logical_partitioning(
                        config.kernel_init, ('mlp', 'embed')
                ),
                bias_init=nn.with_logical_partitioning(config.bias_init, ('embed',)),
        )(x)
        output = nn.Dropout(rate=config.dropout_rate)(
                output, deterministic=config.deterministic
        )
        return output


