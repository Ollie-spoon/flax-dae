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
## 2. Implement the unembedding layer

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
        
        x = jnp.reshape(inputs, (*inputs.shape, 1))
        
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
        # embed_dim = config.embed_dim
        features = config.embed_features
        # max_tokens = config.max_tokens
        
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
        # x = nn.LayerNorm()(x)
        # x = nn.gelu(x)
        x = jnp.reshape(x, x.shape[:-1])
        
        return x

class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.

    Args:
        config: TransformerConfig dataclass containing hyperparameters.
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
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=config.deterministic)
        x = nn.Dense(
                actual_out_dim,
                dtype=config.dtype,
                kernel_init=nn.with_logical_partitioning(
                        config.kernel_init, ('mlp', 'embed')
                ),
                bias_init=nn.with_logical_partitioning(config.bias_init, ('embed',)),
        )(x)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=config.deterministic)
        return x
    
class EncoderBlock(nn.Module):
    
    config: TransformerConfig
    
    def __call__(self, inputs):
        """Applies Transformer EncoderBlock module.
        
        Encoder block with multi-head self-attention and feed-forward layer.
        """
        config = self.config
        embed_dim = config.embed_dim
        qkv_dim = config.qkv_dim
        num_heads = config.num_heads
        attn_dropout_rate = config.attn_dropout_rate
        eps = config.eps
        
        ## Layernorm before attention and MLP because this stabilizes the training
        
        # Multi-head self-attention
        x = nn.LayerNorm()(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=num_heads, 
            qkv_features=qkv_dim, 
            dropout_rate=attn_dropout_rate,
            deterministic=config.deterministic,
        )(x, mask=None)
        # There could be a dropout layer here
        x = x + inputs
        
        # Feed-forward block
        y = nn.LayerNorm()(x)
        y = MlpBlock(config)(y)
        
        return x + y



