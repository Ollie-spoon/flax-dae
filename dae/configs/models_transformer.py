from jax import jit, vmap, random, numpy as jnp
from flax import linen as nn

class TransformerConfig:
    
    # Tokenizer parameters
    min_tokens: int = 7
    max_tokens: int = 18
    token_dim: int = 64
    overlap_per_token: int = 8
    
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
    
    def __call__(self, inputs):
        """Applies Transformer Tokenizer module.
        
        Tokenize input sequence into overlapping tokens.
        """
        config = self.config
        min_tokens = config.min_tokens
        max_tokens = config.max_tokens
        token_dim = config.token_dim
        overlap_per_token = config.overlap_per_token
        
        # Clip input sequence to fit tokens
        token_clip_len = (inputs.shape[1] - token_dim) % (token_dim - overlap_per_token)
        seq_len = inputs.shape[1] - token_clip_len
        inputs = inputs[:, :seq_len]
        
        # Calculate number of tokens
        num_tokens = (seq_len - token_dim)//(token_dim - overlap_per_token) + 1
        num_tokens = min(num_tokens, max_tokens)
        
        # Calculate number of padding tokens
        pad_len = max(0, max_tokens - num_tokens)
        
        # Pad input sequence
        inputs = nn.pad(inputs, [(0, 0), (0, pad_len)])
        
        # Tokenize input sequence
        tokens = nn.Conv(
            features=token_dim,
            kernel_size=(1, min_tokens),
            strides=(1, overlap_per_token),
            padding='VALID',
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )(inputs)
        
        return tokens

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


