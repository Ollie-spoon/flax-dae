import jax.numpy as jnp
from jax import jit

@jit
def abs_complex_log10(numbers):
    # Compute the complex logarithm
    absolutes = jnp.abs(numbers)  # Adding 0j to ensure complex type
    negative = ((1-numbers/absolutes))
    # Return the absolute value of the complex logarithm
    return jnp.log10(absolutes)+negative

# Example usage:
numbers = jnp.array([1, -10, 10, -100, 2, -2])
result = abs_complex_log10(numbers)
print(result)
