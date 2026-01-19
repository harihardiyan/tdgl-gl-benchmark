
import jax
import jax.numpy as jnp

def make_grid(nx: int, ny: int, Lx: float, Ly: float):
    """
    Create coordinate grids (X, Y) for a rectangular domain.
    """
    dx = Lx / nx
    dy = Ly / ny
    x = jnp.linspace(0.0, Lx - dx, nx)
    y = jnp.linspace(0.0, Ly - dy, ny)
    Y, X = jnp.meshgrid(y, x, indexing="ij")
    return X, Y

def set_seed(seed: int = 0):
    """
    Convenience wrapper to create a JAX PRNGKey.
    """
    return jax.random.PRNGKey(seed)
