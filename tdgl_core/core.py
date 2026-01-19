# tdgl_core/core.py
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jax import tree_util

# ============================================================
# 1. PyTrees: State & Params
# ============================================================

@tree_util.register_pytree_node_class
@dataclass
class State:
    psi: jnp.ndarray  # (ny, nx, 2) real-imag
    A: jnp.ndarray    # (ny, nx, 2)
    mu: jnp.ndarray   # (ny, nx)

    def tree_flatten(self):
        return (self.psi, self.A, self.mu), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        psi, A, mu = children
        return cls(psi=psi, A=A, mu=mu)


@tree_util.register_pytree_node_class
@dataclass
class Params:
    nx: int
    ny: int
    dx: float
    dy: float
    kappa: float
    gamma_psi: float
    gamma_A: float
    sigma_n: float
    J_ext: float
    a: jnp.ndarray      # GL coefficient a(x,y)
    H_ext: jnp.ndarray  # external field Bz(x,y)

    def tree_flatten(self):
        children = (self.nx, self.ny, self.dx, self.dy,
                    self.kappa, self.gamma_psi, self.gamma_A,
                    self.sigma_n, self.J_ext, self.a, self.H_ext)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        nx, ny, dx, dy, kappa, gamma_psi, gamma_A, sigma_n, J_ext, a, H_ext = children
        return cls(nx=nx, ny=ny, dx=dx, dy=dy,
                   kappa=kappa, gamma_psi=gamma_psi, gamma_A=gamma_A,
                   sigma_n=sigma_n, J_ext=J_ext, a=a, H_ext=H_ext)


# ============================================================
# 2. Complex helpers
# ============================================================

def to_complex(phi: jnp.ndarray) -> jnp.ndarray:
    """Convert (ny, nx, 2) real-imag to complex (ny, nx)."""
    return phi[..., 0] + 1j * phi[..., 1]


def to_real(z: jnp.ndarray) -> jnp.ndarray:
    """Convert complex (ny, nx) to (ny, nx, 2) real-imag."""
    return jnp.stack([jnp.real(z), jnp.imag(z)], axis=-1)


# ============================================================
# 3. Differential operators (Neumann via mirroring)
# ============================================================

def grad_x_neumann(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Centered finite difference with Neumann BC in x."""
    fL = f[:, 0:1]
    fR = f[:, -1:]
    f_ext = jnp.concatenate([fL, f, fR], axis=1)
    df = (f_ext[:, 2:] - f_ext[:, :-2]) / (2.0 * dx)
    return df


def grad_y_neumann(f: jnp.ndarray, dy: float) -> jnp.ndarray:
    """Centered finite difference with Neumann BC in y."""
    fB = f[0:1, :]
    fT = f[-1:, :]
    f_ext = jnp.concatenate([fB, f, fT], axis=0)
    df = (f_ext[2:, :] - f_ext[:-2, :]) / (2.0 * dy)
    return df


def curl_A_neumann(A: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Compute Bz = ∂x Ay − ∂y Ax with Neumann BC."""
    Ax = A[..., 0]
    Ay = A[..., 1]
    return grad_x_neumann(Ay, dx) - grad_y_neumann(Ax, dy)


# ============================================================
# 4. Gauge-invariant covariant gradient (link variables)
# ============================================================

def covariant_grad_psi_link(psi_c: jnp.ndarray,
                            A: jnp.ndarray,
                            dx: float,
                            dy: float):
    """
    Gauge-invariant covariant gradient using link variables.
    psi_c: complex field (ny, nx)
    A: vector potential (ny, nx, 2)
    """
    Ax = A[..., 0]
    Ay = A[..., 1]

    # x-direction
    psiL = psi_c[:, 0:1]
    psiR = psi_c[:, -1:]
    psi_x_ext = jnp.concatenate([psiL, psi_c, psiR], axis=1)

    AxL = Ax[:, 0:1]
    AxR = Ax[:, -1:]
    Ax_ext = jnp.concatenate([AxL, Ax, AxR], axis=1)

    Ax_plus = Ax_ext[:, 1:-1]
    Ax_minus = Ax_ext[:, 1:-1]

    Ux_plus = jnp.exp(-1j * Ax_plus * dx)
    Ux_minus = jnp.exp(+1j * Ax_minus * dx)

    psi_plus_x = psi_x_ext[:, 2:]
    psi_minus_x = psi_x_ext[:, :-2]

    Dx_psi = (Ux_plus * psi_plus_x - Ux_minus * psi_minus_x) / (2.0 * dx)

    # y-direction
    psiB = psi_c[0:1, :]
    psiT = psi_c[-1:, :]
    psi_y_ext = jnp.concatenate([psiB, psi_c, psiT], axis=0)

    AyB = Ay[0:1, :]
    AyT = Ay[-1:, :]
    Ay_ext = jnp.concatenate([AyB, Ay, AyT], axis=0)

    Ay_plus = Ay_ext[1:-1, :]
    Ay_minus = Ay_ext[1:-1, :]

    Uy_plus = jnp.exp(-1j * Ay_plus * dy)
    Uy_minus = jnp.exp(+1j * Ay_minus * dy)

    psi_plus_y = psi_y_ext[2:, :]
    psi_minus_y = psi_y_ext[:-2, :]

    Dy_psi = (Uy_plus * psi_plus_y - Uy_minus * psi_minus_y) / (2.0 * dy)

    return Dx_psi, Dy_psi


# ============================================================
# 5. Energy functional
# ============================================================

def energy_density(state: State, params: Params) -> jnp.ndarray:
    """
    Ginzburg–Landau energy density:
    f = a|ψ|² + ½|ψ|⁴ + |Dψ|² + κ²(B − H_ext)²
    """
    psi_c = to_complex(state.psi)
    A = state.A
    abs2 = jnp.abs(psi_c)**2

    term_local = params.a * abs2 + 0.5 * abs2**2

    Dx, Dy = covariant_grad_psi_link(psi_c, A, params.dx, params.dy)
    term_kin = jnp.abs(Dx)**2 + jnp.abs(Dy)**2

    Bz = curl_A_neumann(A, params.dx, params.dy)
    term_mag = params.kappa**2 * (Bz - params.H_ext)**2

    return term_local + term_kin + term_mag


def total_energy(state: State, params: Params) -> jnp.ndarray:
    """Integrate energy density over the domain."""
    f = energy_density(state, params)
    return jnp.sum(f) * params.dx * params.dy


# ============================================================
# 6. μ-solver: Poisson (SOR)
# ============================================================

@jax.jit
def solve_mu_poisson(mu0: jnp.ndarray,
                     params: Params,
                     n_iter: int = 80,
                     omega: float = 1.6) -> jnp.ndarray:
    """
    Solve Poisson-like equation for μ with SOR and Neumann-like BC
    enforcing external current J_ext.
    """
    dx = params.dx
    dy = params.dy
    sigma = params.sigma_n
    Jext = params.J_ext

    def body(i, mu):
        mu_new = mu

        coef = 1.0 / (2.0/dx**2 + 2.0/dy**2)
        center = (mu[1:-1, 2:] + mu[1:-1, :-2]) / dx**2 \
               + (mu[2:, 1:-1] + mu[:-2, 1:-1]) / dy**2
        mu_relaxed = coef * center
        mu_old = mu[1:-1, 1:-1]
        mu_new = mu_new.at[1:-1, 1:-1].set(mu_old + omega * (mu_relaxed - mu_old))

        dmu = -Jext * dx / sigma
        mu_new = mu_new.at[:, 0].set(mu_new[:, 1] + dmu)
        mu_new = mu_new.at[:, -1].set(mu_new[:, -2] - dmu)

        return mu_new

    mu_final = jax.lax.fori_loop(0, n_iter, body, mu0)
    return mu_final


# ============================================================
# 7. TDGL evolution
# ============================================================

@jax.jit
def tdgl_rhs(state: State, params: Params) -> State:
    """
    Compute TDGL right-hand side:
    dψ/dt = -γ_ψ δF/δψ*
    dA/dt = -γ_A (δF/δA + J_n)
    """
    def F(psi, A):
        return total_energy(State(psi=psi, A=A, mu=state.mu), params)

    gpsi, gA = jax.grad(F, argnums=(0, 1))(state.psi, state.A)

    grad_mu_x = grad_x_neumann(state.mu, params.dx)
    grad_mu_y = grad_y_neumann(state.mu, params.dy)
    Jn_x = -params.sigma_n * grad_mu_x
    Jn_y = -params.sigma_n * grad_mu_y

    dA_dt = -params.gamma_A * (gA + jnp.stack([Jn_x, Jn_y], axis=-1))
    dpsi_dt = -params.gamma_psi * gpsi

    return State(psi=dpsi_dt,
                 A=dA_dt,
                 mu=jnp.zeros_like(state.mu))


@jax.jit
def tdgl_step(state: State,
              params: Params,
              dt: float,
              n_mu_iter: int = 80) -> State:
    """
    One TDGL time step:
    - Relax μ via Poisson solver
    - Update ψ and A via gradient descent on F
    """
    mu_new = solve_mu_poisson(state.mu, params, n_iter=n_mu_iter)
    rhs = tdgl_rhs(State(state.psi, state.A, mu_new), params)
    return State(
        psi = state.psi + dt * rhs.psi,
        A   = state.A   + dt * rhs.A,
        mu  = mu_new
    )


# ============================================================
# 8. Geometry & initialization
# ============================================================

def make_uniform_field(nx: int, ny: int, B0: float = 0.0) -> jnp.ndarray:
    """Uniform external field H_ext = B0."""
    return jnp.ones((ny, nx)) * B0


def make_uniform_a(nx: int, ny: int, a0: float = -1.0) -> jnp.ndarray:
    """Uniform GL coefficient a(x,y) = a0."""
    return jnp.ones((ny, nx)) * a0


def init_params_basic(nx: int = 64,
                      ny: int = 64,
                      Lx: float = 20.0,
                      Ly: float = 20.0,
                      kappa: float = 2.0,
                      gamma_psi: float = 1.0,
                      gamma_A: float = 1.0,
                      sigma_n: float = 1.0,
                      J_ext: float = 0.0,
                      B0: float = 0.1,
                      a0: float = -1.0) -> Params:
    """
    Basic parameter initialization for a rectangular domain.
    """
    dx = Lx / nx
    dy = Ly / ny
    a = make_uniform_a(nx, ny, a0=a0)
    H_ext = make_uniform_field(nx, ny, B0=B0)
    return Params(nx=nx, ny=ny, dx=dx, dy=dy,
                  kappa=kappa,
                  gamma_psi=gamma_psi,
                  gamma_A=gamma_A,
                  sigma_n=sigma_n,
                  J_ext=J_ext,
                  a=a,
                  H_ext=H_ext)


def init_state_random(params: Params,
                      noise: float = 0.3,
                      key: jax.random.PRNGKey = jax.random.PRNGKey(0)) -> State:
    """
    Random initial ψ with small amplitude and phase noise,
    zero A and μ.
    """
    ny, nx = params.ny, params.nx
    amp = 1.0 + 0.1 * jax.random.normal(key, (ny, nx))
    phase = jax.random.uniform(key, (ny, nx), minval=-jnp.pi, maxval=jnp.pi)
    psi0 = amp * jnp.exp(1j * phase)
    psi0 = psi0 + noise * (jax.random.normal(key, (ny, nx)) +
                           1j * jax.random.normal(key, (ny, nx))) * 0.1
    psi0 = to_real(psi0.astype(jnp.complex64))

    mu0 = jnp.zeros((ny, nx))
    A0 = jnp.zeros((ny, nx, 2))
    return State(psi=psi0, A=A0, mu=mu0)
