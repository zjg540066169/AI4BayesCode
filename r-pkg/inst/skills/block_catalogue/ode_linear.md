## ode_linear.hpp (exact solution of constant-coefficient linear ODE systems)

Header-only, namespace `AI4BayesCode::ode`, beside `ode_rk45.hpp`. For
    y'(t) = A(theta) y(t) + b(theta),   A and b constant in t,
the solution is a matrix exponential, and this header evaluates it EXACTLY
with ANALYTIC sensitivities. No step size, no tolerance, no integration error.

- `linear(A, b, y0, ts)` -- trajectory, `n_times x n`. Row i = y(ts[i]);
  ts[0] is t0 and row 0 equals y0, the same layout as `rk45()`.
- `linear_sens(A, b, y0, ts, dA, db, dy0)` -- trajectory plus dy/dtheta.
  `dA[j]`, `db[j]`, `dy0[j]` are the derivatives of A, b, y0 with respect
  to theta[j]. Returns `ode::rk45_sens_result`, so `ode::sens_chain(res,
  dlp_dy)` contracts it into the gradient exactly as for rk45.
- `linear_overflows(A, b, y0, t_span)` -- predicts a non-finite result
  before it happens, from all of A, b, y0 and the horizon.

WHEN. Any model whose state equation is linear in the state with
t-independent coefficients. Decide it from the equations, not from the
field the model comes from. The inhomogeneous term b is
handled by the augmented matrix [[A, b], [0, 0]], so A may be singular.
This is the REQUIRED path for such a model; `rk45` on a linear system is a
validator failure (see `codegen_priors.md`, "ODE models: FIRST decide
whether the system is linear").

HOW IT IS COMPUTED.
- n == 2: a closed form worked entirely in w = s^2 t^2, where
  s^2 = ((a11 - a22)/2)^2 + a12 a21 is the squared eigenvalue half-gap.
  s itself is never formed: ds/dtheta diverges at s = 0 while dw/dtheta is
  a polynomial, so the apparent s -> 0 singularity is a coordinate artefact
  and the code has none. w < 0 (complex eigenvalues) takes the cos / sinc
  branch of the same even functions. The inhomogeneous part uses divided
  differences of exp, never A^{-1}.
- other n: scaling-and-squaring Pade-13 exponential of the augmented
  matrix (Higham 2005); sensitivities from the block-triangular identity
  exp([[A, E], [0, A]] t), whose upper-right block is the directional
  derivative in the direction E (Van Loan 1978).
- The two paths are independent derivations and are tested against each
  other on n == 2.

LIMITATION (the one that is real). A component whose true value is of
order e^{x_small} is reconstructed from terms of order e^{x_big} and loses
accuracy like eps * e^{2q}, q the eigenvalue half-gap times t. Measured:
relative error 6e-16 at 2q = 2, 6e-9 at 2q = 20, order 1 at 2q = 40. This
is a property of ANY closed form for the sum of two exponentials, not of
this implementation; it bites only when a dominated component is itself
the quantity of interest over a horizon of many decay times.

TESTS. `tests/test_ode_linear.cpp`: T0 sanity (paths agree, t = 0, n = 1,
A = 0), T1 parity vs rk45 at 1e-12 and sensitivities vs Richardson FD per
eigenvalue family with a conditioning floor, T2 recovery on a linear ODE
model, T4 stress (s -> 0 through exactly 0, stiff to 1e8, overflow
prediction, two cancellation matrices with a12 a21 = -d^2 to rounding --
one with a11 - a22 exact and one inexact -- and y0, b and directions along
the null direction of A - m I, each for both y and S against a pinned
120-digit reference), and the Check #12 AD-twin of the hand-written
sensitivities. T3 cross-chain R-hat is exercised end-to-end
by an end-to-end comparison rather than in the unit test.
