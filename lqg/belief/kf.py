from jax import numpy as jnp, lax

from lqg.spec import LQGSpec


def forward(spec: LQGSpec, Sigma0: jnp.ndarray) -> jnp.ndarray:
    def loop(P_prev_post, step):
        A, F, V, W = step

        # P_prev_post - posterior covariance from previous time step

        # A - state transition matrix, i.e. deterministic evolution of state
        # F - observation model, i.e. how elements of state project to observations, identity in this case
        # V - dynamics noise covariance
        # W - observation noise covariance

        I = jnp.eye(A.shape[0])

        # Noise covariances (built from square-roots)
        Q = V @ V.T              # process noise covariance
        R = W @ W.T              # observation noise covariance

        # 1) Predict covariance
        P_prior = A @ P_prev_post @ A.T + Q

        # 2) Innovation covariance
        G = F @ P_prior @ F.T + R  # often called S

        # 3) Kalman gain: K = P_pred F^T G^{-1}
        # Avoid inverse: solve G^T X = (F P_pred)^T for X, then K = X^T
        # FP = F @ P_prior                    # (m x n)
        # K = jnp.linalg.solve(G.T, FP).T     # (n x m)
        K = P_prior @ F.T @ jnp.linalg.inv(G)

        # 4) Posterior covariance (simple form)
        P_curr_post = (I - K @ F) @ P_prior

        return P_curr_post, K

    _, K = lax.scan(loop, Sigma0,
                    (spec.A, spec.F, spec.V, spec.W))

    return K
