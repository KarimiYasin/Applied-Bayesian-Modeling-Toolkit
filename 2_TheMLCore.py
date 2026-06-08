import numpy as np
from scipy.stats import norm

print("="*60)
print("LIKELIHOOD FUNCTION (sigma = 1.0)")
print("="*60)

m_true, c_true, sigma = 1.5, 2.0, 1.0
x_obs, y_obs = -0.1273, 1.5743

def likelihood(x, y, m, c, sigma):
    return norm.pdf(y, loc=m*x + c, scale=sigma)

hypotheses = [(1.5, 2.0, "Good"), (1.4, 2.1, "Decent"), (0.5, 3.0, "Bad")]

print(f"\nPoint: x={x_obs:.4f}, y={y_obs:.4f}")
print(f"sigma = {sigma}")
print("-"*60)
for m, c, label in hypotheses:
    like = likelihood(x_obs, y_obs, m, c, sigma)
    error = y_obs - (m*x_obs + c)
    print(f"{label}: m={m}, c={c}, error={error:.4f}, likelihood={like:.6f}")

m_range = np.linspace(0, 3, 50)
c_range = np.linspace(-2, 5, 50)
M, C = np.meshgrid(m_range, c_range)

like_grid = likelihood(x_obs, y_obs, M, C, sigma)
max_idx = np.unravel_index(like_grid.argmax(), like_grid.shape)

print("\n" + "-"*60)
print(f"Max likelihood: m={M[max_idx]:.2f}, c={C[max_idx]:.2f}")
print(f"True values:    m={m_true:.2f}, c={c_true:.2f}")
print(f"sigma used: {sigma}")


