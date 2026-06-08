import numpy as np
from scipy.stats import norm

np.random.seed(0)
x = np.linspace(-2, 3, 10)
y = 1.5 * x + 2.0 + np.random.normal(0, 0.5, len(x))
sigma = 0.5

m_vals = np.linspace(0, 3, 50)
c_vals = np.linspace(-2, 5, 50)
M, C = np.meshgrid(m_vals, c_vals)

posterior = np.ones_like(M)
posterior /= posterior.sum()

print("Step |   m_MAP   c_MAP   |  MaxProb   Entropy")
print("-" * 55)

for step, (xi, yi) in enumerate(zip(x, y), 1):
    likelihood = norm.pdf(yi, M * xi + C, sigma)
    posterior *= likelihood
    posterior /= posterior.sum()

    idx = np.unravel_index(np.argmax(posterior), posterior.shape)
    m_map, c_map = M[idx], C[idx]
    entropy = -np.sum(posterior * np.log(posterior + 1e-12))

    print(f"{step:>4} | {m_map:7.3f}  {c_map:7.3f} | "
          f"{posterior.max():8.6f}  {entropy:8.3f}")

m_mean = np.sum(M * posterior)
c_mean = np.sum(C * posterior)

print("\nFinal Estimates:")
print(f"MAP estimate     : m = {m_map:.3f}, c = {c_map:.3f}")
print(f"Posterior mean   : m = {m_mean:.3f}, c = {c_mean:.3f}")
print("True parameters  : m = 1.500, c = 2.000")

# additional data
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(posterior, extent=[0, 3, -2, 5], origin='lower', cmap='hot')
plt.plot(1.5, 2.0, 'b+', markersize=15)
plt.colorbar()
plt.title('Final Posterior')

plt.subplot(1, 2, 2)
plt.scatter(x, y, c='red')
x_line = np.linspace(-2, 3)
plt.plot(x_line, m_map * x_line + c_map, 'b-', label='MAP')
plt.plot(x_line, 1.5 * x_line + 2.0, 'g--', label='True')
plt.legend()
plt.title('Data & MAP Line')
plt.tight_layout()
plt.show()
