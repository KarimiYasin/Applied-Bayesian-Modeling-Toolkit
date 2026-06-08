import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

print("=" * 60)
print("QUESTION 5: FINAL VISUALIZATION")
print("=" * 60)

fig = plt.figure(figsize=(16, 7))

ax1 = fig.add_subplot(121)
im = ax1.imshow(posterior, extent=[0, 3, -2, 5], origin='lower',
                aspect='auto', cmap='hot', interpolation='bilinear')
ax1.set_xlabel('Slope (m)')
ax1.set_ylabel('Intercept (c)')
ax1.set_title('Left Panel: Final Posterior Distribution')
plt.colorbar(im, ax=ax1, label='Probability')

max_idx = np.unravel_index(posterior.argmax(), posterior.shape)
m_map = M[max_idx]
c_map = C[max_idx]

ax1.plot(m_true, c_true, 'b+', markersize=20, markeredgewidth=3, label='True parameters')
ax1.plot(m_map, c_map, 'r*', markersize=15, label='MAP estimate')
ax1.legend()

smoothed = gaussian_filter(posterior, sigma=1.0)
threshold = np.percentile(smoothed, 68)
ax1.contour(M, C, smoothed, levels=[threshold], colors='white', linewidths=2, linestyles='--')

# Right panel: Data space with sampled lines
ax2 = fig.add_subplot(122)

n_samples = 50
m_samples = []
c_samples = []

posterior_flat = posterior.flatten()
posterior_flat = posterior_flat / posterior_flat.sum()
indices = np.random.choice(len(posterior_flat), size=n_samples, p=posterior_flat)

for idx in indices:
    i, j = np.unravel_index(idx, M.shape)
    m_samples.append(M[i, j])
    c_samples.append(C[i, j])

ax2.scatter(x_data, y_data, color='red', s=120, label='Observed data',
            edgecolors='black', linewidth=2, zorder=5)

x_line = np.linspace(-2, 3, 100)
for m, c in zip(m_samples, c_samples):
    y_line = m * x_line + c
    ax2.plot(x_line, y_line, 'b-', alpha=0.1, linewidth=1)

y_true_line = m_true * x_line + c_true
ax2.plot(x_line, y_true_line, 'g--', linewidth=3, label='True line', color='green')

y_samples = np.array([m * x_line + c for m, c in zip(m_samples, c_samples)])
y_upper = np.percentile(y_samples, 97.5, axis=0)
y_lower = np.percentile(y_samples, 2.5, axis=0)
ax2.fill_between(x_line, y_lower, y_upper, alpha=0.2, color='blue', label='95% CI')

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Right Panel: Sampled Lines from Posterior')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-2.5, 3.5)
ax2.set_ylim(-2, 8)

plt.suptitle('Figure 5: Bayesian Linear Regression Results')
plt.tight_layout()
plt.show()

print(f"\nFinal MAP estimate: m = {m_map:.2f}, c = {c_map:.2f}")
print(f"True parameters: m = {m_true:.2f}, c = {c_true:.2f}")
print(f"Absolute error: delta_m = {abs(m_map - m_true):.3f}, delta_c = {abs(c_map - c_true):.3f}")
