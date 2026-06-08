import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
m_true, c_true, sigma = 1.5, 2.0, 1.0

x = np.random.uniform(-2, 3, 10)
y_true = m_true * x + c_true
noise = np.random.normal(0, sigma, 10)
y = y_true + noise

print("Ground Truth: y = 1.5x + 2.0, sigma = 1.0\n")
print("Index     x          y          y_true     Noise")
print("-" * 55)
for i, (xi, yi, yti, ni) in enumerate(zip(x, y, y_true, noise), 1):
    print(f"{i:2d}     {xi:8.4f}   {yi:8.4f}   {yti:8.4f}   {ni:8.4f}")

print(f"\nNoise mean: {np.mean(noise):.4f}, Noise std: {np.std(noise):.4f}")
print(f"Correlation: {np.corrcoef(x, y)[0, 1]:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', s=100, label='Data with noise')
plt.plot(np.linspace(-2, 3, 100), m_true * np.linspace(-2, 3, 100) + c_true, 'g-', linewidth=2, label='True line')
plt.xlabel('x');
plt.ylabel('y');
plt.title('Synthetic Data: y = 1.5x + 2.0 + noise')
plt.legend();
plt.grid(alpha=0.3);
plt.xlim(-2.5, 3.5);
plt.ylim(-2, 8)
plt.tight_layout()
plt.show()