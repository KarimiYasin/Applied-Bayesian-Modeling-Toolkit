import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

samples_A = beta.rvs(41, 961, size=100000)
samples_B = beta.rvs(53, 949, size=100000)

diff = samples_B - samples_A

plt.figure(figsize=(10, 6))
plt.hist(diff, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='No Difference')
plt.title('Distribution of Difference (B - A)')
plt.xlabel('Delta (Improvement)')
plt.ylabel('Frequency')

loss_scenarios = diff[diff < 0]
expected_loss = np.abs(np.mean(loss_scenarios)) if len(loss_scenarios) > 0 else 0

prob_b_better = np.mean(diff > 0)

print(f"Probability B is better than A: {prob_b_better:.2%}")
print(f"Expected Loss (if we are wrong): {expected_loss:.5f}")

plt.legend()
plt.show()