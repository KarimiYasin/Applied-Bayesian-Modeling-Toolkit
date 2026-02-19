import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, beta

n_trials = 250
observed_heads = 140

samples_fair = binom.rvs(n=n_trials, p=0.5, size=1000)

p_samples = beta.rvs(141, 111, size=1000)

samples_bayesian = binom.rvs(n=n_trials, p=p_samples)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.hist(samples_fair, bins=30, color='lightgray', edgecolor='black')
ax1.axvline(observed_heads, color='red', linestyle='--', label=f'Observed: {observed_heads}')
ax1.set_title("Model 1: Fair Coin (Fixed p=0.5)")
ax1.set_xlabel("Number of Heads")
ax1.legend()

ax2.hist(samples_bayesian, bins=30, color='skyblue', edgecolor='black')
ax2.axvline(observed_heads, color='red', linestyle='--', label=f'Observed: {observed_heads}')
ax2.set_title("Model 2: Bayesian Learner (Updating p)")
ax2.set_xlabel("Number of Heads")
ax2.legend()

plt.tight_layout()
plt.show()

p_val = np.mean(samples_fair >= observed_heads)
print(f"Probability of seeing 140 or more heads in Model 1: {p_val:.4f}")