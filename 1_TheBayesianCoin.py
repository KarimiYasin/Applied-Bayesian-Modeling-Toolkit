import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

true_p = 0.7
N = 100
np.random.seed(42)

alpha = 1
beta_param = 1

posterior_history = [(alpha, beta_param)]

for i in range(N):
    toss = np.random.rand() < true_p
    if toss:
        alpha += 1
    else:
        beta_param += 1
    posterior_history.append((alpha, beta_param))

p_vals = np.linspace(0, 1, 500)

plt.figure(figsize=(8, 5))

for k in [0, 5, 100]:
    a, b = posterior_history[k]
    plt.plot(p_vals, beta.pdf(p_vals, a, b), label=f"{k} tosses")

plt.axvline(true_p, linestyle='--', label="True p = 0.7")

plt.title("Bayesian Learning of Coin Bias")
plt.xlabel("Probability of Heads")
plt.ylabel("Density")
plt.legend()
plt.show()
