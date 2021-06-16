import matplotlib.pyplot as plt
import numpy as np
from d2l import torch as d2l
import torch

# Vanishing gradients tanh/ReLU plot
x = torch.arange(-8, 8, 0.01)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(x, torch.sigmoid(x))
ax1.set(xlabel="x", ylabel="Sigmoid(x)")
ax1.set_title("Sigmoid activation function")
ax1.annotate('Small gradient region', (7, 1), (-0.5, 0.1),
             arrowprops=dict(facecolor='black', width=0.5, headwidth=9, shrink=0.05))
ax2.plot(x, torch.relu(x))
ax2.set(xlabel="x", ylabel="ReLU(x)")
ax2.set_title("ReLU activation function")
ax2.annotate('Corner point', (0, 0), (-6, 2),
             arrowprops=dict(facecolor='black', width=0.5, headwidth=9, shrink=0.05))
plt.show()
fig.savefig("../Figures/Vanishing_gradients.svg")
