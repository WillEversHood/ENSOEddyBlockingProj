import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Simulate a time series for T and h with autocorrelation
np.random.seed(0)
n = 100
time = np.arange(n)

# Simulating correlated time series (T and h)
T = np.sin(0.1 * time) + np.random.normal(0, 0.1, n)  # T is a sinusoidal series
h = np.cos(0.1 * time) + np.random.normal(0, 0.1, n)  # h is a sinusoidal series

# Stack T and h into a 2D array (each row is [T_i, h_i])
data = np.vstack([T, h]).T

# Define a grid over which to evaluate the joint PDF
T_grid, h_grid = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
grid_points = np.vstack([T_grid.ravel(), h_grid.ravel()]).T

# Fit the KDE model
kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(data)

# Evaluate the KDE model over the grid
log_density = kde.score_samples(grid_points)
density = np.exp(log_density).reshape(T_grid.shape)

# Plot the estimated joint PDF
plt.figure(figsize=(8, 6))
plt.contourf(T_grid, h_grid, density, cmap='Blues')
plt.title('Joint PDF of T and h (Time Series KDE)')
plt.xlabel('T')
plt.ylabel('h')
plt.colorbar(label='Density')
plt.show()
