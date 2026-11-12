
# PCA (Principal Component Analysis) 

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------
# STEP 1: Make some simple data
# ------------------------------------------
# We make 100 points with 2 features (x and y).
# The points go in a line shape, so x and y are related.
np.random.seed(0)                       # same random numbers every time
x = np.random.randn(100) * 3
y = 0.8 * x + np.random.randn(100)      # y follows x, plus a little noise
data = np.column_stack((x, y))          # shape: (100 rows, 2 columns)

# ------------------------------------------
# STEP 2: Move the data to the center (mean = 0)
# ------------------------------------------
mean = data.mean(axis=0)
centered = data - mean

# ------------------------------------------
# STEP 3: Make the covariance matrix
# ------------------------------------------
# It shows how the features change together.
cov_matrix = np.cov(centered, rowvar=False)
print("Covariance matrix:")
print(cov_matrix)

# ------------------------------------------
# STEP 4: Find eigenvalues and eigenvectors
# ------------------------------------------
# Eigenvector  = a direction (this is a Principal Component)
# Eigenvalue   = how much the data changes in that direction
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# ------------------------------------------
# STEP 5: Sort from biggest to smallest
# ------------------------------------------
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

# How much information does each component keep?
explained = eigenvalues / eigenvalues.sum() * 100
print("\nVariance explained (%):")
print(f"  PC1: {explained[0]:.1f}%")
print(f"  PC2: {explained[1]:.1f}%")

# ------------------------------------------
# STEP 6: Turn the data into the new axes
# ------------------------------------------
new_data = centered @ eigenvectors      # all components
pc1_only = new_data[:, 0]               # keep only PC1 (2D -> 1D)

# ------------------------------------------
# STEP 7: Show the results (pictures)
# ------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Picture 1: Original data with the PC directions
ax = axes[0]
ax.scatter(data[:, 0], data[:, 1], color="skyblue", edgecolor="gray")
colors = ["red", "green"]
for i in range(2):
    direction = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2
    ax.arrow(mean[0], mean[1], direction[0], direction[1],
             color=colors[i], width=0.08, head_width=0.5,
             length_includes_head=True, label=f"PC{i+1}")
ax.set_title("1. Original data + PC directions")
ax.set_xlabel("Feature X")
ax.set_ylabel("Feature Y")
ax.axis("equal")
ax.legend()

# Picture 2: Data on the new axes (PC1 and PC2)
ax = axes[1]
ax.scatter(new_data[:, 0], new_data[:, 1], color="orange", edgecolor="gray")
ax.set_title("2. Data on the NEW axes")
ax.set_xlabel("PC1 (most information)")
ax.set_ylabel("PC2 (less information)")
ax.axis("equal")

# Picture 3: How much information each PC keeps
ax = axes[2]
ax.bar(["PC1", "PC2"], explained, color=["red", "green"])
ax.set_title("3. Variance explained (%)")
ax.set_ylabel("Percent (%)")
for i, value in enumerate(explained):
    ax.text(i, value + 1, f"{value:.1f}%", ha="center")
ax.set_ylim(0, 110)

plt.tight_layout()
plt.savefig("pca_result.png")           # save the picture
plt.show()                              # show the picture
