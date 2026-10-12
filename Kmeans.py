import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Generate sample data (replace this with your own dataset)
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

# 2. Scale the features (important for distance-based algorithms)
X_scaled = StandardScaler().fit_transform(X)

# 3. Compute inertia (WCSS) for different values of k
k_range = range(1, 11)
inertias = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# 4. Plot the Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker="o", linestyle="-", color="royalblue")
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.xticks(k_range)
plt.grid(True, alpha=0.3)
plt.show()

# 5. Fit the final model with the chosen k (read from the elbow plot)
optimal_k = 4
final_model = KMeans(n_clusters=optimal_k, init="k-means++", n_init=10, random_state=42)
labels = final_model.fit_predict(X_scaled)
centroids = final_model.cluster_centers_

# 6. Visualize the resulting clusters
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="viridis", s=30, alpha=0.8)
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200, label="Centroids")
plt.title(f"K-Means Clustering (k = {optimal_k})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()