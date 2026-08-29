import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# 1. Load the data (assuming no headers in the Excel file)
df = pd.read_excel('amorphous.xlsx', header=None)
positions = df.values  # Shape: (1000, 3)

# 2. Define Simulation Box Parameters
L = 30.0  # 3 nm box length = 30 Angstroms
N = len(positions)
V = L**3
rho = N / V  # Number density

# 3. Calculate Pairwise Distances with Periodic Boundary Conditions (PBC)
# Calculate differences between all pairs
diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
# Apply minimum image convention for the 30 Å box
diff = diff - L * np.round(diff / L)
# Compute Euclidean distance
dist = np.sqrt(np.sum(diff**2, axis=-1))

# Extract unique pairs (upper triangle of the distance matrix)
distances = dist[np.triu_indices(N, k=1)]

# 4. Compute the Histogram
dr = 0.05
r_max = L / 2.0  # Maximum radius is half the box length (15 Å)
bins = np.arange(0, r_max + dr, dr)
r_centers = 0.5 * (bins[1:] + bins[:-1])

hist, _ = np.histogram(distances, bins=bins)
hist = hist * 2.0  # Multiply by 2 to account for both i->j and j->i directions

# 5. Calculate g(r) and RDF(r)
shell_volumes = (4.0 / 3.0) * np.pi * (bins[1:]**3 - bins[:-1]**3)

g_r = hist / (N * shell_volumes * rho)
rdf_r = 4.0 * np.pi * (r_centers**2) * rho * g_r

# 6. Determine the Nearest-Neighbor Coordination Number
# Find the first peak (expected around 2.35 - 2.40 Å for Si)
first_peak_idx = np.argmax(rdf_r[r_centers < 3.0])

# Find the first minimum after the peak to define the first coordination shell
search_range = (r_centers > r_centers[first_peak_idx]) & (r_centers < 3.5)
first_min_idx_relative = np.argmin(rdf_r[search_range])
first_min_idx = np.where(search_range)[0][first_min_idx_relative]
r_min = r_centers[first_min_idx]

# Integrate RDF(r) up to the first minimum using Simpson's rule
coord_num = simpson(rdf_r[:first_min_idx+1], x=r_centers[:first_min_idx+1])

print(f"Density (\u03c1): {rho:.4f} atoms/\u00c5\u00b3")
print(f"First peak at: {r_centers[first_peak_idx]:.2f} \u00c5")
print(f"First minimum (Shell boundary) at: {r_min:.2f} \u00c5")
print(f"Nearest-Neighbor Coordination Number: {coord_num:.4f}")

# 7. Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# RDF Plot
axes[0].plot(r_centers, rdf_r, label='RDF(r)', color='tab:blue')
axes[0].fill_between(r_centers[:first_min_idx+1], rdf_r[:first_min_idx+1], color='tab:blue', alpha=0.3)
axes[0].axvline(r_min, color='red', linestyle='--', label=f'r_min = {r_min:.2f} $\\AA$\nCN = {coord_num:.2f}')
axes[0].set_xlim(0, 10)
axes[0].set_ylim(0, max(rdf_r[r_centers < 10]) * 1.1)
axes[0].set_title('Radial Distribution Function (RDF)')
axes[0].set_xlabel('Distance ($\\AA$)')
axes[0].set_ylabel('RDF(r)')
axes[0].legend()

# g(r) Plot
axes[1].plot(r_centers, g_r, label='g(r)', color='tab:blue')
axes[1].axhline(1, color='tab:orange', alpha=0.5, label='g(r) = 1 (Ideal Gas)')
axes[1].set_xlim(0, 10)
axes[1].set_ylim(0, max(g_r[r_centers < 10]) * 1.1)
axes[1].set_title('Pair Correlation Function (PCF)')
axes[1].set_xlabel('Distance ($\\AA$)')
axes[1].set_ylabel('g(r)')
axes[1].legend()

plt.tight_layout()
output_path = 'rdf_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved to: {output_path}")