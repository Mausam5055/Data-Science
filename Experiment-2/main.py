# ============================================
# Experiment-2: Singular Value Decomposition (SVD)
# ============================================
# This experiment demonstrates Singular Value Decomposition using NumPy
# It decomposes a matrix and reconstructs it from its components

import numpy as np

# Create a random matrix for demonstration
np.random.seed(42)
matrix = np.random.random((5, 3))

# Perform Singular Value Decomposition
U, S, Vt = np.linalg.svd(matrix, full_matrices=True)

# Reconstruct the original matrix from the SVD components
reconstructed_matrix = np.dot(U, np.dot(np.diag(S), Vt))

# Print the original matrix
print("Original Matrix:")
print(matrix)

# Print the decomposed components
print("\nU matrix:")
print(U)
print("\nS matrix (diagonal matrix):")
print(np.diag(S))
print("\nVt matrix:")
print(Vt)

# Print the reconstructed matrix
print("\nReconstructed Matrix:")
print(reconstructed_matrix)