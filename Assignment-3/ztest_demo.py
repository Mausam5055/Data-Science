import numpy as np
from statsmodels.stats.weightstats import ztest
import matplotlib.pyplot as plt

# Step 1: Generate sample data (e.g., students' test scores)
np.random.seed(42)  # For reproducibility
sample_data = np.random.normal(loc=70, scale=10, size=100)  # Mean=70, SD=10, n=100

# Step 2: Define the known population mean to test against
population_mean = 75

# Step 3: Perform the one-sample Z-test
z_score, p_value = ztest(sample_data, value=population_mean)

# Step 4: Print results
print("🔍 One-Sample Z-Test")
print("Z-score:", round(z_score, 3))
print("P-value:", round(p_value, 4))

# Step 5: Interpret the results
alpha = 0.05
if p_value < alpha:
    print("→ The sample mean is significantly different from the population mean.")
else:
    print("→ The sample mean is NOT significantly different from the population mean.")

# Step 6: Optional - Visualize the distribution
plt.hist(sample_data, bins=20, edgecolor='black')
plt.axvline(np.mean(sample_data), color='blue', linestyle='dashed', linewidth=2, label='Sample Mean')
plt.axvline(population_mean, color='red', linestyle='solid', linewidth=2, label='Population Mean')
plt.title("Sample Data Distribution")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.legend()
plt.show()
