# ============================================
# Experiment-4: T-Test Analysis with SciPy
# ============================================
# This experiment demonstrates how to perform a two-sample t-test using SciPy
# It generates two samples and compares their means statistically

import numpy as np
from scipy import stats

# Generate two independent samples for demonstration
np.random.seed(42)
sample1 = np.random.normal(loc=5, scale=2, size=30)
sample2 = np.random.normal(loc=7, scale=2, size=30)

# Perform a two-sample t-test
t_statistic, p_value = stats.ttest_ind(sample1, sample2)

# Set the significance level (alpha)
alpha = 0.05

# Print the results of the t-test
print(f'T-statistic: {t_statistic}')
print(f'P-value: {p_value}')

# Check if the null hypothesis can be rejected
if p_value < alpha:
    print(f'Reject the null hypothesis at alpha = {alpha}')
else:
    print(f'Fail to reject the null hypothesis at alpha = {alpha}')