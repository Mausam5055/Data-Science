from scipy import stats

# Independent T-Test: Comparing two unrelated groups
group1 = [23, 21, 18, 25, 30]
group2 = [31, 29, 35, 32, 30]

t_stat_ind, p_val_ind = stats.ttest_ind(group1, group2)
print("🔹 Independent T-Test")
print("T-statistic:", t_stat_ind)
print("P-value:", p_val_ind)
print("-" * 40)

# Paired T-Test: Comparing scores from the same group (before vs after)
before = [88, 92, 94, 90, 91]
after  = [85, 89, 92, 87, 90]

t_stat_rel, p_val_rel = stats.ttest_rel(before, after)
print("🔹 Paired T-Test")
print("T-statistic:", t_stat_rel)
print("P-value:", p_val_rel)