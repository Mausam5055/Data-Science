# 📊 T-Test Analysis using SciPy

This project demonstrates how to conduct **Independent** and **Paired T-tests** using Python's `scipy.stats` module to compare two groups for statistical significance.

---

## 🎯 Objectives

| Task | Description |
|------|-------------|
| ✅ Independent T-Test | Compare two **independent groups** |
| ✅ Paired T-Test | Compare two **related groups** |
| ✅ P-value Analysis | Determine if the difference is statistically significant |

---

## 📘 T-Test Types Explained

| Test Type | Use When... | Example |
|-----------|-------------|---------|
| **Independent** | Two unrelated groups | Test scores from Class A and Class B |
| **Paired** | Same group, tested twice | Scores **before and after** training |

---

## 🧪 Implementation Summary

### ✅ Independent T-Test

```python
from scipy import stats

group1 = [23, 21, 18, 25, 30]
group2 = [31, 29, 35, 32, 30]

t_stat, p_val = stats.ttest_ind(group1, group2)
print("T-statistic:", t_stat, "P-value:", p_val)
```

---

### ✅ Paired T-Test

```python
from scipy import stats

before = [88, 92, 94, 90, 91]
after  = [85, 89, 92, 87, 90]

t_stat, p_val = stats.ttest_rel(before, after)
print("T-statistic:", t_stat, "P-value:", p_val)
```

---

## 📈 Interpreting P-values

| P-value | Meaning |
|---------|---------|
| < 0.05 | Statistically significant (reject null hypothesis) |
| ≥ 0.05 | Not significant (fail to reject null) |

---

## 🚀 How to Run

```bash
pip install scipy
python ttest_demo.py
```

---

## 🗂️ Files

| File | Description |
|------|-------------|
| `ttest_demo.py` | Contains both independent and paired T-tests |
| `README.md` | Documentation and explanation |