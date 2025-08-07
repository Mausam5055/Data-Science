# 📊 Z-Test Assignment

This assignment demonstrates how to perform a **One-Sample Z-Test** using `statsmodels` in Python.

## 🎯 Objectives

| Objective | Covered |
|-----------|---------|
| Perform a one-sample Z-test on generated data | ✅ |
| Use statsmodels to test the mean of a sample against a known population mean | ✅ |
| Interpret z-score and p-value | ✅ |

---

## 🧪 Steps Performed

| Step | Description |
|------|-------------|
| 1 | Generate random sample data with known mean and std deviation |
| 2 | Define population mean to test against |
| 3 | Perform Z-test using `ztest()` from `statsmodels` |
| 4 | Print Z-score and P-value |
| 5 | Interpret results (significance test with α = 0.05) |
| 6 | (Optional) Visualize sample distribution and comparison with population mean |

---

## 📦 How to Run

```bash
pip install statsmodels matplotlib
python ztest_demo.py
```

---

## 📈 Interpretation Guide

- **Z-score**: How far the sample mean deviates from the population mean (in standard deviations)
- **P-value**: If less than 0.05 → result is statistically significant

---

**Author**: Mausam Kar  
**Topic**: Data Science - Z-Test (One Sample)
