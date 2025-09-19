# ============================================
# Experiment-9: DBSCAN Clustering on Customer Data
# ============================================
# This experiment demonstrates DBSCAN clustering on customer data
# It identifies clusters and outliers in customer spending patterns

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Note: This code requires a 'Wall_Customers.csv' file which is not included in the repository
# df = pd.read_csv('Wall_Customers.csv')
# X_train = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# For demonstration purposes, we'll create sample data since the CSV is missing
np.random.seed(42)
n_customers = 200
sample_data = {
    'Age': np.random.randint(18, 70, n_customers),
    'Annual Income (k$)': np.random.randint(15, 140, n_customers),
    'Spending Score (1-100)': np.random.randint(1, 100, n_customers)
}
df = pd.DataFrame(sample_data)
X_train = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

clustering = DBSCAN(eps=12.5, min_samples=4).fit(X_train)
DBSCAN_dataset = X_train.copy()
DBSCAN_dataset.loc[:,'Cluster'] = clustering.labels_
DBSCAN_dataset.Cluster.value_counts().to_frame()

outliers = DBSCAN_dataset[DBSCAN_dataset['Cluster']==-1]

fig2, axes = plt.subplots(1,2,figsize=(12,5))

sns.scatterplot('Annual Income (k$)', 'Spending Score (1-100)',
data=DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1],
hue='Cluster', ax=axes[0], palette='Set2',
legend='full', s=200)

sns.scatterplot('Age', 'Spending Score (1-100)',
data=DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1],
hue='Cluster', palette='Set2', ax=axes[1],
legend='full', s=200)

axes[0].scatter(outliers['Annual Income (k$)'], outliers['Spending Score (1-100)'], s=10, label='outliers', c="k")
axes[1].scatter(outliers['Age'], outliers['Spending Score (1-100)'], s=10, label='outliers', c="k")
axes[0].legend()
axes[1].legend()

plt.setp(axes[0].get_legend().get_texts(), fontsize='12')
plt.setp(axes[1].get_legend().get_texts(), fontsize='12')

plt.show()