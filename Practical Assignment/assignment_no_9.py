# -*- coding: utf-8 -*-
"""Assignment_no.9.ipynb

Write a program to do following: 

This dataset gives the data of Income and money spent by the customers visiting a shopping mall. 
The data set contains Customer ID, Gender, Age, Annual Income, Spending Score.
 Therefore, as a mall owner you need to find the group of people who are the profitable customers for the mall owner.
  Apply at least two clustering algorithms (based on Spending Score) to find the group of customers.

Apply Data pre-processing 
Perform data-preparation (Train-Test Split)
Apply Machine Learning Algorithm
Evaluate Model.

**Import Libraries**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

"""**Load Dataset**"""

df = pd.read_csv('/content/Mall_Customers.csv')
df.head()

print(df.info())
print(df.describe())

"""**Data Preprocessing**"""

# Remove missing values
df = df.dropna()

# Convert Gender into numeric
df['Genre'] = df['Genre'].map({'Male': 0, 'Female': 1})

# Select important features
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

"""Feature Scaling"""

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""Train-Test Split"""

X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

"""**Elbow Method (Find Best K)**"""

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

"""**K-Means Clustering**"""

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_train)

# Evaluation
kmeans_score = silhouette_score(X_train, kmeans_labels)
print("K-Means Silhouette Score:", kmeans_score)

"""**Hierarchical Clustering**"""

hc = AgglomerativeClustering(n_clusters=5)
hc_labels = hc.fit_predict(X_train)

# Evaluation
hc_score = silhouette_score(X_train, hc_labels)
print("Hierarchical Silhouette Score:", hc_score)

"""**Visualization**"""

plt.figure(figsize=(12,5))

# K-Means
plt.subplot(1,2,1)
plt.scatter(X_train[:,1], X_train[:,2], c=kmeans_labels)
plt.title("K-Means Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

# Hierarchical
plt.subplot(1,2,2)
plt.scatter(X_train[:,1], X_train[:,2], c=hc_labels)
plt.title("Hierarchical Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.show()

"""**Find Profitable Customers**"""

# Apply KMeans on full dataset
df['Cluster'] = KMeans(n_clusters=5, random_state=42).fit_predict(X_scaled)

# Group analysis
cluster_summary = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(cluster_summary)

"""Pairplot"""

sns.pairplot(df, hue='Cluster')
plt.show()