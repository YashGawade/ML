# 9.Write a program to do following: 

# This dataset gives the data of Income and money spent by the customers visiting a shopping mall.
#  The data set contains Customer ID, Gender, Age, Annual Income, Spending Score. 
#  Therefore, as a mall owner you need to find the group of people who are the profitable customers
# for the mall owner. Apply at least two clustering algorithms (based on Spending Score)
# to find the group of customers.

# Apply Data pre-processing 
# Perform data-preparation (Train-Test Split)
# Apply Machine Learning Algorithm
# Evaluate Model.
#       Data Set: https://www.kaggle.com/shwetabh123/mall-customers


# =========================================
# Import Libraries
# =========================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering


# =========================================
# Load Dataset
# =========================================
df = pd.read_csv("Mall_Customers.csv")

print("Dataset Loaded\n")
print(df.head())


# =========================================
# Data Preprocessing
# =========================================

# Convert Gender to numeric (Male=1, Female=0)
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Select important features (Income & Spending Score)
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]


# =========================================
# Data Preparation (Scaling)
# =========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# =========================================
# K-Means Clustering
# =========================================

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataset
df["KMeans_Cluster"] = kmeans_labels


# =========================================
# Agglomerative Clustering
# =========================================

agg = AgglomerativeClustering(n_clusters=5)
agg_labels = agg.fit_predict(X_scaled)

df["Agg_Cluster"] = agg_labels


# =========================================
# Visualization
# =========================================

# KMeans clusters
plt.figure(figsize=(6,5))
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], 
            c=df["KMeans_Cluster"])
plt.title("K-Means Clustering")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.show()

# Agglomerative clusters
plt.figure(figsize=(6,5))
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], 
            c=df["Agg_Cluster"])
plt.title("Agglomerative Clustering")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.show()