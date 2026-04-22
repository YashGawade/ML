# -*- coding: utf-8 -*-
"""ML_Assignment_1.ipynb

1.Perform the following operations using Python on suitable data sets:

Read data from different formats (like csv, xls)
Find Shape of Data
Find Missing Values
Find data type of each column
Finding out Zero's
Indexing and selecting data, sort data, 
Describe attributes of data, checking data types of each column, 
counting unique values of data, format of each column, converting variable data type (e.g. from long to short, vice versa)

Step 1: Upload Data
"""

from google.colab import files
uploaded = files.upload()

"""1) Read Data from CSV and **Excel**"""

import pandas as pd
df = pd.read_excel("research_student (1).xlsx")
df.head()

"""Step 3: Find Shape of Data"""

print("shape of  data",df.shape)

"""4: Find Missing Values"""

print("Missing values in each column")
print(df.isnull().sum())

"""5: Find Data Type of Each Column"""

print("data types of each column")
print(df.dtypes)

"""6: Finding out Zero’s"""

print("zeros in each row ")
print((df==0).sum())

"""STEP 8: Indexing & Selecting Data"""

df[["Branch", "CGPA"]].head()

"""a.Select first row using iloc
b.Filter Students with CGPA > 7
"""

df.iloc[0]
df[df["CGPA"] > 7]

"""9: Sorting Data
Sort by CGPA (High to Low)
"""

df.sort_values(by="CGPA", ascending=False).head()

"""10: Describe Dataset (Statistics)"""

df.describe()

"""11.Dataset info"""

df.info()

"""12: Unique Value Count"""

print(df.nunique())
print(df["Branch"].unique())
print(df["Branch"].value_counts())

"""13: Convert Data Types
Convert “Current Back” to int (if possible)
"""

df["Current Back"] = df["Current Back"].fillna(0).astype(int)

