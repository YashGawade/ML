# 1.Perform the following operations using Python on suitable data sets:

# Read data from different formats (like csv, xls)
# Find Shape of Data
# Find Missing Values
# Find data type of each column
# Finding out Zero's
# Indexing and selecting data, sort data, 
# Describe attributes of data, checking data types of each column, 
# counting unique values of data, format of each column, converting variable data type 
# (e.g. from long to short, vice versa)



# Import pandas library for handling data
import pandas as pd

# -------------------------------------
# 1. Read Data (CSV / Excel)
# -------------------------------------

# Read CSV file
df = pd.read_csv("student_data.csv")

print("Dataset Loaded Successfully\n")

# -------------------------------------
# 2. Find Shape of Data
# -------------------------------------

# Shape gives (rows, columns)
print("Shape of Data:", df.shape)


# -------------------------------------
# 3. Find Missing Values
# -------------------------------------

# Check how many missing values in each column
print("\nMissing Values:")
print(df.isnull().sum())


# -------------------------------------
# 4. Data Types of Each Column
# -------------------------------------

# Shows type like int, float, object
print("\nData Types:")
print(df.dtypes)


# -------------------------------------
# 5. Finding Zero Values
# -------------------------------------

# Count how many zero values are present in each column
print("\nZero Values in Dataset:")
print((df == 0).sum())


# -------------------------------------
# 6. Indexing and Selecting Data
# -------------------------------------

# Select a single column
print("\nSelecting one column (example):")
print(df["Name"].head())

# Select multiple columns
print("\nSelecting multiple columns:")
print(df[["Name", "Age"]].head())

# Select rows using index (first 5 rows)
print("\nFirst 5 rows:")
print(df.iloc[0:5])


# -------------------------------------
# 7. Sorting Data
# -------------------------------------

# Sort data based on Age column
print("\nData sorted by Age:")
print(df.sort_values(by="Age").head())


# -------------------------------------
# 8. Describe Data (Statistics)
# -------------------------------------

# Gives mean, min, max, etc.
print("\nStatistical Summary:")
print(df.describe())


# -------------------------------------
# 9. Count Unique Values
# -------------------------------------

# Count unique values in each column
print("\nUnique Values Count:")
print(df.nunique())


# -------------------------------------
# 10. Value Counts (Frequency)
# -------------------------------------

# Count frequency of values in a column
print("\nValue count of Gender column:")
print(df["Gender"].value_counts())


# -------------------------------------
# 11. Convert Data Type
# -------------------------------------

# Example: converting Age to float
df["Age"] = df["Age"].astype(float)

print("\nAfter converting Age to float:")
print(df.dtypes)