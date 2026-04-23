# 2. Perform the following operations using  Python on the data sets:

# Compute and display summary statistics for each feature available in the dataset. 
# (e.g. minimum value, maximum value, mean, range, standard deviation, variance and percentiles
# Illustrate the feature distributions using histogram.
# Data cleaning, Data integration, Data transformation, Data model building (e.g. Classification) 

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# -------------------------------------
# 1. Load Dataset
# -------------------------------------

# Read dataset (Iris dataset)
df = pd.read_csv("Iris.csv")

print("Dataset Loaded Successfully\n")


# -------------------------------------
# 2. Summary Statistics
# -------------------------------------

# Display basic statistical summary
print("Statistical Summary:")
print(df.describe())

# Individual statistics
print("\nMinimum Values:\n", df.min(numeric_only=True))
print("\nMaximum Values:\n", df.max(numeric_only=True))
print("\nMean Values:\n", df.mean(numeric_only=True))
print("\nStandard Deviation:\n", df.std(numeric_only=True))
print("\nVariance:\n", df.var(numeric_only=True))

# Range = max - min
print("\nRange of Values:\n", df.max(numeric_only=True) - df.min(numeric_only=True))


# -------------------------------------
# 3. Histogram (Feature Distribution)
# -------------------------------------

# Plot histogram for all numeric columns
df.iloc[:, :-1].hist(figsize=(8,6))
plt.suptitle("Histogram of Features")
plt.show()


# -------------------------------------
# 4. Data Cleaning
# -------------------------------------

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Remove duplicate rows if any
df.drop_duplicates(inplace=True)

print("\nShape after removing duplicates:", df.shape)


# -------------------------------------
# 5. Data Integration
# -------------------------------------

# Separate features (X) and target (y)
X = df.iloc[:, 1:5]   # input features
y = df["Species"]     # output


# -------------------------------------
# 6. Data Transformation
# -------------------------------------

# Convert categorical data (Species) to numeric
le = LabelEncoder()
df["Species_encoded"] = le.fit_transform(df["Species"])

# Update target variable
y = df["Species_encoded"]


# -------------------------------------
# 7. Model Building (Classification)
# -------------------------------------

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict output
y_pred = model.predict(X_test)


# -------------------------------------
# 8. Model Evaluation
# -------------------------------------

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)