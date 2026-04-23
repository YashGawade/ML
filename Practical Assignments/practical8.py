# 8. Apply appropriate ML algorithm on a dataset collected in a cosmetics shop showing
#    details of customers to predict customer response for special offer. Create confusion  
#    matrix based on above data and find

# Accuracy
# Precision
# Recall
# F-1 score


# =========================================
# Import Libraries
# =========================================
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# =========================================
# Load Dataset
# =========================================
df = pd.read_csv("cosmetics.csv")   # dataset of customers

print("Dataset Loaded\n")
print(df.head())


# =========================================
# Data Preprocessing
# =========================================

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Remove missing values (if any)
df.dropna(inplace=True)


# =========================================
# Convert Categorical Data
# =========================================

# Convert text columns into numbers
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])


# =========================================
# Feature Selection
# =========================================

# Assume last column is target (Response: Yes/No)
X = df.iloc[:, :-1]   # all columns except last
y = df.iloc[:, -1]    # last column


# =========================================
# Train-Test Split
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================================
# Model Building (Classification)
# =========================================

model = LogisticRegression()
model.fit(X_train, y_train)

# Predict results
y_pred = model.predict(X_test)


# =========================================
# Confusion Matrix
# =========================================
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)


# =========================================
# Evaluation Metrics
# =========================================

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Precision
precision = precision_score(y_test, y_pred)

# Recall
recall = recall_score(y_test, y_pred)

# F1 Score
f1 = f1_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)