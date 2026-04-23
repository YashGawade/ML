# 11. Assignment on Classification technique. Every year many students give the 
# GRE exam to get admission in foreign Universities. The data set contains GRE Scores (out of 340), 
# TOEFL Scores (out of 120), University Rating (out of 5), Statement of Purpose strength (out of 5),
#  Letter of Recommendation strength (out of 5), Undergraduate GPA (out of 10), 
#  Research Experience (0=no, 1=yes), Admitted (0=no, 1=yes). Admitted is the target variable. 

# Data Set: https://www.kaggle.com/mohansacharya/graduate-admissions

# The counsellor of the firm is supposed check whether the student will get an admission 
# or not based on his/her GRE score and Academic Score. So to help the counsellor to make 
# appropriate decisions, build a machine learning model classifier using a Decision tree to
#  predict whether a student will get admission or not.  Apply Data pre-processing 
#  (Label Encoding, Data Transformation….) techniques if necessary. 

# Perform data-preparation (Train-Test Split)
# Apply Machine Learning Algorithm
# Evaluate Model.


# =========================================
# Import Libraries
# =========================================
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# =========================================
# Load Dataset
# =========================================
df = pd.read_csv("Admission_Predict.csv")

print("Dataset Loaded\n")
print(df.head())


# =========================================
# Data Preprocessing
# =========================================

# Drop unnecessary column (if present)
df.drop(columns=["Serial No."], inplace=True, errors='ignore')

# Convert target into binary (Admitted / Not Admitted)
# If dataset has "Chance of Admit", convert it
if "Chance of Admit " in df.columns:
    df["Admitted"] = df["Chance of Admit "] > 0.5
    df["Admitted"] = df["Admitted"].astype(int)


# =========================================
# Feature Selection
# =========================================

# Use GRE Score and CGPA as input features
X = df[["GRE Score", "CGPA"]]

# Target variable
y = df["Admitted"]


# =========================================
# Train-Test Split
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================================
# Apply Decision Tree Classifier
# =========================================
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# =========================================
# Prediction
# =========================================
y_pred = model.predict(X_test)


# =========================================
# Evaluation
# =========================================

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)
