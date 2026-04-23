# 3.Visualize the data using Python by plotting the graphs for assignment no. 1 and 2. 
# Consider a suitable data set. Use Scatter plot, Bar plot, Box plot, Pie chart, Line Chart.


# =========================================
# Import Libraries
# =========================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================
# Load Dataset
# =========================================
df = pd.read_csv("Iris.csv")


# =========================================
# 1. Scatter Plot
# =========================================
# Shows relationship between two features

sns.scatterplot(
    data=df,
    x="SepalLengthCm",
    y="PetalLengthCm",
    hue="Species"
)

plt.title("Scatter Plot: Sepal vs Petal Length")
plt.show()


# =========================================
# 2. Bar Plot
# =========================================
# Compares average values across categories

sns.barplot(
    data=df,
    x="Species",
    y="SepalLengthCm"
)

plt.title("Bar Plot: Average Sepal Length")
plt.show()


# ==================================
# 3. Box Plot
# =========================================
# Shows distribution and outliers

sns.boxplot(
    data=df,
    x="Species",
    y="SepalLengthCm"
)

plt.title("Box Plot: Sepal Length Distribution")
plt.show()


# ===================================
# 4. Pie Chart
# =========================================
# Shows percentage distribution of species

counts = df["Species"].value_counts()

plt.pie(
    counts.values,
    labels=counts.index,
    autopct="%1.1f%%"
)

plt.title("Pie Chart: Species Distribution")
plt.show()


# =========================================
# 5. Line Chart
# =========================================
# Shows trend of values

plt.plot(
    df.index,
    df["SepalLengthCm"]
)

plt.title("Line Chart: Sepal Length Trend")
plt.show()