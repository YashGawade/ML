# -*- coding: utf-8 -*-
"""ML_Assignment_no.3.ipynb
3.Visualize the data using Python by plotting the graphs for assignment no. 1 and 2. 
Consider a suitable data set. Use Scatter plot, Bar plot, Box plot, Pie chart, Line Chart.

**Scatter Plot**
"""

import matplotlib.pyplot as plt

experience = [1, 3, 5, 7, 9]
salary = [20, 35, 50, 65, 80]

colors = ['red', 'blue', 'green', 'orange', 'purple']
labels = ['E1', 'E2', 'E3', 'E4', 'E5']

for i in range(len(experience)):
    plt.scatter(experience[i], salary[i], color=colors[i], label=labels[i])

plt.xlabel("Experience (Years)")
plt.ylabel("Salary (in Thousands)")
plt.title("Scatter Plot: Employee Experience vs Salary")
plt.legend(title="Employees")
plt.show()

"""**Bar Plot**"""

import matplotlib.pyplot as plt

employees = ["E1", "E2", "E3", "E4", "E5"]
salary = [20, 35, 50, 65, 80]

colors = ['red', 'blue', 'green', 'orange', 'purple']

plt.bar(employees, salary, color=colors)
plt.xlabel("Employees")
plt.ylabel("Salary (in Thousands)")
plt.title("Bar Plot: Employee Salary")
plt.show()

"""**Box** **Plot**"""

import matplotlib.pyplot as plt

salary = [20, 35, 50, 65, 80]

plt.boxplot(salary)
plt.ylabel("Salary (in Thousands)")
plt.title("Box Plot: Salary Distribution")
plt.show()

"""**Pie Chart**"""

import matplotlib.pyplot as plt

employees = ["E1", "E2", "E3", "E4", "E5"]
salary = [20, 35, 50, 65, 80]

plt.pie(salary, labels=employees, autopct="%1.1f%%")
plt.title("Pie Chart: Salary Share of Employees")
plt.show()

"""Line Chart"""

import matplotlib.pyplot as plt

experience = [1, 3, 5, 7, 9]
salary = [20, 35, 50, 65, 80]

plt.plot(experience, salary, marker='o', color='blue', linestyle='-')
plt.xlabel("Experience (Years)")
plt.ylabel("Salary (in Thousands)")
plt.title("Line Chart: Experience vs Salary")
plt.grid(True)
plt.show()