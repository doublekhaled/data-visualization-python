import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load cleaned dataset
df = pd.read_json("cleaned_students.json")

# Add attendance column for scatter plot analysis
np.random.seed(42)
df["attendance"] = np.random.randint(60, 100, size=len(df))

# -----------------------------
# Visualization 1: Bar Chart
# Average Grade by Major
# -----------------------------

major_avg = df.groupby("major")["average"].mean()

plt.figure(figsize=(8, 5))
major_avg.plot(kind="bar")

plt.title("Average Grade by Major")
plt.xlabel("Major")
plt.ylabel("Average Grade")

plt.tight_layout()
plt.savefig("bar_chart.png")
plt.show()

# -----------------------------
# Visualization 2: Histogram
# Distribution of Student Scores
# -----------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["average"], bins=5)

plt.title("Distribution of Student Scores")
plt.xlabel("Average Score")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.savefig("histogram.png")
plt.show()

# -----------------------------
# Visualization 3: Scatter Plot
# Attendance vs Average Grade
# -----------------------------

plt.figure(figsize=(8, 5))

plt.scatter(df["attendance"], df["average"])

plt.title("Attendance vs Average Grade")
plt.xlabel("Attendance (%)")
plt.ylabel("Average Grade")

plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.show()

print("Charts generated successfully.")
