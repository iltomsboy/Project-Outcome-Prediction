import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Load dataset
df = pd.read_csv(r"C:\Users\Toms\Desktop\MASTER THESIS\Simulated_Dataset.csv")

# Variables to analyze
variables = ["budget", "duration", "team_size", "manager_experience"]

# -----------------------------
# DESCRIPTIVE STATISTICS
# -----------------------------
stats = pd.DataFrame(columns=["min", "q25", "median", "q75", "max", "mean"])

for var in variables:
    stats.loc[var] = [
        df[var].min(),
        df[var].quantile(0.25),
        df[var].median(),
        df[var].quantile(0.75),
        df[var].max(),
        df[var].mean()
    ]

print("\nDescriptive Statistics:\n")
print(stats)

# -----------------------------
# NORMAL DISTRIBUTION CURVES
# -----------------------------
for var in variables:

    plt.figure(figsize=(8,5))

    # Mean and standard deviation
    mean = df[var].mean()
    std = df[var].std()

    # X values for smooth curve
    x = np.linspace(df[var].min(), df[var].max(), 1000)

    # Normal distribution curve
    y = norm.pdf(x, mean, std)

    # Plot the line
    plt.plot(x, y)

    plt.title(f"Normal Distribution - {var}")
    plt.xlabel(var)
    plt.ylabel("Density")

    plt.show()

# -----------------------------
# BOXPLOTS
# -----------------------------
for var in variables:

    plt.figure(figsize=(6,4))

    plt.boxplot(df[var])

    # Remove the "1" from x-axis
    plt.xticks([])

    plt.title(f"Boxplot - {var}")
    plt.ylabel(var)

    plt.show()