# Project Success Prediction Using a Custom Bayesian Classifier

## Overview

This project demonstrates how Bayesian inference can be applied to estimate the probability of project success based on key project characteristics.

The workflow includes:

1. Synthetic dataset generation
2. Exploratory data analysis
3. Development of a custom Gaussian Naïve Bayes classifier

---

## Project Structure

```text
├── Dataset_Simulation.py      # Synthetic dataset generation
├── Dataset_Analysis.py        # Exploratory data analysis
├── Bayesian_Algorithm.py      # Custom Bayesian classifier
├── Simulated_Dataset.csv      # Generated dataset
└── README.md
```

---

## Features

The model uses four project-related variables:

- Budget (€ thousands)
- Duration (months)
- Team Size
- Manager Experience (years)

and predicts:

- Project Success (1)
- Project Failure (0)

---

## Scripts

### Dataset_Simulation.py

Generates a synthetic project management dataset using statistical distributions and a hidden probability model to simulate project outcomes.

### Dataset_Analysis.py

Performs exploratory data analysis by calculating descriptive statistics and visualizing data distributions through normal curves and boxplots.

### Bayesian_Algorithm.py

Implements a Gaussian Naïve Bayes classifier from scratch, trains the model, evaluates its performance, and predicts the probability of success for new projects.

---

## Model Evaluation

The dataset is split into:

- 80% Training Set
- 20% Test Set

Performance is evaluated using:

- Accuracy Score
- Confusion Matrix

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-Learn

---

## Purpose

This project was developed for educational and research purposes to demonstrate:

- Synthetic data generation
- Exploratory data analysis
- Bayesian inference
- Machine learning fundamentals
- Custom implementation of a Gaussian Naïve Bayes classifier

---

## Author

**Tommaso Cecchellero**

Master's Degree in Business Management

Interest areas: Project Management, Data Science, and Artificial Intelligence.
