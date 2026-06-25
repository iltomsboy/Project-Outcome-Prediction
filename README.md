Project Success Prediction Using a Custom Bayesian Classifier

-------------------------------
Overview:

This project demonstrates how a data-driven approach can support project management decision-making by estimating the probability of project success. The workflow consists of three main stages:
1) Synthetic dataset generation,
2) Exploratory data analysis,
3) Development of a custom Gaussian Naïve Bayes classifier.

The project was developed as part of a research study exploring the application of probabilistic models to project success prediction.

Project Structure:
├── Dataset_Simulation.py      # Synthetic dataset generation
├── Dataset_Analysis.py        # Exploratory data analysis
├── Bayesian_Algorithm.py      # Custom Bayesian prediction model
├── Simulated_Dataset.csv      # Generated dataset (created by Dataset_Simulation.py)
└── README.md

-------------------------------
Project Workflow:

----- 1. Dataset Simulation -----

File: Dataset_Simulation.py

The first script generates a synthetic dataset representing projects characterized by four key variables:
- Budget (thousand €),
- Duration (months),
- Team Size,
- Manager Experience (years).

The variables are generated using normal distributions and constrained to realistic values.

A hidden logistic function is then used to create the target variable:
- Success = 1,
- Failure = 0.

The underlying assumption is that:
- Higher budgets increase success probability,
- More experienced managers increase success probability,
- Larger teams slightly increase success probability, 
- Longer project durations decrease success probability.

The generated dataset is exported as: Simulated_Dataset.csv

----- 2. Exploratory Data Analysis -----

File: Dataset_Analysis.py

This script performs a preliminary analysis of the generated dataset.

Descriptive Statistics - For each variable it computes:
>> Minimum, 
>> 25th Percentile, 
>> Median, 
>> 75th Percentile,
>> Maximum, 
>> Mean.

Distribution Analysis - The script visualizes the distribution of each feature by plotting:
- Normal distribution curves,
- Boxplots.

These visualizations help verify whether the generated data approximately follows the assumptions required by the Bayesian classifier.

----- 3. Bayesian Classification Model -----

File: Bayesian_Algorithm.py

This script implements a Gaussian Naïve Bayes classifier from scratch, without using Scikit-Learn's built-in Naïve Bayes implementation.

Main Components:

Prior Probabilities - The model estimates:
- P(success),
- P(failure).

from the training dataset.

Gaussian Parameters - For every feature and class, the model stores:
- Mean,
- Standard deviation.

Examples:
Budget | Success
Budget | Failure
Duration | Success
Duration | Failure
Gaussian Probability Density Function

The likelihood of observing a feature value is computed using the Gaussian PDF:
>> P(xi | Success),
>> P(xi | Failure).

assuming each feature follows a normal distribution.

Bayes' Theorem - The posterior probabilities are calculated as:
- P(Success | X),
- P(Failure | X).

where X represents the project characteristics.

Classification - Projects are classified using a threshold:
If P(Success) ≥ 0.5 → Successful Project
Else → Failed Project

Model Evaluation - The dataset is divided into:
- 80% Training Set,
- 20% Test Set.

The model performance is evaluated using:
- Accuracy Score => Measures the proportion of correctly classified projects.
- Confusion Matrix => Provides detailed information about:
        - True Positives,
        - True Negatives
        - False Positives
        - False Negatives

User Interaction - After training the model, the user can manually enter the characteristics of a new project:
>> Budget,
>> Duration, 
>> Team Size,
>> Manager Experience.

The model then estimates:
- P(Failure | Project Data),
- P(Success | Project Data).

allowing project managers to assess the likelihood of project success before execution.

-------------------------------
Technologies Used:
Python
NumPy
Pandas
Matplotlib
SciPy
Scikit-Learn
Educational Purpose

-------------------------------
This project was developed primarily for educational and research purposes to demonstrate:
1) Synthetic data generation,
2) Exploratory data analysis,
3) Probabilistic machine learning,
4) Bayesian inference,
5) Implementation of a Gaussian Naïve Bayes classifier from scratch.

The focus is not on achieving state-of-the-art predictive performance, but on understanding the mathematical foundations behind Bayesian classification and its potential application in project management decision-making.

-------------------------------
Future Improvements - Possible extensions of the project include:
--> Using real-world project management datasets,
--> Feature engineering and additional project variables,
--> Cross-validation techniques,
--> Comparison with other machine learning models:
        - Logistic Regression,
        - Decision Trees,
        - Random Forests.
--> Hyperparameter optimization,
--> Development of a graphical user interface (GUI).

-------------------------------
Author: Tommaso Cecchellero

Master's Degree in Business Management
Research focus: Project Management, Data Analytics, and Artificial Intelligence Applications in Decision-Making.
