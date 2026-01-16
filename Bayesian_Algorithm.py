import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------
# 1. MODELLO BAYESIANO (NAIVE BAYES GAUSSIANO)
# ---------------------------------------------------------

# Saving mean and st. deviation for each variable
class GaussianParams:   
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

# This is the structure (skeleton) for the model 
class BayesianProjectSuccessModel:
    def __init__(self):
        self.prior_success = None       # Estimating P(success) 
        self.prior_fail = None          # Estimating P(fail)
        self.params_success = {}        # Saving mean + st. deviation for each variable >> case of success
        self.params_fail = {}           # Saving mean + st. deviation for each variable >> case of fail
        self.features = None            # list of variables' names

    # Counting how many projects lead to success or fail
    def fit(self, X, y, feature_names):
        self.features = feature_names

        # Computing prior probabilities
        n_success = np.sum(y == 1) # counting how many projects are successful
        n_fail = np.sum(y == 0)    # counting how many projects are failures
        n_total = len(y)           # tot values of projects

        self.prior_success = n_success / n_total     # P(success) = n.. success / n. tot projects
        self.prior_fail = n_fail / n_total           # P(fail) = n. fail / n. tot projects

        # 
        for i, name in enumerate(feature_names):
            x_success = X[y == 1, i]    # xi | success
            x_fail = X[y == 0, i]       # xi | fail

            # Estimating the Noral Distribution using mean and st. deviation of the features for Successful Projects 
            # P(xi|success)
            self.params_success[name] = GaussianParams(
                mean=np.mean(x_success),
                std=np.std(x_success, ddof=1) + 1e-6    # to avoid Standard Deviation = 0 >> numerical problems! (AI suggestion)
            )

            # Estimating the Normal Distribution using mean and st. deviation of the features for the Failed projects
            # P(xi|fail)
            self.params_fail[name] = GaussianParams(
                mean=np.mean(x_fail),
                std=np.std(x_fail, ddof=1) + 1e-6       # to avoid Standard Deviation = 0 >> numerical problems! (AI suggestion)
            )

    # Implementing the Gaussian Probability Density Function (PDF) 
    # >> based on the ASSUMPTION it follows a Normal Distribution!
    # to compute P(xi|success) and P(xi|fail)
    def _gaussian_pdf(self, x, mean, std):         
        
        # where: x = feature's value, 
        # mean = estimated mean of the feature for a specific class (e.g. budget mean for successful projects), 
        # std = estimated standard deviation.
        
        coeff = 1.0 / (np.sqrt(2 * np.pi) * std)
        exponent = -0.5 * ((x - mean) / std) ** 2
        return coeff * np.exp(exponent)

    # Applying the Bayes Theorem
    def predict_proba(self, X):
        
        # Preparing all the structures
        n_samples = X.shape[0]                 # n. projects to be evaluated
        probs_success = np.zeros(n_samples)    # 1st vector = probability of success
        probs_fail = np.zeros(n_samples)       # 2nd vector = probability of fail

        for idx in range(n_samples): # loop for every project
            x = X[idx, :]

            # starting point for likelihood functions
            likelihood_success = 1.0
            likelihood_fail = 1.0
            
            for j, name in enumerate(self.features): # loop on the variables
                gp_s = self.params_success[name]
                gp_f = self.params_fail[name]

                # Computing Gaussian's likelihoods
                likelihood_success *= self._gaussian_pdf(x[j], gp_s.mean, gp_s.std) # P(xj|success)
                likelihood_fail *= self._gaussian_pdf(x[j], gp_f.mean, gp_f.std)    # P(xj|fail)
            
            # Applying Bayes Theorem
            numerator_s = likelihood_success * self.prior_success   # P(success|X) = [P(X|success)*P(success)] / P(X)
            numerator_f = likelihood_fail * self.prior_fail         # P(fail|X) = [P(X|fail)*P(fail)] / P(X)
            evidence = numerator_s + numerator_f                    # sum 

            probs_success[idx] = numerator_s / evidence     # computing P(success|X)
            probs_fail[idx] = numerator_f / evidence        # computing P(fail|X)

        # returning a matrix composed by 2 columns: P(fail) and P(success)
        return np.vstack([probs_fail, probs_success]).T

    # Creating a binary classification of success-fail projects
    # Creating 2 colums where if P(success) >= 0.5 -> classifies as success (1), otherwise as fail (0)
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        p_success = proba[:, 1]
        return (p_success >= threshold).astype(int)

# ---------------------------------------------------------
# 2. MAIN: CARICA DATASET + ALLENA + VALUTA + INPUT UTENTE
# ---------------------------------------------------------

def main():
    df = pd.read_csv(r"C:/Users/Toms/Desktop/MASTER THESIS/Simulated_Dataset.csv")

    # Identifying target (independent) and predictors (dependent)
    feature_names = ["budget", "duration", "team_size", "manager_experience"]
    X = df[feature_names].values
    y = df["success"].values

    # Splitting the dataset >> training data 80% + test data 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Training the model
    model = BayesianProjectSuccessModel()
    model.fit(X_train, y_train, feature_names)

    # Evaluating the model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\nAccuracy:", acc)
    print("Confusion matrix:\n", cm)

    # --- INPUT MANUALE UTENTE ---
    print("\n--- INSERISCI I DATI DEL NUOVO PROGETTO ---")

    budget = float(input("Budget del progetto (in migliaia di euro): "))
    duration = float(input("Durata del progetto (in mesi): "))
    team_size = float(input("Numero di persone nel team: "))
    manager_experience = float(input("Anni di esperienza del project manager (in anni): "))

    # Creating an array with the inputs
    new_project = np.array([[budget, duration, team_size, manager_experience]])

    # Computing P(fail|X) and P(success|X)
    proba = model.predict_proba(new_project)
    print("\nProbabilità di fallimento e successo (fail, success):")
    print(proba)

# Starting the algorithm
if __name__ == "__main__":
    main()