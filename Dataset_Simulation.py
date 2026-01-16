import numpy as np
import pandas as pd

def simulate_project_data(n=200, random_state=42): # n = number of rows in the simulate dataset
    np.random.seed(random_state)                   # using "seed" i'm generating always the same values >> useful for debuging

    # Feature distributions >> generated using a Normal Distribution
    # where: loc = average value; scale = standard deviation; n = number of vause to be generated
    budget = np.random.normal(loc=250, scale=80, size=n)              # set randomly the budget in float k€
    duration = np.random.normal(loc=8, scale=3, size=n)               # set randomly the duration in float months
    team_size = np.random.normal(loc=10, scale=4, size=n)             # set randomly the team size float value
    manager_experience = np.random.normal(loc=7, scale=3, size=n)     # set randomly the manager's experience float years

    # Delete unrealistic values
    duration = np.clip(duration, 1, None)                             # project must last at least 1 month
    team_size = np.clip(team_size, 2, None)                           # team size at least 2 persons: manager + employee
    manager_experience = np.clip(manager_experience, 0, None)         # manager's experience cannot be negative

    # True underlying probability model (hidden from the classifier)
    # Logistic-like function to generate realistic success probabilities 
    # Definiton of the weight of each feature for the final otucome
    logits = (
        0.015 * budget
        - 0.20 * duration
        + 0.05 * team_size
        + 0.35 * manager_experience
        - 5  # intercept to keep probabilities reasonable (AI choice)
    )

    probs = 1 / (1 + np.exp(-logits)) # transforming logits into Boolean probability

    # Generate binary outcomes
    success = np.random.binomial(1, probs)

    # Creating the dataframe
    df = pd.DataFrame({
        "budget": budget,
        "duration": duration,
        "team_size": team_size,
        "manager_experience": manager_experience,
        "success": success
    })

    return df

# Creating the simulated dataset
df = simulate_project_data(n=200)
print(df.head(7)) #print first 7 rows
df.to_csv(r"C:\Users\Toms\Desktop\MASTER THESIS\Simulated_Dataset.csv", index=False) #to save it localy and use it for the algorithm creation


