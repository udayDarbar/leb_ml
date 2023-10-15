import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load the Heart Disease dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
data = pd.read_csv(url, names=names, na_values='?')

# Handling missing values
data.dropna(inplace=True)

# Define the structure of the Bayesian network
model = BayesianNetwork([
    ('age', 'trestbps'),
    ('age', 'thalach'),
    ('sex', 'trestbps'),
    ('sex', 'chol'),
    ('trestbps', 'target'),
    ('chol', 'target'),
    ('thalach', 'target'),
    ('target', 'restecg')
])

# Learning CPDs using Maximum Likelihood Estimation
data_model = MaximumLikelihoodEstimator(model, data)
for node in model.nodes():
    cpd = data_model.estimate_cpd(node)
    model.add_cpds(cpd)

# Performing inference (diagnosis of heart patients)
inference = VariableElimination(model)
query = inference.query(variables=['target'], evidence={'age': 50, 'sex': 1})
print(query)

# Visualization of the Bayesian network (requires matplotlib and networkx)
# model.draw()
