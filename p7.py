import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer

# Load the data from the CSV file with appropriate encoding
data = pd.read_csv('iris.csv', encoding='latin1')

# Assuming 'data.csv' contains relevant numeric features and we need to remove non-numeric columns
data_numeric = data.select_dtypes(include=['float', 'int'])

# Impute missing values with mean (you can choose other strategies)
imputer = SimpleImputer(strategy='mean')
data_numeric_imputed = imputer.fit_transform(data_numeric)

# Check the first few rows of the numeric data
print("Numeric Data Preview:")
print(data_numeric.head())

# Assume n_clusters is the number of clusters you want to find
n_clusters = 3  # You can change this based on your requirement

# Fit the Gaussian Mixture Model using the numeric data
em_model = GaussianMixture(n_components=n_clusters)
em_model.fit(data_numeric_imputed)

# Get the cluster assignments for each data point
em_labels = em_model.predict(data_numeric_imputed)

# Display the cluster assignments
print("\nEM Algorithm - Cluster Assignments:")
print(em_labels)
