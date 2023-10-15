import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Load the CSV file into a pandas DataFrame
def load_data(filename):
    df = pd.read_csv(filename)
    return df

# Preprocess the data
def preprocess_data(data):
    # Assuming the last column is the target variable
    target_column = data.columns[-1]
    # Convert categorical variables to numerical labels
    data = pd.get_dummies(data, columns=[col for col in data.columns if col != target_column])
    # Separate features and the target variable
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y

# Train the Naive Bayes classifier
def train_classifier(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

# Test the classifier and compute accuracy
def test_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy

# Main function
def main():
    # Load the data
    filename = 'iris.csv'  # Replace with the actual file path
    data = load_data(filename)

    # Preprocess the data
    X, y = preprocess_data(data)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Naive Bayes classifier
    model = train_classifier(X_train, y_train)

    # Test the classifier and compute accuracy
    accuracy = test_classifier(model, X_test, y_test)
    print('Accuracy:', accuracy)

if __name__ == "__main__":
    main()
