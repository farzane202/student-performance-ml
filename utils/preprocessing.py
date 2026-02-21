import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess(path):
    # Load dataset (semicolon separated)
    data = pd.read_csv(path, sep=";")

    # Define features and target
    X = data.drop("G3", axis=1)
    y = data["G3"]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, X.columns