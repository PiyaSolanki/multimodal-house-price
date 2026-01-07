import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess(train_df, test_df, target="price"):
    X = train_df.drop(columns=[target])
    y = train_df[target]

    X_test = test_df.copy()

    numeric_cols = X.select_dtypes(include="number").columns

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X, y, X_test
