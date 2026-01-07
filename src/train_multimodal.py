import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


FUSED_TRAIN_PATH = "outputs/fused_features_train.csv"
RAW_TRAIN_PATH = "data/raw/train.csv"


def main():
    # Load fused features (tabular + image)
    fused_df = pd.read_csv(FUSED_TRAIN_PATH)

    # Load original train data to get target
    target_df = pd.read_csv(RAW_TRAIN_PATH)[["id", "price"]]

    # Merge price back using id
    df = fused_df.merge(target_df, on="id", how="inner")

    # Separate features and target
    X = df.drop(columns=["id", "price"])
    y = df["price"]

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train simple multimodal model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    val_preds = model.predict(X_val)

    # Metrics
    mse = mean_squared_error(y_val, val_preds)
    rmse = mse**0.5
    r2 = r2_score(y_val, val_preds)

    print(f"Multimodal Validation RMSE: {rmse:.2f}")
    print(f"Multimodal Validation R2: {r2:.3f}")


if __name__ == "__main__":
    main()
