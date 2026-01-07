import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import load_data, preprocess
from model import train_model, evaluate_model, train_rf_model

TRAIN_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/test.csv"
OUTPUT_PATH = "outputs/predictions.csv"


def main():
    # Load raw data
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    # Drop non-numeric columns
    if "date" in train_df.columns:
        train_df = train_df.drop(columns=["date"])
    if "date" in test_df.columns:
        test_df = test_df.drop(columns=["date"])

    # Preprocess (THIS DEFINES X and y)
    X, y, X_test = preprocess(train_df, test_df, target="price")

    # Train-validation split (USES X and y)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------- Linear Regression --------
    lr_model = train_model(X_train, y_train)
    lr_rmse, lr_r2 = evaluate_model(lr_model, X_val, y_val)

    print(f"Linear Regression Validation RMSE: {lr_rmse:.2f}")
    print(f"Linear Regression Validation R2: {lr_r2:.3f}")

    # -------- Random Forest --------
    rf_model = train_rf_model(X_train, y_train)
    rf_rmse, rf_r2 = evaluate_model(rf_model, X_val, y_val)

    print(f"Random Forest Validation RMSE: {rf_rmse:.2f}")
    print(f"Random Forest Validation R2: {rf_r2:.3f}")

    # -------- Final Predictions --------
    test_preds = rf_model.predict(X_test)

    output_df = pd.DataFrame({
        "id": test_df["id"],
        "predicted_price": test_preds
    })

    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main() 
