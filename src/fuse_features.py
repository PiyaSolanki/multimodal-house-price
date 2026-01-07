import pandas as pd
from preprocessing import load_data, preprocess

# Paths
TRAIN_CSV = "data/raw/train.csv"
TEST_CSV = "data/raw/test.csv"

TRAIN_IMG_FEATS = "outputs/image_features/train_image_features.csv"
TEST_IMG_FEATS = "outputs/image_features/test_image_features.csv"

OUT_TRAIN = "outputs/fused_features_train.csv"
OUT_TEST = "outputs/fused_features_test.csv"


def main():
    # Load raw tabular data
    train_df, test_df = load_data(TRAIN_CSV, TEST_CSV)

    # Drop non-numeric column if present
    if "date" in train_df.columns:
        train_df = train_df.drop(columns=["date"])
    if "date" in test_df.columns:
        test_df = test_df.drop(columns=["date"])

    # Preprocess tabular features
    X_tab, y, X_tab_test = preprocess(train_df, test_df, target="price")

    # Add ID back (needed for fusion)
    X_tab["id"] = train_df["id"].values
    X_tab_test["id"] = test_df["id"].values

    # Load CNN image features
    img_train = pd.read_csv(TRAIN_IMG_FEATS)
    img_test = pd.read_csv(TEST_IMG_FEATS)

    # Fuse tabular + image features
    fused_train = X_tab.merge(img_train, on="id", how="inner")
    fused_test = X_tab_test.merge(img_test, on="id", how="inner")

    # Save fused datasets
    fused_train.to_csv(OUT_TRAIN, index=False)
    fused_test.to_csv(OUT_TEST, index=False)

    print("Multimodal feature fusion completed successfully.")
    print(f"Fused train shape: {fused_train.shape}")
    print(f"Fused test shape: {fused_test.shape}")


if __name__ == "__main__":
    main()
