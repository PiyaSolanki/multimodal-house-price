import os
import pandas as pd
import torch

from image_features import extract_image_features


TRAIN_CSV = "data/raw/train.csv"
TEST_CSV = "data/raw/test.csv"

TRAIN_IMG_DIR = "data/images/train"
TEST_IMG_DIR = "data/images/test"

OUT_TRAIN = "outputs/image_features/train_image_features.csv"
OUT_TEST = "outputs/image_features/test_image_features.csv"


def get_existing_ids(df, img_dir):
    image_ids = {
        int(f.replace(".png", ""))
        for f in os.listdir(img_dir)
        if f.endswith(".png")
    }
    return df[df["id"].isin(image_ids)]


def run(csv_path, img_dir, out_path):
    df = pd.read_csv(csv_path)

    # 🔑 FILTER TO ONLY DOWNLOADED IMAGES
    df = get_existing_ids(df, img_dir)

    if len(df) == 0:
        print(f"No images found in {img_dir}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    features = extract_image_features(
        image_dir=img_dir,
        ids=df["id"].tolist(),
        device=device
    )

    feature_df = pd.DataFrame(features)
    feature_df["id"] = df["id"].values

    feature_df.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(feature_df)} rows")


if __name__ == "__main__":
    run(TRAIN_CSV, TRAIN_IMG_DIR, OUT_TRAIN)
    run(TEST_CSV, TEST_IMG_DIR, OUT_TEST)
