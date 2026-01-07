# Multimodal House Price Prediction

This project implements a multimodal house price prediction pipeline using tabular housing data and satellite imagery.

## Project Structure

- notebooks/
  - 01_EDA.ipynb
  - 02_preprocessing.ipynb
  - 03_model_training.ipynb

- src/
  - preprocessing.py
  - model.py
  - data_fetcher.py
  - image_features.py
  - extract_all_image_features.py
  - fuse_features.py

- outputs/
  - predictions.csv
  - image_features/
  - fused_features_train.csv
  - fused_features_test.csv


## How to Run

```bash
pip install -r requirements.txt
python src/train.py
python src/fuse_features.py
```


