import os
import requests
from tqdm import tqdm

from src.config import MAPBOX_TOKEN, ZOOM_LEVEL, IMAGE_SIZE, MAP_STYLE



def get_mapbox_image(lat, lon, save_path):
    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/{MAP_STYLE}/static/"
        f"{lon},{lat},{ZOOM_LEVEL}/"
        f"{IMAGE_SIZE}"
        f"?access_token={MAPBOX_TOKEN}"
    )

    response = requests.get(url)

    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
    else:
        raise Exception(f"Mapbox error: {response.status_code}")


def download_images(df, output_dir, lat_col="lat", lon_col="long", id_col="id"):
    os.makedirs(output_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_name = f"{row[id_col]}.png"
        save_path = os.path.join(output_dir, image_name)

        if os.path.exists(save_path):
            continue

        get_mapbox_image(row[lat_col], row[lon_col], save_path)
