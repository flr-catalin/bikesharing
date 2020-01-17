from pathlib import Path

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
FILENAME = "Bike-Sharing-Dataset.zip"
SAVE_PATH = Path("C:/Users/Catalin/PycharmProjects/licenta/bikesharing/datasets")
DAY_CSV_PATH = SAVE_PATH.joinpath(Path("day.csv"))
HOUR_CSV_PATH = SAVE_PATH.joinpath(Path("hour.csv"))
