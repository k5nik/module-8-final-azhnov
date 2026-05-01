import pandas as pd
from pathlib import Path

FEATURES_TO_DROP = ["key", "pickup_datetime"]


# Функция подготовки данных
def prepare_data(dataset_path):
    # Чтение данных из csv-файла
    df = pd.read_csv(dataset_path).drop(columns="Unnamed: 0")

    # Убираем ненужные колонки
    df.drop(
        columns=FEATURES_TO_DROP,
        inplace=True,
    )

    X = df.loc[:, df.columns != "fare_amount"]
    y = df["fare_amount"]

    return X, y


dataset_path = Path(__file__).resolve().parents[1] / "uber.csv"
X, _ = prepare_data(dataset_path)

print(X.dtypes)