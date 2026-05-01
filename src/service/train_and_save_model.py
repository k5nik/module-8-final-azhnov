import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "passenger_count",
]
DEFAULT_TARGET = "fare_amount"


def load_dataset(csv_path: Path, features: list[str], target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    missing_columns = [col for col in features + [target] if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"В датасете отсутствуют необходимые колонки: {missing_columns}.\n"
            f"Найдены колонки: {list(df.columns)}"
        )

    # Базовая очистка
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features + [target])

    X = df[features].copy()
    y = df[target].astype(float).copy()
    return X, y


def build_model(features: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, features),
        ],
        remainder="drop",
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])
    return model


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = build_model(list(X.columns))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение модели прогноза стоимости такси и сохранение в model.pkl")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "uber.csv",
        help="Путь к CSV с данными (по умолчанию src/uber.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "model.pkl",
        help="Куда сохранить обученную модель (по умолчанию src/service/model.pkl)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Размер тестовой выборки (доля)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Зерно генератора для воспроизводимости",
    )

    args = parser.parse_args()

    features = DEFAULT_FEATURES
    target = DEFAULT_TARGET

    X, y = load_dataset(args.csv, features, target)
    model, metrics = train_and_evaluate(X, y, test_size=args.test_size, random_state=args.random_state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)

    print(
        f"Эксперимент завершён. Метрики: RMSE={metrics['rmse']:.3f}, "
        f"MAE={metrics['mae']:.3f}, R2={metrics['r2']:.3f}. Модель сохранена в: {args.output}"
    )


if __name__ == "__main__":
    main()
