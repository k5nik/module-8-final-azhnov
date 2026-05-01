import onnxruntime as ort
import numpy as np
import time
import json
import pandas as pd
import joblib
from pathlib import Path


FEATURES_TO_DROP = ["key", "pickup_datetime"]
DEFAULT_FEATURES = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "passenger_count",
]

# Функция подготовки данных
def prepare_data(dataset_path):
    # Чтение данных из csv-файла
    df = pd.read_csv(dataset_path).drop(columns="Unnamed: 0")
    # Убираем ненужные колонки
    df.drop(
        columns=FEATURES_TO_DROP,
        inplace=True,
    )
    df = df.dropna(subset=DEFAULT_FEATURES)
    
    X = df.loc[:, df.columns != "fare_amount"]
    y = df["fare_amount"].astype(np.float32)
    return X, y


def benchmark_onnx(model_path: str, X: pd.DataFrame, num_runs: int = 1000):
    # Загружаем модель
    sess = ort.InferenceSession(model_path)

    # Тестовые данные
    test_inputs = {}

    for input_info in sess.get_inputs():
        tensor_type = input_info.type
        if tensor_type == "tensor(float)":
            test_inputs[input_info.name] = (
                X[input_info.name].values.reshape(-1, 1).astype(np.float32)
            )
        elif tensor_type == "tensor(int64)":
            test_inputs[input_info.name] = (
                X[input_info.name].values.reshape(-1, 1).astype(np.int64)
            )

    # Тестирование
    latencies = []
    for i in range(num_runs):
        start = time.perf_counter()
        result = sess.run(None, test_inputs)
        end = time.perf_counter()
        latencies.append(end - start)

    # Результаты
    latencies.sort()
    results = {
        "runs": num_runs,
        "min_s": np.round(min(latencies), 3),
        "mean_s": np.round((sum(latencies) / len(latencies)), 3),
        "p50_s": np.round(np.percentile(latencies, 50), 3),
        "p95_s": np.round(np.percentile(latencies, 95), 3),
        "p99_s": np.round(np.percentile(latencies, 99), 3),
        "max_s": np.round(max(latencies), 3),
    }

    return results


def benchmark_sklearn(model_path: str, X: pd.DataFrame, num_runs: int = 1000):
    # Загружаем модель
    clf = joblib.load(model_path)

    # Тестирование
    latencies = []
    for i in range(num_runs):
        start = time.perf_counter()
        predicts = clf.predict(X)
        end = time.perf_counter()
        latencies.append(end - start)

    # Результаты
    latencies.sort()
    results = {
        "runs": num_runs,
        "min_s": np.round(min(latencies), 3),
        "mean_s": np.round((sum(latencies) / len(latencies)), 3),
        "p50_s": np.round(np.percentile(latencies, 50), 3),
        "p95_s": np.round(np.percentile(latencies, 95), 3),
        "p99_s": np.round(np.percentile(latencies, 99), 3),
        "max_s": np.round(max(latencies), 3),
    }

    return results


# Запуск
if __name__ == "__main__":
    dataset_path = Path(__file__).resolve().parents[1] / "uber.csv"
    X, y = prepare_data(dataset_path)

    model_path = "artifacts/model.onnx"
    results = benchmark_onnx(model_path, X)
    
    #Сохраняем только результаты для onnx
    with open("artifacts/profile_onnx_IvanAzhnov.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Вывод
    print(f"\nLatency for onnx (s):")
    print(f"  Min:  {results['min_s']:.3f}")
    print(f"  Mean: {results['mean_s']:.3f}")
    print(f"  P50:  {results['p50_s']:.3f}")
    print(f"  P95:  {results['p95_s']:.3f}")
    print(f"  P99:  {results['p99_s']:.3f}")
    print(f"  Max:  {results['max_s']:.3f}")

    model_path = "artifacts/model.pkl"
    results = benchmark_sklearn(model_path, X)

    # Вывод
    print(f"\nLatency for sklearn(s):")
    print(f"  Min:  {results['min_s']:.3f}")
    print(f"  Mean: {results['mean_s']:.3f}")
    print(f"  P50:  {results['p50_s']:.3f}")
    print(f"  P95:  {results['p95_s']:.3f}")
    print(f"  P99:  {results['p99_s']:.3f}")
    print(f"  Max:  {results['max_s']:.3f}") 
    
    