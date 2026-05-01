import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy
from onnxruntime import InferenceSession
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import joblib
from pathlib import Path


FEATURES_TO_DROP = ["key", "pickup_datetime"]


# Функция подготовки данных
def prepare_data(dataset_path):
    # Чтение данных из csv-файла
    df = pd.read_csv(dataset_path).drop(columns="Unnamed: 0")
    print(df)
    # Убираем ненужные колонки
    df.drop(
        columns=FEATURES_TO_DROP,
        inplace=True,
    )
    
    X = df.loc[:, df.columns != "fare_amount"]
    y = df["fare_amount"].astype(np.float32)
    print(X)
    return X, y


def diff(p1, p2):
    p1 = p1.ravel()
    p2 = p2.ravel()
    d = numpy.abs(p2 - p1)
    return d.max(), (d / numpy.abs(p1)).max()


dataset_path = Path(__file__).resolve().parents[1] / "test.csv"
X, y = prepare_data(dataset_path)

sess = InferenceSession("artifacts/tree_model.onnx")
input_name = sess.get_inputs()
label_name = sess.get_outputs()

prepare_inputs = {}

for input_info in sess.get_inputs():
    tensor_type = input_info.type
    if tensor_type == "tensor(float)":
        prepare_inputs[input_info.name] = (
            X[input_info.name].values.reshape(-1, 1).astype(np.float32)
        )
    elif tensor_type == "tensor(int64)":
        prepare_inputs[input_info.name] = (
            X[input_info.name].values.reshape(-1, 1).astype(np.int64)
        )

pred_ort = sess.run(None, prepare_inputs)[0]

clf = joblib.load("model.pkl")
print(X)
predicts = clf.predict(X)
print(diff(predicts, pred_ort))