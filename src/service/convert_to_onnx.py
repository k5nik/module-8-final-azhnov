import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
import logging
import onnx

model = joblib.load("model.pkl")
print(f"Загружена модель: {type(model)}")
print(f"Pipeline steps: {model.steps}")

initial_types = [
    ("pickup_longitude", FloatTensorType([None, 1])),
    ("pickup_latitude", FloatTensorType([None, 1])),
    ("dropoff_longitude", FloatTensorType([None, 1])),
    ("dropoff_latitude", FloatTensorType([None, 1])),
    ("passenger_count", Int64TensorType([None, 1])),
]

print("Конвертируем Pipeline в ONNX...")

onnx_model = convert_sklearn(
    model, initial_types=initial_types, name="fare_amount_prediction_model",
    target_opset=12
)

# Сохраняем
with open("artifacts/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Модель успешно конвертирована в artifacts/model.onnx")