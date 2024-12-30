import pytest
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error

# Пути к файлам модели и данных
MODEL_PATH = "xgboost_model.pkl"
DATA_PATH = "Main_Data.csv"


# Фикстура для загрузки модели
@pytest.fixture
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


# Фикстура для загрузки тестовых данных
@pytest.fixture
def test_data():
    data = pd.read_csv(DATA_PATH)
    X = data[["T", "E", "C", "FM", "Xfm", "AFM", "Xafm"]]
    y = data["h"]
    return X, y


# Параметризованный тест: проверка работы модели на разных входных данных
@pytest.mark.parametrize(
    "test_input,expected",
    [
        ([5.056, -0.423195, 0.095988, 0.000066, 0.099824, 0.000442, 0.558774], 1),  # Ожидаемое значение "1"
        ([4.977, -0.430664, 0.098109, 0.000083, 0.099011, 0.000524, 0.575897], 1),
        ([4.898, -0.438642, 0.102846, 0.000079, 0.100388, 0.000426, 0.60144], 1),
        ([4.819, -0.447129, 0.105791, 0.000076, 0.101086, 0.000267, 0.631363], 1),
    ],
)
def test_model_with_various_inputs(load_model, test_input, expected):
    model = load_model

    # Преобразуем входные данные в формат numpy
    test_input_np = np.array([test_input])
    prediction = model.predict(test_input_np)[0]  # Получаем предсказание

    # Отладочный вывод
    print(f"Вход: {test_input}, Предсказание: {prediction}, Ожидаемое: {expected}")

    # Проверка предсказания
    assert prediction == expected, (
        f"Ошибка предсказания для входа {test_input}. "
        f"Ожидалось: {expected}, Получено: {prediction}"
    )


# Проверка метрик модели
def test_model_metrics(load_model, test_data):
    model = load_model
    X, y = test_data

    # Получаем предсказания
    predictions = model.predict(X)

    # Отладочный вывод
    mse = mean_squared_error(y, predictions)
    print(f"Mean Squared Error (MSE): {mse}")

    # Проверяем метрику MSE как пример
    assert mse < 0.1, f"Слишком большое значение MSE: {mse}"


# Дополнительный тест для проверки фикстуры test_data
def test_data_fixture(test_data):
    X, y = test_data
    assert not X.empty, "Тестовые данные X пусты"
    assert not y.empty, "Тестовые данные y пусты"
    print("Фикстура test_data загружена корректно")
