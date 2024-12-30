import pytest
import pandas as pd
from main import load_data, load_model
import numpy as np

DATA_PATH = "Main_Data.csv"
MODEL_PATH = "xgboost_model.pkl"

@pytest.fixture
def loaded_data():
    """Фикстура для загрузки данных."""
    X, y = load_data(DATA_PATH)
    return X, y

@pytest.fixture
def loaded_model():
    """Фикстура для загрузки модели."""
    model = load_model(MODEL_PATH)
    return model

def test_load_data(loaded_data):
    """
    Проверка корректности загрузки данных.
    """
    X, y = loaded_data
    
    # Проверяем, что X — это DataFrame, а y — Series
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    
    # Проверяем, что данные не пустые
    assert not X.empty
    assert not y.empty
    
    # Проверяем, что в X есть ожидаемые колонки
    expected_columns = ["T", "E", "C", "FM", "Xfm", "AFM", "Xafm"]
    assert all(col in X.columns for col in expected_columns)
    
    # Проверяем, что y содержит только бинарные значения (0 и 1)
    assert set(y.unique()).issubset({0, 1})
    
    # Выводим информацию о данных
    print("\nДанные успешно загружены:")
    print(f"X (признаки): {X.shape[0]} строк, {X.shape[1]} столбцов")
    print(f"y (целевая переменная): {len(y)} значений")

def test_load_model(loaded_model):
    """
    Проверка корректности загрузки модели.
    """
    model = loaded_model
    
    # Проверяем, что модель загружена
    assert model is not None
    
    # Проверяем, что модель имеет метод predict (это XGBoost модель)
    assert hasattr(model, "predict")
    
    # Выводим информацию о модели
    print("\nМодель успешно загружена:")
    print(f"Тип модели: {type(model)}")

def test_predict(loaded_model, loaded_data):
    """
    Проверка корректности предсказательной функции модели.
    """
    model = loaded_model
    X, _ = loaded_data
    
    # Получаем предсказания
    predictions = model.predict(X)
    
    # Проверяем, что предсказания — это массив numpy
    assert isinstance(predictions, np.ndarray)
    
    # Проверяем, что количество предсказаний совпадает с количеством строк в данных
    assert len(predictions) == len(X)
    
    # Проверяем, что предсказания содержат только бинарные значения (0 и 1)
    assert set(predictions).issubset({0, 1})
    
    # Выводим информацию о предсказаниях
    print("\nПредсказания успешно получены:")
    print(f"Количество предсказаний: {len(predictions)}")
    print(f"Пример предсказаний: {predictions[:5]}")  # Выводим первые 5 предсказаний
