import pytest
import pandas as pd
from main import load_data, load_model
import numpy as np

MODEL_PATH = "xgboost_model.pkl"

@pytest.fixture
def loaded_model():
    """Фикстура для загрузки модели."""
    model = load_model(MODEL_PATH)
    return model

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

