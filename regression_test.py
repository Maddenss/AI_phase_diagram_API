import json
import numpy as np
from main import load_data, load_model

def test_regression_predictions():
    """
    Регрессионный тест: проверка, что предсказания модели остаются неизменными
    после изменений в коде.
    """
    # Загружаем данные и модель
    X, _ = load_data("Main_Data.csv")
    model = load_model("xgboost_model.pkl")
    
    # Получаем текущие предсказания
    current_predictions = model.predict(X)
    
    # Загружаем эталонные предсказания
    with open('predictions.json', 'r') as f:
        expected_predictions = json.load(f)
    
    # Проверяем, что предсказания не изменились
    assert np.array_equal(current_predictions, np.array(expected_predictions)), "Предсказания изменились!"

    print("Регрессионный тест пройден: предсказания модели не изменились.")