# AI Phase Diagram API

## Описание:
Это приложение реализует API для выполнения различных задач машинного обучения, включая:
- Расчет метрик используемой модели.
- Генерацию ROC-кривых.
- Кросс-валидацию.
- SHAP-анализ для интерпретации моделей.
- Получение важности признаков.
- Выполнение предсказаний классов на основе загружаемых данных по используемой модели.

## Запуск приложения на локальной машине:

1. Необходимо установить и запустить ПО "Docker Desktop"
2. Скачать архив файлов приложение с GitHub
3. Разархивировать файлы приложения в созданную папку `app`.
4. Запустить терминал (командную строку), перейти к папке `app`
5. Запустить приложение через терминал (командную строку)
   ```bash
   docker-compose up --build
   ```   
6. Перейдите в браузере по адресу
   ```bash
   http://localhost:8501/
   ```

## Функции приложения

### **Метрики модели**
Возвращает метрики модели, обученной на тестовых данных, включая:

- **Accuracy**: Точность классификации.
- **Precision**: Точность для классов 0 и 1.
- **Recall**: Полнота для классов 0 и 1.
- **F1-score**: F1-метрика для каждого класса.
- **ROC-AUC**: Площадь под кривой ошибок классификации.

---

### **ROC-кривая**
Генерирует и возвращает изображение **ROC-кривой**, визуализируя производительность модели.

---

### **Кросс-валидация**
Проводит 5-фолд кросс-валидацию на данных и возвращает результаты:

- Матрица ошибок (**Confusion Matrix**) для каждого фолда.
- Значение **ROC-AUC** для каждого фолда.

---

### **ROC-AUC для кросс-валидации**
Генерирует и возвращает график значений **ROC-AUC** для всех фолдов.

---

### **Важность признаков**
Возвращает график важности признаков модели, выполненный с использованием **SHAP**.

---

### **SHAP-анализ**
Генерирует подробную интерпретацию модели с использованием **SHAP** и визуализирует вклад каждого признака в предсказания.

---

### **Предсказания**
Позволяет загрузить **CSV-файл** с данными для выполнения предсказаний.

**Требования к файлу**:
- Должны присутствовать следующие столбцы:
  - `T`, `E`, `C`, `FM`, `Xfm`, `AFM`, `Xafm`.
- **Формат файла**: CSV.

**Для проведения тестов по загрузки файла и выполнения предсказаний приложен архив test_data.zip с двумя файлами**:
- Model_0_1_test.data.csv
- Model_0_1_test.data_2.csv
